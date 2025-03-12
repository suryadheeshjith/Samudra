import logging
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from datasets import InferenceDataset, TrainData
from models.blocks import BilinearUpsample, ConvNeXtBlock, TransposedConvUpsample
from models.factory import (
    create_block,
    create_downsample,
    create_upsample,
    get_activation_cl,
)
from utils.device import get_device
from utils.train import pairwise


class BaseModel(torch.nn.Module):
    """
    Base class that contains generic functionality for forward pass and inference.
    """

    def __init__(self, ch_width, n_out, wet, hist, last_kernel_size, pad) -> None:
        super().__init__()
        assert last_kernel_size % 2 != 0, "Cannot use even kernel sizes!"
        self.N_in = ch_width[0]
        self.N_out = ch_width[-1]
        self.ch_width = ch_width
        self.wet = wet.bool()
        self.N_pad = int((last_kernel_size - 1) / 2)
        self.pad = pad
        self.hist = hist
        self.input_channels = ch_width[0]
        self.prognostic_channels = n_out

    def forward_once(self, fts):
        raise NotImplementedError()

    def forward(
        self,
        train_data: TrainData,
        loss_fn=None,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        outputs: list[torch.Tensor] = []
        loss = torch.tensor(torch.nan)
        for step in range(len(train_data)):
            if step == 0:
                input_tensor = train_data.get_initial_input()
            else:
                input_tensor = train_data.merge_prognostic_and_boundary(
                    prognostic=outputs[-1], step=step
                )

            decodings = self.forward_once(input_tensor)
            pred = decodings  # Absolute prediction

            if loss_fn is not None:
                if torch.isnan(loss).all():
                    loss = loss_fn(
                        pred,
                        train_data.get_label(step),
                    )
                else:
                    loss += loss_fn(
                        pred,
                        train_data.get_label(step),
                    )

            outputs.append(pred)

        if loss_fn is None:
            return outputs
        else:
            return loss

    def inference(
        self,
        dataset: InferenceDataset,
        initial_prognostic=None,
        steps_completed=0,
        num_steps=None,
        epoch=None,
    ) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        for step in range(num_steps):
            logging.info(
                f"Inference [epoch {epoch}]: Rollout step {steps_completed + step} "
                f"of {steps_completed + num_steps - 1}."
            )
            if step == 0 and steps_completed == 0:
                input_tensor = dataset.get_initial_input().to(device=get_device())

            elif step == 0 and steps_completed > 0:
                input_tensor = dataset.merge_prognostic_and_boundary(
                    prognostic=initial_prognostic,
                    step=steps_completed,
                )
            else:
                input_tensor = dataset.merge_prognostic_and_boundary(
                    prognostic=outputs[-1],
                    step=steps_completed + step,
                )

            decodings = self.forward_once(input_tensor)
            pred = decodings

            outputs.append(pred)

        return outputs


class Samudra(BaseModel):
    """
    Samudra model.

    This is the main model class for the Samudra model.
    """

    def __init__(self, config, hist, wet):
        super().__init__(
            ch_width=config.ch_width,
            n_out=config.n_out,
            wet=wet,
            hist=hist,
            last_kernel_size=config.last_kernel_size,
            pad=config.pad,
        )

        # Get activation class
        activation = get_activation_cl(config.core_block.activation)

        # Create local copies of config lists that will be reversed
        ch_width = config.ch_width.copy()
        dilation = config.dilation.copy()
        n_layers = config.n_layers.copy()

        # going down
        layers = []
        for i, (a, b) in enumerate(pairwise(ch_width)):
            # Core block
            layers.append(
                create_block(
                    config.core_block.block_type,
                    in_channels=a,
                    out_channels=b,
                    kernel_size=config.core_block.kernel_size,
                    dilation=dilation[i],
                    n_layers=n_layers[i],
                    activation=activation,
                    pad=config.pad,
                    upscale_factor=config.core_block.upscale_factor,
                    norm=config.core_block.norm,
                )
            )
            # Down sampling block
            layers.append(create_downsample(config.down_sampling_block))

        # Middle block
        layers.append(
            create_block(
                config.core_block.block_type,
                in_channels=b,
                out_channels=b,
                kernel_size=config.core_block.kernel_size,
                dilation=dilation[i],
                n_layers=n_layers[i],
                activation=activation,
                pad=config.pad,
                upscale_factor=config.core_block.upscale_factor,
                norm=config.core_block.norm,
            )
        )

        # First upsampling
        layers.append(
            create_upsample(config.up_sampling_block, in_channels=b, out_channels=b)
        )

        # Reverse for upsampling path
        ch_width.reverse()
        dilation.reverse()
        n_layers.reverse()

        # going up
        for i, (a, b) in enumerate(pairwise(ch_width[:-1])):
            layers.append(
                create_block(
                    config.core_block.block_type,
                    in_channels=a,
                    out_channels=b,
                    kernel_size=config.core_block.kernel_size,
                    dilation=dilation[i],
                    n_layers=n_layers[i],
                    activation=activation,
                    pad=config.pad,
                    upscale_factor=config.core_block.upscale_factor,
                    norm=config.core_block.norm,
                )
            )
            layers.append(
                create_upsample(config.up_sampling_block, in_channels=b, out_channels=b)
            )

        # Final conv block
        layers.append(
            create_block(
                config.core_block.block_type,
                in_channels=b,
                out_channels=b,
                kernel_size=config.core_block.kernel_size,
                dilation=dilation[i],
                n_layers=n_layers[i],
                activation=activation,
                pad=config.pad,
                upscale_factor=config.core_block.upscale_factor,
                norm=config.core_block.norm,
            )
        )

        # Final output conv
        layers.append(nn.Conv2d(b, config.n_out, config.last_kernel_size))

        self.layers = nn.ModuleList(layers)
        self.num_steps = int(len(config.ch_width) - 1)

    def forward_once(self, fts):
        temp: list[torch.Tensor] = []
        for i in range(self.num_steps):
            temp.append(torch.zeros_like(fts))
        count = 0
        for layer in self.layers:
            crop = fts.shape[2:]
            if isinstance(layer, nn.Conv2d):
                fts = torch.nn.functional.pad(
                    fts, (self.N_pad, self.N_pad, 0, 0), mode=self.pad
                )
                fts = torch.nn.functional.pad(
                    fts, (0, 0, self.N_pad, self.N_pad), mode="constant"
                )
            fts = layer(fts)
            if count < self.num_steps:
                if isinstance(layer, ConvNeXtBlock):
                    temp[count] = fts
                    count += 1
            elif count >= self.num_steps:
                if isinstance(layer, BilinearUpsample) or isinstance(
                    layer, TransposedConvUpsample
                ):
                    crop = np.array(fts.shape[2:])
                    shape = np.array(
                        temp[int(2 * self.num_steps - count - 1)].shape[2:]
                    )
                    pads = shape - crop
                    pads = [
                        pads[1] // 2,
                        pads[1] - pads[1] // 2,
                        pads[0] // 2,
                        pads[0] - pads[0] // 2,
                    ]
                    fts = nn.functional.pad(fts, pads)
                    fts += temp[int(2 * self.num_steps - count - 1)]
                    count += 1
        return torch.where(self.wet, fts, 0.0)
