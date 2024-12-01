import torch
import torch.nn as nn
import numpy as np
from itertools import tee

def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

class CappedGELU(torch.nn.Module):
    """
    Implements a GeLU with capped maximum value.
    """

    def __init__(self, cap_value=1.0, **kwargs):
        """
        :param cap_value: float: value at which to clip activation
        :param kwargs: passed to torch.nn.LeadyReLU
        """
        super().__init__()
        self.add_module("gelu", torch.nn.GELU(**kwargs))
        self.register_buffer("cap", torch.tensor(cap_value, dtype=torch.float32))

    def forward(self, inputs):
        x = self.gelu(inputs)
        x = torch.clamp(x, max=self.cap)
        return x

class BilinearUpsample(torch.nn.Module):
    def __init__(self, upsampling: int = 2, **kwargs):
        super().__init__()
        self.upsampler = torch.nn.Upsample(scale_factor=upsampling, mode="bilinear")

    def forward(self, x):
        return self.upsampler(x)


class AvgPool(torch.nn.Module):
    def __init__(
        self,
        pooling: int = 2,
    ):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(pooling)

    def forward(self, x):
        return self.avgpool(x)


class ConvNeXtBlock(torch.nn.Module):
    """
    A convolution block as reported in https://github.com/CognitiveModeling/dlwp-hpx/blob/main/src/dlwp-hpx/dlwp/model/modules/blocks.py.

    This is a modified version of the actual ConvNextblock which is used in the HealPix paper.

    """

    def __init__(
        self,
        in_channels: int = 300,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        n_layers: int = 1,
        activation: torch.nn.Module = CappedGELU,
        pad="circular",
        upscale_factor: int = 4
    ):
        super().__init__()
        assert kernel_size % 2 != 0, "Cannot use even kernel sizes!"

        self.N_in = in_channels
        self.N_pad = int((kernel_size + (kernel_size - 1) * (dilation - 1) - 1) / 2)
        self.pad = pad

        assert n_layers == 1, "Can only use a single layer here!"

        # 1x1 conv to increase/decrease channel depth if necessary
        if in_channels == out_channels:
            self.skip_module = lambda x: x  # Identity-function required in forward pass
        else:
            self.skip_module = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding="same",
            )

        # Convolution block
        convblock = []
        convblock.append(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=int(in_channels * upscale_factor),
                kernel_size=kernel_size,
                dilation=dilation,
            )
        )
        convblock.append(torch.nn.BatchNorm2d(in_channels * upscale_factor))
        
        convblock.append(activation())
            
        convblock.append(
            torch.nn.Conv2d(
                in_channels=int(in_channels * upscale_factor),
                out_channels=int(in_channels * upscale_factor),
                kernel_size=kernel_size,
                dilation=dilation,
            )
        )
        convblock.append(torch.nn.BatchNorm2d(in_channels * upscale_factor))

        convblock.append(activation())
            
        # Linear postprocessing
        convblock.append(
            torch.nn.Conv2d(
                in_channels=int(in_channels * upscale_factor),
                out_channels=out_channels,
                kernel_size=1,
                padding="same",
            )
        )
        self.convblock = torch.nn.Sequential(*convblock)

    def forward(self, x):
        skip = self.skip_module(x)
        for l in self.convblock:
            if isinstance(l, nn.Conv2d) and l.kernel_size[0] != 1:
                x = torch.nn.functional.pad(
                    x, (self.N_pad, self.N_pad, 0, 0), mode=self.pad
                )
                x = torch.nn.functional.pad(
                    x, (0, 0, self.N_pad, self.N_pad), mode="constant"
                )
            x = l(x)
        return skip + x

class BaseSamudra(torch.nn.Module):
    def __init__(
        self, ch_width, n_out, wet, hist, pred_residuals, last_kernel_size, pad
    ):
        super().__init__()
        assert last_kernel_size % 2 != 0, "Cannot use even kernel sizes!"
        self.N_in = ch_width[0]
        self.N_out = ch_width[-1]
        self.ch_width = ch_width
        self.wet = wet
        self.N_pad = int((last_kernel_size - 1) / 2)
        self.pad = pad
        self.pred_residuals = pred_residuals
        self.hist = hist
        self.input_channels = ch_width[0]
        self.output_channels = n_out

    def forward_once(self, fts):
        raise NotImplementedError()

    def forward(
        self,
        inputs,
        output_only_last=False,
        loss_fn=None,
    ) -> torch.Tensor:
        outputs = []
        loss = None
        N, C, H, W = inputs[0].shape

        for step in range(len(inputs) // 2):
            if step == 0:
                input_tensor = inputs[
                    0
                ]  # For HIST=1, [0->[0in, 1in], 1->[2out, 3out], 2->[2in, 3in], 3->[4out, 5out]
            else:
                inputs_0 = outputs[-1]  # Last output corresponds to input at current time step
                input_tensor = torch.cat(
                    [
                        inputs_0,
                        inputs[2 * step][
                            :, self.output_channels :
                        ],  # boundary conditions
                    ],
                    dim=1,
                )

            assert (
                input_tensor.shape[1] == self.input_channels
            ), f"Input shape is {input_tensor.shape[1]} but should be {self.input_channels}"
            decodings = self.forward_once(input_tensor)
            if self.pred_residuals:
                reshaped = (
                    input_tensor[
                        :,
                        : self.output_channels,
                    ]  # Residuals on last state in input
                    + decodings
                )  # Residual prediction
            else:
                reshaped = decodings  # Absolute prediction

            if loss_fn is not None:
                assert (
                    reshaped.shape == inputs[2 * step + 1].shape
                ), f"Output shape is {reshaped.shape} but should be {inputs[2 * step + 1].shape}"
                if loss is None:
                    loss = loss_fn(
                        reshaped,
                        inputs[2 * step + 1],
                    )
                else:
                    loss += loss_fn(
                        reshaped,
                        inputs[2 * step + 1],
                    )

            outputs.append(reshaped)

        if loss_fn is None:
            if output_only_last:
                res = outputs[-1]
            else:
                res = outputs
            return res

        else:
            return loss

    def inference(
        self, inputs, initial_input=None, num_steps=None, output_only_last=False, device="cuda"
    ) -> torch.Tensor:
        outputs = []
        for step in range(num_steps):
            if step == 0:
                input_tensor = inputs[0][0].to(
                        device=device
                    )  # inputs[0][0] is the input at step 0. For HIST=1 ; 0->[[0, 1], [2, 3]]; 1->[[2, 3], [4, 5]]; 2->[[4, 5], [6, 7]]; 3->[[6, 7], [8, 9]]
                
                if initial_input is not None:
                    input_tensor[:, :self.output_channels] = initial_input
            else:
                inputs_0 = outputs[-1].unsqueeze(
                    0
                )  # Last output corresponds to input at current time step
                input_tensor = torch.cat(
                    [
                        inputs_0,
                        inputs[step][0][
                            :, self.output_channels :
                        ].to(  # boundary conditions
                            device=device
                        ),
                    ],
                    dim=1,
                )

            assert (
                input_tensor.shape[1] == self.input_channels
            ), f"Input shape is {input_tensor.shape[1]} but should be {self.input_channels}"
            decodings = self.forward_once(input_tensor)
            if self.pred_residuals:
                reshaped = input_tensor[
                    0,
                    : self.output_channels,
                ].to(  # Residuals on last state in input
                    device=device
                ) + decodings.squeeze(
                    0
                )
            else:
                reshaped = decodings.squeeze(0)

            outputs.append(reshaped)

        if output_only_last:
            res = outputs[-1]
        else:
            res = outputs

        return res


class Samudra(BaseSamudra):
    def __init__(
        self,
        wet,
        hist,
        core_block=ConvNeXtBlock,
        down_sampling_block=AvgPool,
        up_sampling_block=BilinearUpsample,
        activation=CappedGELU,
        ch_width=[157,200,250,300,400],
        n_out=77,
        dilation=[1, 2, 4, 8],
        n_layers=[1, 1, 1, 1],
        pred_residuals=False,
        last_kernel_size=3,
        pad="circular",
    ):
        super().__init__(
            ch_width, n_out, wet, hist, pred_residuals, last_kernel_size, pad
        )

        layers = []
        for i, (a, b) in enumerate(pairwise(ch_width)):
            layers.append(
                core_block(
                    a,
                    b,
                    dilation=dilation[i],
                    n_layers=n_layers[i],
                    activation=activation,
                    pad=pad,
                )
            )
            layers.append(down_sampling_block())
        layers.append(
            core_block(
                b,
                b,
                dilation=dilation[i],
                n_layers=n_layers[i],
                activation=activation,
                pad=pad,
            )
        )
        layers.append(up_sampling_block(in_channels=b, out_channels=b))
        ch_width.reverse()
        dilation.reverse()
        n_layers.reverse()
        for i, (a, b) in enumerate(pairwise(ch_width[:-1])):
            layers.append(
                core_block(
                    a,
                    b,
                    dilation=dilation[i],
                    n_layers=n_layers[i],
                    activation=activation,
                    pad=pad,
                )
            )
            layers.append(up_sampling_block(in_channels=b, out_channels=b))
        layers.append(
            core_block(
                b,
                b,
                dilation=dilation[i],
                n_layers=n_layers[i],
                activation=activation,
                pad=pad,
            )
        )
        layers.append(torch.nn.Conv2d(b, n_out, last_kernel_size))

        self.layers = nn.ModuleList(layers)
        self.num_steps = int(len(ch_width) - 1)

    def forward_once(self, fts):
        temp = []
        for i in range(self.num_steps):
            temp.append(None)
        count = 0
        for l in self.layers:
            crop = fts.shape[2:]
            if isinstance(l, nn.Conv2d):
                fts = torch.nn.functional.pad(
                    fts, (self.N_pad, self.N_pad, 0, 0), mode=self.pad
                )
                fts = torch.nn.functional.pad(
                    fts, (0, 0, self.N_pad, self.N_pad), mode="constant"
                )
            fts = l(fts)
            if count < self.num_steps:
                if isinstance(l, ConvNeXtBlock):
                    temp[count] = fts
                    count += 1
            elif count >= self.num_steps:
                if isinstance(l, BilinearUpsample):
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
        return torch.mul(fts, self.wet)


def generate_model_rollout(
    N_eval, test_data, model, hist, N_out, N_extra, initial_input=None, train=False
):

    N_test = test_data.size

    mean_out = test_data.out_mean.to_array().to_numpy().reshape(-1)
    std_out = test_data.out_std.to_array().to_numpy().reshape(-1)
    norm_vals = {"m_out": mean_out, "s_out": std_out}

    model.eval()
    model_pred = np.zeros((N_eval, 180, 360, N_out))

    with torch.no_grad():
        outs = model.inference(test_data, initial_input, num_steps=N_eval // (hist + 1))
    for i in range(N_eval // (hist + 1)):
        pred_temp = outs[i]
        pred_temp = torch.nan_to_num(pred_temp)
        pred_temp = torch.clip(pred_temp, min=-1e5, max=1e5)
        C, H, W = pred_temp.shape
        pred_temp = torch.reshape(pred_temp, (hist + 1, C // (hist + 1), H, W))
        model_pred[i * (hist + 1) : (i + 1) * (hist + 1)] = torch.swapaxes(
            torch.swapaxes(pred_temp, 3, 1), 2, 1
        ).cpu()

    if train:
        return model_pred
    else:
        return model_pred * norm_vals["s_out"] + norm_vals["m_out"], outs
