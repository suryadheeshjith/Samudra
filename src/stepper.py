import logging
from typing import Callable, Optional

import torch

from datasets import InferenceDataset, TrainData
from utils.device import using_gpu
from utils.outputs import InfOutput, TrainOutput, ValOutput
from utils.writer import ZarrWriter


class Stepper:
    def __init__(self):
        pass

    @staticmethod
    def train_step(
        model: torch.nn.Module, batch: TrainData, loss_fn: Callable
    ) -> TrainOutput:
        loss_per_channel = model(batch, loss_fn=loss_fn)
        loss = torch.mean(loss_per_channel)
        return TrainOutput(loss, loss_per_channel)

    @staticmethod
    @torch.no_grad()
    def validate_step(
        model: torch.nn.Module, batch: TrainData, loss_fn: Callable
    ) -> ValOutput:
        assert len(batch) == 1  # Assert we are using one step of input and output
        input = batch.get_input(0)
        label = batch.get_label(0)
        model = model.module if using_gpu() else model
        outs = model.forward_once(input)
        loss_per_channel = loss_fn(outs, label)
        loss = torch.mean(loss_per_channel)
        return ValOutput(loss, loss_per_channel, input, label, outs)

    @staticmethod
    @torch.no_grad()
    def inference(
        model: torch.nn.Module,
        dataset: InferenceDataset,
        epoch: int,
        output_dir: Optional[str] = None,
        model_path: Optional[str] = None,
        num_model_steps_forward: int = 200,
        record_every: int = 10,
        save_zarr: bool = False,
        loss_fn: Optional[Callable] = None,
    ) -> Optional[float]:
        if save_zarr:
            if output_dir is None or model_path is None:
                raise ValueError(
                    "output_dir and model_path must be provided if save_zarr is True"
                )
            coords = dataset.get_coords_dict()
            writer = ZarrWriter(
                output_dir,
                coords=coords,
                hist=dataset.hist,
                model_path=model_path,
            )
        else:
            writer = None
        logging.info(f"Inference [epoch {epoch}]: processing initial prognostic.")

        num_model_steps = len(dataset)
        num_steps_list = []

        # If num_model_steps_forward is -1, then we are doing a full forward pass
        if num_model_steps_forward == -1:
            num_steps_list = [num_model_steps]
        else:
            # Windows of partial forward passes
            num_loops = num_model_steps // num_model_steps_forward
            if num_loops > 0:
                num_steps_list = [num_model_steps_forward] * num_loops
                last_model_steps_forward = num_model_steps % num_model_steps_forward
                if last_model_steps_forward > 0:
                    num_steps_list = num_steps_list + [last_model_steps_forward]
            else:
                num_steps_list = [num_model_steps]

        num_loops = len(num_steps_list)
        initial_prognostic = None
        step = 0
        if loss_fn is not None:
            inference_loss = 0.0
        for loop, num_steps in enumerate(num_steps_list):
            logging.info(
                f"Inference [epoch {epoch}]: loop {loop} of {num_loops - 1}. "
                f"Stepping {num_steps} steps forward."
            )
            outs = model.inference(
                dataset,
                initial_prognostic=initial_prognostic,
                steps_completed=step,
                num_steps=num_steps,
                epoch=epoch,
            )
            # Setting initial prognostic for next loop
            initial_prognostic = outs[-1].clone()

            for i in range(num_steps):
                logging.info(
                    f"Inference [epoch {epoch}]: recording output window {i + step} of "
                    f"{num_model_steps - 1}."
                )
                IO = InfOutput(
                    prediction=outs[i],
                    target=dataset.inference_target(step + i).to(
                        outs[i].device
                    ),  # TODO: Pack with input
                    time=dataset.get_input_time(step + i),
                )  # time-dependent aggs dont work, time is incorrect as well
                if loss_fn is not None:
                    loss_per_channel = loss_fn(IO.prediction, IO.target)
                    inference_loss += torch.mean(loss_per_channel).item()
                if writer:
                    writer.record_batch(IO)
                    if (i + 1) % record_every == 0:
                        logging.info(f"Inference [epoch {epoch}]: writing to zarr...")
                        writer.write()

            if writer and not writer.buffer_empty:
                logging.info(f"Inference [epoch {epoch}]: writing to zarr...")
                writer.write()

            step += num_steps

        if loss_fn is not None:
            return inference_loss / num_model_steps
        else:
            return None
