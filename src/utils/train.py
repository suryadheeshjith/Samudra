import os
from itertools import tee
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F

from datasets import InferenceDataset, TrainData


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def decomposed_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Standard MSE loss computed per channel."""
    return F.mse_loss(pred, target, reduction="none").mean(dim=(0, 2, 3))


def collate_train_data(data: Sequence[TrainData]) -> TrainData:
    prognostic_channels = data[0].prognostic_channels
    steps = len(data[0])

    batched_data = TrainData(prognostic_channels)

    for step in range(steps):
        input = torch.stack([d.get_input(step) for d in data])
        label = torch.stack([d.get_label(step) for d in data])
        batched_data.insert(input, label)

    return batched_data


def collate_inference_data(
    data: Sequence[InferenceDataset],
) -> Tuple[InferenceDataset, int]:
    # TODO: There is probably a better way to do inference batching
    assert len(data) == 1, "Inference batch size must be 1"
    return data[0][0], data[0][1]


class CheckpointPaths:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir

    @property
    def latest_checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, "ckpt.pt")

    def latest_checkpoint_path_with_epoch(self, epoch: int) -> str:
        return os.path.join(self.checkpoint_dir, f"ckpt_{epoch}.pt")

    @property
    def best_inference_checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, "best_inference_ckpt.pt")

    @property
    def best_validation_checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, "best_validation_ckpt.pt")
