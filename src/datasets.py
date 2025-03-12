import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import xarray as xr
from einops import rearrange
from torch.utils.data import Dataset

from utils.data import Normalize
from utils.device import get_device, using_gpu


class InferenceDataset(Dataset):
    """This class is used for inference rollouts.

    It creates rolling indices to keep track of histories/past states.
    For example,
    Hist=0 ; 0->[0, 1]; 1->[1, 2]; 2->[2, 3]; 3->[3, 4]
    Hist=1 ; 0->[[0, 1], [2, 3]]; 1->[[2, 3], [4, 5]];
            2->[[4, 5], [6, 7]]; 3->[[6, 7], [8, 9]]
    Hist=2 ; 0->[[0, 1, 2], [3, 4, 5]];
            1->[[3, 4, 5], [6, 7, 8]];
            2->[[6, 7, 8], [9, 10, 11]];
            3->[[9, 10, 11], [12, 13, 14]]
    """

    def __init__(
        self,
        data,
        prognostic_vars,
        boundary_vars,
        wet,
        wet_surface,
        hist,
    ):
        super().__init__()
        self.device = get_device()

        self.hist = hist

        self._prognostic_data = data[prognostic_vars]
        self.prognostic_channels = (hist + 1) * len(prognostic_vars)
        self._boundary_data = data[boundary_vars]

        time_indices = np.arange(data.time.size)
        indices = xr.DataArray(
            time_indices,
            dims=["time"],
            coords={"time": time_indices},
        )
        total_steps = 2 * self.hist + 1
        rolling_indices = indices.rolling(
            time=len(time_indices) - total_steps, center=False
        ).construct("window_dim")
        rolling_indices = rolling_indices.transpose("window_dim", "time").isel(
            time=slice(len(time_indices) - total_steps - 1, None)
        )  # Remove first few null indices
        self.rolling_indices = rolling_indices.isel(
            window_dim=slice(0, None, self.hist + 1)
        )  # Skip indices based on history
        self.rolling_indices = self.rolling_indices.astype(int)

        logging.info(
            "Inference will use input at time {0} and produce output at {1}".format(
                data.time.values[0],
                data.time.values[self.hist + 1],
            )
        )

        self.wet = wet.bool()
        self.wet_surface = wet_surface.bool()
        self.size = len(self.rolling_indices)

        if using_gpu():
            self.wet = self.wet.pin_memory()
            self.wet_surface = self.wet_surface.pin_memory()

    def __len__(self):
        return self.size

    @property
    def initial_prognostic(self):
        data = self.__getitem__(0)[0]
        return data[:, : self.prognostic_channels]

    def inference_target(self, step: int):
        return self.__getitem__(step)[1]

    def get_initial_input(self):
        data = self.__getitem__(0)[0]
        return data

    # TODO: This is a placeholder for now since time returned is incorrect
    def get_input_time(self, step: int):
        return self._prognostic_data.time[step]

    def merge_prognostic_and_boundary(self, prognostic: torch.Tensor, step: int):
        x_index = self._get_x_index(step)
        boundary = self._get_boundary(x_index).to(prognostic.device)
        data = torch.cat((prognostic, boundary), dim=1)
        return data

    def __getitem__(self, idx):
        x_index = self._get_x_index(idx)
        data_in = self._get_prognostic(x_index)
        data_in_boundary = self._get_boundary(x_index)
        data_in = torch.cat((data_in, data_in_boundary), dim=1)
        label = self._get_label(x_index)
        return (data_in, label)

    def _get_x_index(self, idx):
        if isinstance(idx, slice):
            if idx.start < 0 or idx.stop < 0 or idx.step < 0:
                raise IndexError("Sorry, negative indexing is not supported!")
            elif idx.start >= self.size or idx.stop >= self.size:
                raise IndexError(f"Index {idx} out of range with size {self.size}")
            elif idx.start is None and idx.stop is None:
                idx = slice(0, self.size, idx.step)
            elif idx.start is None:
                idx = slice(0, idx.stop, idx.step)
            elif idx.stop is None:
                idx = slice(idx.start, self.size, idx.step)
        elif isinstance(idx, int):
            if idx < 0:
                raise IndexError("Sorry, negative indexing is not supported!")
            elif idx >= self.size:
                raise IndexError(f"Index {idx} out of range with size {self.size}")
            idx = slice(idx, idx + 1, 1)

        rolling_idx = self.rolling_indices.isel(window_dim=idx)
        x_index = xr.Variable(["window_dim", "time"], rolling_idx)

        return x_index

    def _get_prognostic(self, x_index):
        data_in = self._prognostic_data.isel(time=x_index).isel(
            time=slice(None, self.hist + 1)
        )
        data_in = Normalize.get_instance().normalize_prognostics(data_in)
        data_in = (
            data_in.to_array()
            .transpose("window_dim", "time", "variable", "lat", "lon")
            .to_numpy()
        )
        data_in = rearrange(
            data_in,
            "window_dim time variable lat lon -> window_dim (time variable) lat lon",
        )
        data_in = torch.from_numpy(data_in).float()
        data_in = torch.where(self.wet, data_in, 0.0)
        return data_in

    def _get_boundary(self, x_index):
        """
        This function returns the boundary condition for the current time step.

        With hist > 0, the boundary condition considered is always the last step of
        the input.
        """
        data_in_boundary = self._boundary_data.isel(time=x_index).isel(time=self.hist)
        data_in_boundary = Normalize.get_instance().normalize_boundary(data_in_boundary)
        data_in_boundary = (
            data_in_boundary.to_array()
            .transpose("window_dim", "variable", "lat", "lon")
            .to_numpy()
        )
        data_in_boundary = torch.from_numpy(data_in_boundary).float()
        data_in_boundary = torch.where(self.wet_surface, data_in_boundary, 0.0)
        return data_in_boundary

    def _get_label(self, x_index):
        label = self._prognostic_data.isel(time=x_index).isel(
            time=slice(self.hist + 1, None)
        )
        label = Normalize.get_instance().normalize_prognostics(label)
        label = (
            label.to_array()
            .transpose("window_dim", "time", "variable", "lat", "lon")
            .to_numpy()
        )
        label = rearrange(
            label,
            "window_dim time variable lat lon -> window_dim (time variable) lat lon",
        )
        label = torch.from_numpy(label).float()
        # label = label * self.wet
        label = torch.where(self.wet, label, 0.0)
        return label

    def get_coords_dict(self):
        return {co: self._prognostic_data[co] for co in self._prognostic_data.coords}


class InferenceDatasets(Dataset):
    def __init__(self, datasets: List[InferenceDataset], lengths: List[int]):
        self.datasets = datasets
        self.lengths = lengths

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        return (self.datasets[idx], self.lengths[idx])


class TrainData:
    def __init__(self, prognostic_channels: int):
        self.td_dict: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.prognostic_channels = prognostic_channels
        self.steps = 0

    def insert(self, input: torch.Tensor, label: torch.Tensor):
        self.td_dict[self.steps] = (input, label)
        self.steps += 1

    def get_initial_input(self):
        return self.td_dict[0][0]

    def get_input(self, step: int):
        return self.td_dict[step][0]

    def get_label(self, step: int):
        return self.td_dict[step][1]

    def merge_prognostic_and_boundary(self, prognostic: torch.Tensor, step: int):
        input, _ = self.td_dict[step]
        input[:, : self.prognostic_channels] = prognostic
        return input

    def __getitem__(self, step: int):
        return self.td_dict[step]

    def __len__(self):
        return self.steps

    def to(self, device: torch.device):
        for step in self.td_dict:
            self.td_dict[step] = (
                self.td_dict[step][0].to(device),
                self.td_dict[step][1].to(device),
            )


class TrainDataset(Dataset):
    """
    This class is used for training and validation.

    It creates rolling indices to keep track of histories/past states. But different
    from InferenceDataset, as it creates rolling indices based on stride. By default,
    the sliding window / stride is 1.

    We make use of TrainData class to store a single sample.

    For example,
    Hist=0 ; TD: step=0->[0, 1]; step=1->[1, 2]; step=2->[2, 3]; step=3->[3, 4]
    Hist=1 ; TD: step=0->[[0, 1], [2, 3]]; step=1->[[2, 3], [4, 5]];
            step=2->[[4, 5], [6, 7]]; step=3->[[6, 7], [8, 9]]
    Hist=2 ; TD: step=0->[[0, 1, 2], [3, 4, 5]];
            step=1->[[3, 4, 5], [6, 7, 8]];
            step=2->[[6, 7, 8], [9, 10, 11]];
            step=3->[[9, 10, 11], [12, 13, 14]]
    """

    def __init__(
        self,
        data,
        prognostic_vars,
        boundary_vars,
        wet,
        wet_surface,
        hist,
        steps,
        stride=1,
    ):
        super().__init__()
        self.device = get_device()

        self.hist = hist
        self.steps = steps
        self.stride = stride

        self._prognostic_data = data[prognostic_vars]
        self.prognostic_channels = (hist + 1) * len(prognostic_vars)
        self._boundary_data = data[boundary_vars]

        # This class will be used only for training
        total_steps = 2 * self.hist + 2

        # Calculate the number of windows
        num_windows = data.time.size - (total_steps - 1) * self.stride

        # Create base indices
        indices = np.arange(num_windows)
        indices_da = xr.DataArray(indices, dims=["window_dim"])

        # Create window dimension
        window_dim = xr.DataArray(np.arange(total_steps), dims=["time"])

        # Construct rolling indices
        self.rolling_indices = indices_da + stride * window_dim

        self.wet = wet.bool()
        self.wet_surface = wet_surface.bool()

        self.size = (
            data.time.size
            - self.steps * (self.hist + 1) * self.stride
            - self.hist * self.stride
        )

        if using_gpu():
            self.wet = self.wet.pin_memory()
            self.wet_surface = self.wet_surface.pin_memory()

        # Normalize
        self.normalize = Normalize.get_instance()
        self._prognostic_data = self.normalize.normalize_prognostics(
            self._prognostic_data
        )
        self._boundary_data = self.normalize.normalize_boundary(self._boundary_data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        TD = TrainData(self.prognostic_channels)
        prev_rolling_idx = None
        for step in range(self.steps):
            x_index = self._get_x_index(idx, step, prev_rolling_idx)
            data_in = self._get_input(x_index)
            data_in_boundary = self._get_boundary(x_index)
            data_in = torch.cat((data_in, data_in_boundary), dim=1).squeeze()

            label = self._get_label(x_index)

            TD.insert(
                input=data_in,
                label=label,
            )

        return TD

    def _get_x_index(self, idx, step, prev_rolling_idx):
        assert isinstance(idx, int)
        if idx < 0:
            raise IndexError("Sorry, negative indexing is not supported!")
        elif idx >= len(self):
            raise IndexError("Index out of range")

        start = idx + step * (self.hist + 1) * self.stride
        end = start + 1
        idx_slice = slice(
            start, end
        )  # Create a slice for similar indexing as in InferenceDataset
        rolling_idx = self.rolling_indices.isel(window_dim=idx_slice)

        x_index = xr.Variable(["window_dim", "time"], rolling_idx)
        return x_index

    def _get_input(self, x_index):
        data_in = self._prognostic_data.isel(time=x_index).isel(
            time=slice(None, self.hist + 1)
        )
        data_in = (
            data_in.to_array()
            .transpose("window_dim", "time", "variable", "lat", "lon")
            .to_numpy()
        )
        data_in = rearrange(
            data_in,
            "window_dim time variable lat lon -> \
                window_dim (time variable) lat lon",
        )
        data_in = torch.from_numpy(data_in).float()
        data_in = torch.where(self.wet, data_in, 0.0)
        return data_in

    def _get_boundary(self, x_index):
        """
        This function returns the boundary condition for the current time step.

        With hist > 0, the boundary condition considered is always the last step of
        the input.
        """
        data_in_boundary = self._boundary_data.isel(time=x_index).isel(time=self.hist)
        data_in_boundary = (
            data_in_boundary.to_array()
            .transpose("window_dim", "variable", "lat", "lon")
            .to_numpy()
        )
        data_in_boundary = torch.from_numpy(data_in_boundary).float()
        data_in_boundary = torch.where(self.wet_surface, data_in_boundary, 0.0)
        return data_in_boundary

    def _get_label(self, x_index):
        label = self._prognostic_data.isel(time=x_index).isel(
            time=slice(self.hist + 1, None)
        )
        label = (
            label.to_array()
            .transpose("window_dim", "time", "variable", "lat", "lon")
            .to_numpy()
        )
        label = rearrange(
            label,
            "window_dim time variable lat lon ->\
                window_dim (time variable) lat lon",
        ).squeeze()
        label = torch.from_numpy(label).float()
        label = torch.where(self.wet, label, 0.0)
        return label
