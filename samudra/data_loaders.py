import xarray as xr
import numpy as np
import torch
from einops import rearrange


class Test(torch.utils.data.Dataset):

    def __init__(
        self,
        data,
        inputs_str,
        extra_in_str,
        outputs_str,
        wet,
        data_mean,
        data_std,
        n_samples,
        hist,
        ind_start,
        long_rollout,
        device="cuda",
    ):
        super().__init__()
        interval = 1
        lag = 1
        self.device = device

        self.size = n_samples
        self.lag = lag
        self.interval = interval
        self.hist = hist
        self.ind_start = ind_start

        assert self.interval == 1
        assert self.lag == 1

        data = data.isel(time=slice(self.ind_start, None))
        self.inputs = data[inputs_str + extra_in_str]
        self.outputs = data[outputs_str]
        self.inputs_no_extra = data[inputs_str]
        self.extras = data[extra_in_str]

        # This class will be used only for validation and rollouts
        # Rolling indices to keep track of histories/past states:
        # HIST=0 ; 0->[0, 1]; 1->[1, 2]; 2->[2, 3]; 3->[3, 4]
        # HIST=1 ; 0->[[0, 1], [2, 3]]; 1->[[2, 3], [4, 5]]; 2->[[4, 5], [6, 7]]; 3->[[6, 7], [8, 9]]
        # HIST=2 ; 0->[[0, 1, 2], [3, 4, 5]]; 1->[[3, 4, 5], [6, 7, 8]]; 2->[[6, 7, 8], [9, 10, 11]]; 3->[[9, 10, 11], [12, 13, 14]]
        indices = xr.DataArray(
            np.arange(data.time.size),
            dims=["time"],
            coords={"time": data.time},
        )
        total_steps = 2 * self.hist + 1
        rolling_indices = (
            indices.rolling(time=len(data.time) - total_steps, center=False)
            .construct("window_dim")
            .astype(int)
        )
        rolling_indices = rolling_indices.transpose("window_dim", "time").isel(
            time=slice(len(data.time) - total_steps - 1, None)
        )  # Remove first few null indices
        self.rolling_indices = rolling_indices.isel(
            window_dim=slice(0, None, self.hist + 1)
        )  # Skip indices based on history

        if long_rollout:
            window0 = self.rolling_indices.isel(window_dim=0)
            print(
                "Long rollout will begin with input and produce output from time index {0} and {1} respectively".format(
                    window0.isel(time=0).values + ind_start,
                    window0.isel(time=self.hist + 1).values + ind_start,
                )
            )

        self.in_mean = data_mean[inputs_str + extra_in_str]
        self.in_std = data_std[inputs_str + extra_in_str]
        self.out_mean = data_mean[outputs_str]
        self.out_std = data_std[outputs_str]
        self.inputs_no_extra_mean = data_mean[inputs_str]
        self.inputs_no_extra_std = data_std[inputs_str]
        self.extras_mean = data_mean[extra_in_str]
        self.extras_std = data_std[extra_in_str]

        self.wet = wet

    def set_device(self, device):
        self.device = device

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if type(idx) == slice:
            if idx.start == None and idx.stop == None:
                idx = slice(0, self.size, idx.step)
            elif idx.start == None:
                idx = slice(0, idx.stop, idx.step)
            elif idx.stop == None:
                idx = slice(idx.start, self.size, idx.step)
        elif type(idx) == int:
            idx = slice(idx, idx + 1, 1)

        rolling_idx = self.rolling_indices.isel(window_dim=idx)
        x_index = xr.Variable(
            ["window_dim", "time"], rolling_idx
        )
        print("Out: ", (self.ind_start + x_index.isel(time=slice(self.hist + 1, None))).values, end=' ')
        data_in = self.inputs_no_extra.isel(time=x_index).isel(
            time=slice(None, self.hist + 1)
        )
        data_in = (
            (data_in - self.inputs_no_extra_mean) / self.inputs_no_extra_std
        ).fillna(0)
        data_in = (
            data_in.to_array()
            .transpose("window_dim", "time", "variable", "y", "x")
            .to_numpy()
        )
        data_in = rearrange(
            data_in, "window_dim time variable y x -> window_dim (time variable) y x"
        )
        if len(self.extras.variables) != 0:
            data_in_boundary = self.extras.isel(time=x_index).isel(time=self.hist)
            data_in_boundary = (
                (data_in_boundary - self.extras_mean) / self.extras_std
            ).fillna(0)
            data_in_boundary = (
                data_in_boundary.to_array()
                .transpose("window_dim", "variable", "y", "x")
                .to_numpy()
            )
            data_in = np.concatenate((data_in, data_in_boundary), axis=1)

        label = self.outputs.isel(time=x_index).isel(time=slice(self.hist + 1, None))
        label = ((label - self.out_mean) / self.out_std).fillna(0)
        label = (
            label.to_array()
            .transpose("window_dim", "time", "variable", "y", "x")
            .to_numpy()
        )
        label = rearrange(
            label, "window_dim time variable y x -> window_dim (time variable) y x"
        )

        items = (torch.from_numpy(data_in).float(), torch.from_numpy(label).float())

        return items
    
class Train(torch.utils.data.Dataset):

    def __init__(
        self,
        data,
        inputs_str,
        extra_in_str,
        outputs_str,
        wet,
        data_mean,
        data_std,
        n_samples,
        hist,
        steps,
        stride=1,
        device="cuda",
    ):
        super().__init__()
        self.device = device

        self.size = n_samples
        self.lag = lag
        self.interval = interval
        self.hist = hist
        self.steps = steps
        self.stride = stride

        lag = 1
        interval = 1

        self.inputs = data[inputs_str + extra_in_str]
        self.outputs = data[outputs_str]
        self.inputs_no_extra = data[inputs_str]
        self.extras = data[extra_in_str]

        # This class will be used only for training
        total_steps = (2 * self.hist + 2)

        # Calculate the number of windows
        num_windows = data.time.size - (total_steps - 1) * self.stride

        # Create base indices
        indices = np.arange(num_windows)
        indices_da = xr.DataArray(
            indices,
            dims=["window_dim"]
        )

        # Create window dimension
        window_dim = xr.DataArray(
            np.arange(total_steps),
            dims=["time"]
        )

        # Construct rolling indices
        self.rolling_indices = indices_da + stride * window_dim

        self.inputs_no_extra_mean = data_mean[inputs_str]
        self.inputs_no_extra_std = data_std[inputs_str]
        self.extras_mean = data_mean[extra_in_str]
        self.extras_std = data_std[extra_in_str]
        self.in_mean = data_mean[inputs_str + extra_in_str]
        self.in_std = data_std[inputs_str + extra_in_str]

        self.out_mean = data_mean[outputs_str]
        self.out_std = data_std[outputs_str]

        self.wet = wet

    def set_device(self, device):
        self.device = device

    def __len__(self):
        return (self.size - self.steps * (self.hist + 1) * self.stride - self.hist * self.stride)
    
    def __getitem__(self, idx):
        outputs = []
        
        if idx >= len(self):
            raise IndexError("Index out of range")

        assert type(idx) == int
        prev_rolling_idx = None
        for step in range(self.steps):
            start = idx + step * (self.hist + 1) * self.stride
            end = start + 1
            idx_slice = slice(
                start, end, self.interval
            )  # Create a slice for similar indexing as in data_CNN_Disk
            rolling_idx = self.rolling_indices.isel(window_dim=idx_slice)
            if prev_rolling_idx is not None:
                assert (prev_rolling_idx.isel(time=slice(self.hist+1, None)) - rolling_idx.isel(time=slice(0, self.hist+1))).sum() == 0 # Prev output = Cur Input 
                assert (rolling_idx.diff("time") == self.stride).all() # Stride is maintained
                assert rolling_idx.isel(time=-1) < self.size # Last index check
            x_index = xr.Variable(
                ["window_dim", "time"], rolling_idx
            )
            data_in = self.inputs_no_extra.isel(time=x_index).isel(
                time=slice(None, self.hist + 1)
            )
            data_in = (
                (data_in - self.inputs_no_extra_mean) / self.inputs_no_extra_std
            ).fillna(0)
            data_in = (
                data_in.to_array()
                .transpose("window_dim", "time", "variable", "y", "x")
                .to_numpy()
            )
            data_in = rearrange(
                data_in,
                "window_dim time variable y x -> window_dim (time variable) y x",
            )
            data_in_boundary = self.extras.isel(time=x_index).isel(time=self.hist)
            data_in_boundary = (
                (data_in_boundary - self.extras_mean) / self.extras_std
            ).fillna(0)
            data_in_boundary = (
                data_in_boundary.to_array()
                .transpose("window_dim", "variable", "y", "x")
                .to_numpy()
            )
            data_in = np.concatenate((data_in, data_in_boundary), axis=1).squeeze()

            label = self.outputs.isel(time=x_index).isel(
                time=slice(self.hist + 1, None)
            )
            label = ((label - self.out_mean) / self.out_std).fillna(0)
            label = (
                label.to_array()
                .transpose("window_dim", "time", "variable", "y", "x")
                .to_numpy()
            )
            label = rearrange(
                label, "window_dim time variable y x -> window_dim (time variable) y x"
            ).squeeze()

            outputs.append(torch.from_numpy(data_in).float())
            outputs.append(torch.from_numpy(label).float())

        return outputs

