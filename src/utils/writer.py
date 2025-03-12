import os
from typing import Dict, Optional

import torch
import xarray as xr
from einops import rearrange

from constants import TensorMap
from utils.data import Normalize
from utils.outputs import InfOutput


class ZarrWriter:
    def __init__(
        self,
        output_dir: str,
        coords: Dict[str, xr.DataArray],
        hist: int,
        model_path: str,
    ):
        self.pred_path = os.path.join(output_dir, "predictions.zarr")
        self.hist = hist
        self.buffer: Optional[torch.Tensor] = None
        self.coords = coords
        self.model_path = model_path

        self.normalize = Normalize.get_instance()
        self.tensor_map = TensorMap.get_instance()

    def record_batch(self, IO: InfOutput):
        pred_tensor = IO.prediction
        pred_tensor = pred_tensor.squeeze(0)
        pred_tensor = rearrange(pred_tensor, "(n c) h w -> n c h w", n=self.hist + 1)
        pred_tensor = self.normalize.unnormalize_tensor_prognostics(pred_tensor)
        if self.buffer is None:
            self.buffer = pred_tensor
        else:
            self.buffer = torch.cat([self.buffer, pred_tensor], dim=0)

    @property
    def buffer_empty(self):
        return self.buffer is None

    def write(self):
        # Write to zarr
        if self.buffer is None:
            raise ValueError("No tensor to write")

        coords = {k: v for k, v in self.coords.items()}
        # TODO: Replace by actual time so I dont need to fix this downstream
        coords["time"] = range(self.buffer.shape[0])
        ds = xr.Dataset(
            data_vars={
                var: (["time", "lat", "lon"], self.buffer[:, i, :, :].cpu().numpy())
                for i, var in enumerate(self.tensor_map.prognostic_vars)
            },
            coords=coords,
        )
        ds.attrs["model_path"] = str(self.model_path)
        ds = ds.chunk({"time": 1, "lat": 180, "lon": 360})
        if os.path.exists(self.pred_path):
            ds.to_zarr(self.pred_path, mode="a", append_dim="time")
        else:
            ds.to_zarr(
                self.pred_path,
                mode="w",
                encoding={var: {"compressor": None} for var in ds.data_vars},
            )

        # Reset
        self.buffer = None
