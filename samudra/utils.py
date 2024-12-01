import xarray as xr
import numpy as np
import torch

def post_processor(ds: xr.Dataset, ds_truth: xr.Dataset, ls) -> xr.Dataset:
    """Converts the prediction output to an xarray dataset with the same dimensions/variables as input"""
    # Always run the ds_input_validate in non-deep mode here

    # correct swapped dimensions and warn
    if len(ds.x) == 180 and len(ds.y) == 360:
        ds = ds.rename({"x": "x_i", "y": "y_i"}).rename({"x_i": "y", "y_i": "x"})

    key = list(ds.variables.keys())[0]
    da = ds[key]
    n_lev = 19
    if set(ls) - {"zos"} == set(["uo", "vo", "thetao", "so"]):
        variables = ["uo", "vo", "thetao", "so"]
    elif set(ls) - {"zos"} == set(["thetao", "so"]):
        variables = ["thetao", "so"]

    slices = [slice(i, i + n_lev) for i in range(0, len(variables) * n_lev, n_lev)]
    var_slices = {k: sl for k, sl in zip(variables, slices)}
    variables = {
        k: da.isel(var=sl).rename({"var": "lev"}) for k, sl in var_slices.items()
    }
    if "zos" in ls:
        variables["zos"] = da.isel(var=-1).squeeze()

    ds_out = xr.Dataset(variables)
    for var in ds_out.data_vars:
        if "lev" in ds_out[var].dims:
            ds_out[var] = ds_out[var].where(ds_truth.wetmask)
        else:
            ds_out[var] = ds_out[var].where(ds_truth.wetmask.isel(lev=0))

    ## attach all coordinates from input
    ds_out = ds_out.assign_coords({co: ds_truth[co] for co in ds_truth.coords})
    ds_out.attrs = ds.attrs

    return ds_out


def convert_train_data(ds):
    # Recreate the 'lev' coordinate from the variable names
    lev_values = ds["lev"].values

    # Create an empty dictionary to store the data for each variable
    reconstructed_vars = {
        "vo": [],
        "thetao": [],
        "uo": [],
        "so": []
    }

    # Iterate over the levels and append data to each reconstructed variable
    for lev in lev_values:
        lev_str = str(lev).replace(".", "_")
        
        reconstructed_vars["vo"].append(ds[f"vo_lev_{lev_str}"].expand_dims(dim={"lev": [lev]}))
        reconstructed_vars["thetao"].append(ds[f"thetao_lev_{lev_str}"].expand_dims(dim={"lev": [lev]}))
        reconstructed_vars["uo"].append(ds[f"uo_lev_{lev_str}"].expand_dims(dim={"lev": [lev]}))
        reconstructed_vars["so"].append(ds[f"so_lev_{lev_str}"].expand_dims(dim={"lev": [lev]}))

    # Concatenate along the 'lev' dimension
    for var_name in reconstructed_vars:
        ds[var_name] = xr.concat(reconstructed_vars[var_name], dim="lev")

    # Drop the individual lev variables
    vars_to_drop = [var for var in ds.data_vars if any(var.startswith(prefix) for prefix in ["vo_lev_", "thetao_lev_", "uo_lev_", "so_lev_"])]
    ds = ds.drop_vars(vars_to_drop)

    return ds


def extract_wet(wet_zarr, outputs, hist):
    depths = [var.split('lev_')[-1].replace('_', '.') for var in outputs]
    if 'zos' in depths:
        zos_index = depths.index('zos')
        depths[zos_index] = str(wet_zarr.lev.values[0])
        assert depths[zos_index] == '2.5'
    depths = [float(depth) for depth in depths]
    wet = wet_zarr.sel(lev=depths)
    wet = torch.from_numpy(wet.to_numpy().squeeze())
    wet = torch.concat([wet] * (hist + 1), dim=0)
    return wet
