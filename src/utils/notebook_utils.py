from copy import deepcopy

import xarray as xr

from utils.data import rename_vars


def _combine_variables_by_level(ds, lev, combine_vars):
    """
    Combine variables in the dataset along a new 'lev' dimension based on their suffix.

    Parameters:
    ds (xarray.Dataset): The input dataset containing variables with suffixes
                        (e.g., thetao_0, so_1).
    lev (xarray.DataArray): lev dataarray containing lev values
    combine_vars (list): List of variable prefixes to combine.

    Returns:
    xarray.Dataset: The dataset with combined variables and a new 'lev' dimension.
    """
    for v in combine_vars:
        levels = lev.values
        level_numbers = [i for i in range(19)]
        sorted_vars = [v + "_" + str(lev) for lev in level_numbers]
        if sorted_vars[0] not in ds.data_vars:
            continue
        combined = xr.concat([ds[var] for var in sorted_vars], dim="lev")
        combined = combined.assign_coords(lev=levels)
        ds[v] = combined
        ds = ds.drop_vars(sorted_vars)
    return ds


def combine_variables_by_level(ds_groundtruth, lev, pred_dict, combine_ground=True):
    """
    Combine variables by level for ground truth and predictions.

    Parameters:
    ds_groundtruth (xarray.Dataset): The ground truth dataset.
    lev (xarray.DataArray): lev dataarray containing lev values
    pred_dict (dict): Dictionary containing prediction datasets.

    Returns:
    xarray.Dataset, dict: Updated ground truth and prediction datasets.
    """
    if combine_ground:
        ds_groundtruth = _combine_variables_by_level(
            ds_groundtruth, lev, ["thetao", "so", "uo", "vo", "mask"]
        )
    for key in pred_dict.keys():
        pred_dict[key]["ds_prediction"] = _combine_variables_by_level(
            pred_dict[key]["ds_prediction"], lev, pred_dict[key]["ls"]
        )
    return ds_groundtruth, pred_dict


def _postprocess_for_plot(ds, areacello, dz, times, wetmask, coords=None):
    """
    Postprocess the dataset to make it compatible with plotting functions.
    """
    ds = ds.transpose("time", "lev", ...)
    ds["time"] = times
    if coords is not None:
        ds = ds.assign_coords(coords)
    if "thetao" in ds.data_vars:
        ds["thetao"] = ds["thetao"].assign_attrs(
            long_name=r"${\theta_O}$", units=r"$\degree C$"
        )
    if "so" in ds.data_vars:
        ds["so"] = ds["so"].assign_attrs(long_name=r"${s}$", units=r"psu")
    if "zos" in ds.data_vars:
        ds["zos"] = ds["zos"].assign_attrs(long_name=r"SSH", units=r"m")
    if "vo" in ds.data_vars:
        ds["vo"] = ds["vo"].assign_attrs(long_name=r"${v}$", units=r"m/s")
    if "uo" in ds.data_vars:
        ds["uo"] = ds["uo"].assign_attrs(long_name=r"${u}$", units=r"m/s")

    ds["lev"] = ds["lev"].assign_attrs(long_name="depth", units="m")
    if "init_time" in ds.coords:
        ds = ds.drop(["init_time", "valid_time"])

    for var in ds.data_vars:
        if "lev" in ds[var].dims:
            ds[var] = ds[var].where(wetmask)
        else:
            ds[var] = ds[var].where(wetmask.isel(lev=0))

    ds["areacello"] = (["lat", "lon"], areacello)
    ds["dz"] = ("lev", dz)
    return ds


def postprocess_for_plot(ds_groundtruth, areacello, dz, pred_dict):
    """
    Postprocess for plotting.

    Parameters:
    ds_groundtruth (xarray.Dataset): The ground truth dataset.
    areacello (xarray.DataArray): areacello dataarray.
    dz (xarray.DataArray): dz dataarray.
    pred_dict (dict): Dictionary containing prediction datasets.

    Returns:
    xarray.Dataset, dict: Postprocessed ground truth and prediction datasets.
    """
    areacello = areacello.values
    dz = dz.data
    times = ds_groundtruth.time

    # Masking land with NaNs
    if "mask" in ds_groundtruth.data_vars:
        wetmask = ds_groundtruth["mask"].isel(time=0)
    else:
        wetmask = ds_groundtruth.wetmask

    ds_groundtruth = _postprocess_for_plot(
        ds_groundtruth, areacello, dz, times, wetmask
    )
    coords = ds_groundtruth.coords

    for key in pred_dict.keys():
        pred_dict[key]["ds_prediction"] = _postprocess_for_plot(
            pred_dict[key]["ds_prediction"],
            areacello,
            dz,
            times,
            wetmask,
            coords=coords,
        )
        # Rename lat and lon to y and x
        pred_dict[key]["ds_prediction"] = pred_dict[key]["ds_prediction"].rename(
            {"lat": "y", "lon": "x"}
        )

    # Rename lat and lon to y and x (This needs to be done in the end!)
    ds_groundtruth = ds_groundtruth.rename({"lat": "y", "lon": "x"})

    return ds_groundtruth, pred_dict


def process_data(data, pred_dict):
    """
    Get plot ready OM4 data.
    """
    ds_groundtruth = rename_vars(data)

    # Renames so further processing is easier
    ds_groundtruth = ds_groundtruth.rename({"lat": "lat_t", "lon": "lon_t"})
    ds_groundtruth = ds_groundtruth.rename({"y": "lat", "x": "lon"})

    # Store ds_prediction
    copy_dict = deepcopy(pred_dict)

    for key in pred_dict.keys():
        ds_prediction = xr.open_zarr(
            pred_dict[key]["path"], chunks={"time": 10, "lat": 180, "lon": 360}
        )

        if ds_prediction.time.size != 600:
            raise Exception(
                "Are you sure your run is complete? Current prediction size: ",
                ds_prediction.time.size,
            )

        assert ds_prediction.time.size == ds_groundtruth.time.size, (
            f"Sizes different for {key}: {ds_prediction.time.size}!={ds_groundtruth.time.size}"
        )
        if "model_path" in ds_prediction.attrs:
            copy_dict[key]["model_path"] = ds_prediction.attrs["model_path"]

        pred_dict[key]["ds_prediction"] = ds_prediction

    ### Combine Variables by level
    ds_groundtruth, pred_dict = combine_variables_by_level(
        ds_groundtruth, ds_groundtruth.lev, pred_dict
    )

    ### Postprocess predictions for plotting
    ds_groundtruth, pred_dict = postprocess_for_plot(
        ds_groundtruth, ds_groundtruth.areacello, ds_groundtruth.dz, pred_dict
    )

    return ds_groundtruth, pred_dict
