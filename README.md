# Samudra: A Global Ocean Emulator

This repository contains the implementation of Samudra, a global ocean emulator described in ["Samudra: An AI Global Ocean Emulator for Climate"](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024GL114318). Samudra efficiently emulates the ocean component of a state-of-the-art climate model, accurately reproducing key ocean variables including sea surface height, horizontal velocities, temperature, and salinity, across their full depth.

<p align="center">
  <img src="/assets/globe.gif" >
</p>

## Overview

Samudra reproduces the 3D ocean temperature structure and variability of the OM4 climate model with high fidelity, while also demonstrating stable long-term performance under realistic, time-varying forcing conditions. It achieves a significant speedup as well—rolling out a 100-year simulation is approximately 150 times faster than the original model.

<p align="center">
  <img src="/assets/enso.gif" >
</p>

## Installation

1. Clone the repository:
```bash
git clone https://github.com/suryadheeshjith/Samudra.git
cd Samudra
```

2. Set up the environment using either method:

Using conda:
```bash
conda env create -f environment.yml
conda activate samudra
```

Using [`uv`](https://docs.astral.sh/uv/):
```bash
uv sync
source .venv/bin/activate
```

## Usage

To train or rollout Samudra, download the OM4 data and statistics referenced in the [OM4 Data](#om4-data) section. You can also substitute your own data, provided it aligns with the same format. Note that mean and standard deviation files are precomputed for training and model rollouts; if you opt to use your own data, you will need to compute these statistics yourself.

Should you wish to evaluate Samudra directly, you may use the pre-trained model weights discussed in the [Trained Model Weights](#trained-model-weights) section.

### Training
A default training configuration is provided in the file configs/train_samudra_om4.yaml. Update all fields marked with # FILL IN to reference your local data paths and files for OM4 data and statistics.

> Note: Ensure your environment is activated before training.
```bash
# Train a new model
torchrun src/train.py --config path/to/train_config.yaml
```

### Rollout
A default rollout configuration is provided in configs/rollout_samudra_om4.yaml. Update all fields marked with # FILL IN to reference your local data paths and files for OM4 data and statistics.

> Note: Ensure your environment is activated before training.
```bash
# Produce a rollout from a trained model (and optionally save the result)
python src/rollout.py --config path/to/rollout_config.yaml --ckpt_path path/to/checkpoint.pt --save_zarr
```

> Note: For both training and rollout, you may change the experiment name in the config files or use `--sub_name` argument in the command line to specify a different name for the output directory.

## OM4 Data
The OM4 data and corresponding statistics are publicly available as Zarr files via our hosted pod.

```python
import xarray as xr

# Download statistics
means = xr.open_dataset("https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/OM4_means", engine='zarr', chunks={})
means.to_zarr("local/path/to/data-dir/means.zarr")

stds = xr.open_dataset("https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/OM4_stds", engine='zarr', chunks={})
stds.to_zarr("local/path/to/data-dir/stds.zarr")
```

Downloading the entire dataset (~70 GB) can be time-consuming, depending on your network speed. If you only need to evaluate Samudra on the test set, a reduced dataset (~12 GB) is sufficient.

```python
import xarray as xr

# Download the entire data
data = xr.open_dataset("https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/OM4", engine='zarr', chunks={})

# For training ~ 70GB
data.to_zarr("local/path/to/data-dir/data.zarr")

# (OR) For evaluation ~ 12GB
data.sel(time=slice("2014-10-10", "2022-12-24")).to_zarr("local/path/to/data-dir/data.zarr")
```

## Trained Model Weights
Pre-trained weights for Samudra are available on [Hugging Face](https://huggingface.co/M2LInES/Samudra). You can download them as follows:
```bash
# Download the weights for thermo model
wget https://huggingface.co/M2LInES/Samudra/resolve/main/samudra_thermo_seed1.pt

# (OR) Download the weights for thermo-dynamic model
wget https://huggingface.co/M2LInES/Samudra/resolve/main/samudra_thermo_dynamic_seed1.pt
```

There are 5 seeds saved for each model.

## Paper Plots
The notebooks in the `notebooks` folder reproduce most of the plots from the paper.

Further methodological details and model architecture specifications can be found in the [paper](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024GL114318).

## Citation

If you find this code useful in your research, please cite:
```
@article{dheeshjith2025samudra,
  title={Samudra: An ai global ocean emulator for climate},
  author={Dheeshjith, Surya and Subel, Adam and Adcroft, Alistair and Busecke, Julius and Fernandez-Granda, Carlos and Gupta, Shubham and Zanna, Laure},
  journal={Geophysical Research Letters},
  volume={52},
  number={10},
  pages={e2024GL114318},
  year={2025},
  publisher={Wiley Online Library}
}
```
