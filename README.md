# Samudra: A Global Ocean Emulator

This repository contains the implementation of Samudra, a global ocean emulator described in ["Samudra: An AI Global Ocean Emulator for Climate"](https://arxiv.org/abs/2412.03795). Samudra efficiently emulates the ocean component of a state-of-the-art climate model, accurately reproducing key ocean variables including sea surface height, horizontal velocities, temperature, and salinity, across their full depth.

<p align="center">
  <img src="/assets/globe.gif" >
</p>

## Overview

Samudra demonstrates:
- Capable of reproducing the temperature structure and variability of a realistic 3D ocean climate model
- Long-term stability under realistic time-dependent forcing conditions
- Performance improvement of ~150x compared to the original model
- Public availability via Hugging Face

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
```

Using [`uv`](https://docs.astral.sh/uv/):
```bash
uv sync
source .venv/bin/activate
```

## Usage

To train or rollout Samudra, you can download the OM4 data and statistics from the [Training Data](#training-data) section. Optionally, you could use your own data ensuring the data is in the same format as the OM4 data. The mean and standard deviation files are pre-computed for training and rollout, so you will need to compute them yourself if you use your own data.

You can also rollout the model using our trained model weights specified in the [Trained Model Weights](#trained-model-weights) section.

### Training
Start with the train_samudra_om4.yaml in the `configs` folder to train the model. You will need to update the fields that have `# FILL IN` with the correct paths to the base data directory and file names for the data and statistics.

> Make sure your environment is activated!
```bash
# Train a new model
python src/train.py --config path/to/train_config.yaml
```

### Rollout
Start with the rollout_samudra_om4.yaml in the `configs` folder to rollout the model. You will need to update the fields that have `# FILL IN` with the correct paths to the base data directory and file names for the data and statistics.

> Make sure your environment is activated!
```bash
# Produce a rollout from a trained model (and optionally save zarr)
python src/rollout.py --config path/to/rollout_config.yaml --ckpt_path path/to/checkpoint.pt --save_zarr
```

## Training Data
The OM4 data can be downloaded from our publicly hosted pod:

```python
import xarray as xr

# Download the data
data = xr.open_dataset("https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/OM4", engine='zarr', chunks={})
data.to_zarr("local/path/to/data.zarr")

# Download statistics
means = xr.open_dataset("https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/OM4_means", engine='zarr', chunks={})
means.to_zarr("local/path/to/means.zarr")

stds = xr.open_dataset("https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/OM4_stds", engine='zarr', chunks={})
stds.to_zarr("local/path/to/stds.zarr")
```

Now you can train / rollout and be sure to update the corresponding config file with the correct paths to the data and statistics.

## Trained Model Weights
If you want to use our trained model, download the weights from [Hugging Face](https://huggingface.co/M2LInES/Samudra).

```bash
# Download the weights for thermo model
wget https://huggingface.co/M2LInES/Samudra/blob/main/samudra_thermo_seed1.pt

# (OR) Download the weights for thermo-dynamic model
wget https://huggingface.co/M2LInES/Samudra/blob/main/samudra_thermo_dynamic_seed1.pt
```


For detailed methodology and model architecture, please refer to the [paper](https://arxiv.org/abs/2412.03795).

## Citation

If you use this code in your research, please cite:
```
@article{dheeshjith2024samudra,
  title={Samudra: An AI Global Ocean Emulator for Climate},
  author={Dheeshjith, Surya and Subel, Adam and Adcroft, Alistair and Busecke, Julius and Fernandez-Granda, Carlos and Gupta, Shubham and Zanna, Laure},
  journal={arXiv preprint arXiv:2412.03795},
  year={2024}
}
```
