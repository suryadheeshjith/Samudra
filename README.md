# Samudra 🌊
This repository contains the code of the Samudra Ocean Emulator described in the paper ["Samudra: An AI Global Ocean Emulator for Climate"](https://arxiv.org/abs/2412.03795). Samudra is a skillful global emulator of the ocean component of a state-of-the-art climate model. We emulate key ocean variables, sea surface height, horizontal velocities, temperature, and salinity, across their full depth. 

<p align="center">
  <img src="/assets/globe.gif" >
</p>


### Key features

- Reliable ✅ : Capable of reproducing the temperature structure and variability of a realistic 3D ocean climate model
- Stable ✅ : run for multiple centuries in a realistic configuration with time-dependent forcing while maintaining stability and accuracy 
- Fast ✅ : it is 🚅150 times faster🚅 than its original counterpart 
- Open ✅ : Samudra is available on Hugging Face. Anyone can now run a global ocean emulator! 

<p align="center">
  <img src="/assets/enso.gif" >
</p>

### Getting Started
1. Clone this repository. 
```
git clone https://github.com/suryadheeshjith/Samudra.git
cd Samudra
```
2. Create a new conda environment using the `environment.yml` file.
```bash
conda env create -f environment.yml
```
Alternatively, create a new python environment with [`uv`](https://docs.astral.sh/uv/): [Preferred Method]
```bash
uv sync
source .venv/bin/activate
```

3. Run the notebook `samudra_rollout.ipynb` to generate a rollout with trained model weights.

Notebooks `samudra_plotting.ipynb` and `samudra_plotting_multiseed.ipynb` contain the code to generate the plots in the paper using a generated rollout.

### Model Weights and Data
The model weights are currently hosted on huggingface and can be downloaded from [here](https://huggingface.co/M2LInES/Samudra). The OM4 data used for training and testing the models in the paper can be accessed using: 

```python
import xarray as xr
data = xr.open_dataset("https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/OM4", engine='zarr', chunks={})
```

For more details on the data and the model, please refer to the paper.

### Citing
If you use this code in your research, please consider citing the following paper:
```
@article{dheeshjith2024samudra,
  title={Samudra: An AI Global Ocean Emulator for Climate},
  author={Dheeshjith, Surya and Subel, Adam and Adcroft, Alistair and Busecke, Julius and Fernandez-Granda, Carlos and Gupta, Shubham and Zanna, Laure},
  journal={arXiv preprint arXiv:2412.03795},
  year={2024}
}
```
