# Samudra 🌊
This repository contains the code of the Samudra Ocean Emulator described in the paper ["Samudra: An AI Global Ocean Emulator for Climate"](https://arxiv.org/abs/2412.03795). Samudra is a skillful global emulator of the ocean component of a state-of-the-art climate model. We emulate key ocean variables, sea surface height, horizontal velocities, temperature, and salinity, across their full depth. 

<p align="center">
  <img src="/assets/globe.gif" >
</p>


### Key features

✅ Reliable: Capable of reproducing the temperature structure and variability of a realistic 3D ocean climate model

<p align="center">
  <img src="/assets/enso.gif" >
</p>

✅ Stable: run for multiple centuries in a realistic configuration with time-dependent forcing while maintaining stability and accuracy 

✅ Fast: it is 🚅150 times faster🚅 than its original counterpart 

✅ Open: Samudra is available on Hugging Face. Anyone can now run a global ocean emulator! 

## TODO

- [ ] Using HuggingFace data is currently very slow to generate a rollout, need a better method.
- [x] Add # Samudra content 
- [ ] Add visual aids (Rollout Video / Model arch fig), maybe https://drive.google.com/file/d/1S0rA0pUZWi4B0YlNztiLKCwRhGrBcbIX/view?resourcekey
- [ ] Fix environment.yml
- [x] Fix citation
- [ ] Code refactor / cleanup (?)
- [ ] HuggingFace ReadMEs
- [ ] Training code


### Getting Started
1. Clone this repository. 
2. Create a new conda environment using the `environment.yml` file.
3. Run the notebook `samudra_rollout.ipynb` to see how to generate a rollout with trained model weights.

### Model Weights
The model weights can be downloaded from [here](https://huggingface.co/M2LInES/Samudra).

### Data
The data used in the paper can be downloaded from [here](https://huggingface.co/datasets/M2LInES/Samudra_OM4). See the paper for more details on the data.

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
