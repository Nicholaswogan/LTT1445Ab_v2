# Reproducing Wogan et al. (2026b)

This repository contains most of the code used in Wogan et al. (2026b), titled "A Climate-Constrained Bayesian Inverse Method for JWST Rocky Exoplanet Eclipse Spectra: Case Study of LTT 1445A b".

## Step 1: Installation and Setup

If you do not have Anaconda on your system, install it here or in any way you prefer: [https://www.anaconda.com/download](https://www.anaconda.com/download). Next, run the following code to create a conda environment `ltt`.

```sh
conda env create -f environment.yaml
conda activate ltt
```

For certain high performance computers, you may have to reinstall mpi4py with commands like the below.

```sh
conda uninstall mpi4py # uninstall the conda mpi4py
module load mpi-hpe/mpt # load up the HPC mpi
export MPICC=mpicc # MPI compiler
python -m pip install --no-binary=mpi4py mpi4py -v # re-install mpi4py linking to HPC MPI
```

## Step 2: Input files

With the `ltt` environment active, run the following script to generate needed input files.

```sh
python input_files.py
```

## Step 3: Climate and Spectral grid

This archive already contains a completed climate and spectral grid for LTT 1445A b located at `results/LTT1445Ab.h5`.

This file can be reproduced by running the following script. To start fresh, you must delete the file `results/LTT1445Ab.h5` manually otherwise the script will start from the already completed file. Again ensure you have the `ltt` environment active. This step will require a high performance computer outfitted with MPI (i.e., access to >10 cores). In the commands below, you should replace `NUMBER_OF_CORES` with however cores you want to distribute the calculation across (e.g., 10 nodes w/ 40 cores/node, then `NUMBER_OF_CORES` should be replaced with 400).

```sh
mpiexec -n NUMBER_OF_CORES python LTT1445Ab_grid.py
```

## Step 4: Run the retrievals

This archive already contains the two main retrieval results discussed in the manuscript in `pymultinest/`.

These results can be reproduced by running the script below. To start fresh, you must delete the folders in `pymultinest/` manually otherwise the script will start from the already completed results. Again ensure you have the `ltt` environment active. These retrievals are merely using linear interpolation of the pre-computed grid at `results/LTT1445Ab.h5`, so you can run the scripts in serial (no MPI) on a laptop.

```sh
python retrieval_run.py
```

## Step 5: Figures

Finally, make some of the key figures in the paper with.

```sh
python figures.py
```