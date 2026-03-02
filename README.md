## Setup

```sh
conda env create -f environment.yaml
mamba activate ltt
python input_files.py
```

On an HPC, you may have to further do something like the below in order to get a mpi4py that works well with the system.

```sh
conda uninstall mpi4py # uninstall the conda mpi4py
module load mpi-hpe/mpt # load up the HPC mpi
export MPICC=mpicc # MPI compiler
python -m pip install --no-binary=mpi4py mpi4py -v # re-install mpi4py linking to HPC MPI
```
