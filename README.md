## Setup

```sh
conda env create -f environment.yaml
conda activate ltt
python input_files.py
```

On an HPC, you may have to further do something like the below in order to get a mpi4py that works well with the system.

```sh
conda uninstall mpi4py # uninstall the conda mpi4py
module load mpi-hpe/mpt # load up the HPC mpi
export MPICC=mpicc # MPI compiler
python -m pip install --no-binary=mpi4py mpi4py -v # re-install mpi4py linking to HPC MPI
```

## Using Codex/Claude

If you use a coding tool like Codex or Claude, then download all the Photochem source with the commands bellow, and tell the bot to read the code when trying to make changes.

```sh
mkdir -p codex_reference && cd codex_reference
for u in \
  https://github.com/Nicholaswogan/photochem/archive/refs/tags/v0.8.2.zip \
  https://github.com/Nicholaswogan/clima/archive/refs/tags/v0.7.4.zip \
  https://github.com/Nicholaswogan/Equilibrate/archive/refs/tags/v0.2.1.zip \
  https://github.com/natashabatalha/picaso/archive/refs/tags/v4.0.zip \
  https://github.com/Nicholaswogan/photochem_clima_data/archive/refs/tags/v0.3.1.zip
do
  f="$(basename "$u")"
  wget -O "$f" "$u"
  unzip -q "$f"
  rm -f "$f"
done
cd ..
```

## Observed F1500W inference

Use [JOINT_RETRIEVAL.md](JOINT_RETRIEVAL.md) and `retrieval_run_joint.py` for the
final photometric/RV GP chain, 157 +18/-17 ppm F1500W depth, correlated system
constraints, and explicit Teq truncation. Run in the `ltt` conda environment.

`retrieval_run.py` preserves the historical cases and settings. The current
grid assessment is described in [NEW_DATA.md](NEW_DATA.md).

For a free temperature-profile retrieval of the combined 16-bin LRS and
F1500W observations, use `retrieval_run_free.py`; MPI/HPC instructions and
prior definitions are in [FREE_RETRIEVAL.md](FREE_RETRIEVAL.md).
