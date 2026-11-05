# Free-temperature LRS + F1500W retrieval

`retrieval_run_free.py` fits the same observations as
`pymultinest_joint/nonrv_gp_v1/atm_16_F1500W_obs`, calling PICASO directly
instead of interpolating the climate grid. Existing runners and results are
not modified. Use conda environment `ltt` locally, or an equivalent HPC setup.

## Parameters and forward model

The 11 parameters are four log10 partial pressures, surface albedo, four
temperature-profile parameters, stellar effective temperature, and Rp/Rstar.
The partial-pressure priors in bar are H2O [-7,2], CO2 [-7,2], O2 [-5,2], and
SO2 [-7,2], uniform in log10 pressure. Surface albedo is uniform [0,0.4].
These are identical to the combined climate-constrained retrieval.

The profile is continuous and piecewise linear in log10 pressure. T_ref is
defined at **0.1 bar**, including when the break pressure is above/below that
reference. It is an extrapolated reference temperature if the surface lies
above 0.1 bar. Its new independent uniform priors are:

| Parameter | Bounds |
| --- | --- |
| T_ref | 100–2000 K |
| log10(P_break/bar) | -6–1 |
| Upper slope | -400–400 K per pressure decade |
| Lower slope | 0–1000 K per pressure decade |

Negative upper slopes permit inversions. A break outside the atmospheric
column is permitted; the corresponding unused slope can be unconstrained.
The formula corrects the reference-temperature offset in the older
`hotrocks/retrieval_run.py` example.

Surface pressure is the sum of the sampled partial pressures. Their fractions
give height-independent gas mixing ratios; condensation and chemical
equilibrium are not imposed. Sixty pressure levels span the common molecular
opacity-table minimum pressure to the surface (default top: 1e-6 bar).
The bottom atmospheric temperature is also the surface temperature. The same
gray surface albedo is passed to PICASO's hard-surface boundary, with the
corresponding thermal emissivity. Mass and radius remain 2.73 Earth masses
and 1.34 Earth radii, matching the original grid's gravity.

Profiles outside the common molecular opacity temperature/pressure bounds
receive zero likelihood (currently 50–2000 K and 1e-6–1000 bar). Temperatures
are not clipped. The declared rectangular priors are **not renormalized**
after this validity cut. About 24.4% of 1000 pilot prior draws were valid.
Invalid profiles receive a numerically zero likelihood, with log likelihood
`-1e10 - sum(squared temperature-bound violations in K)`. This varying floor
avoids an identical-likelihood plateau that can bias nested-sampling volume
accounting. It stays above the explicit MultiNest `log_zero=-1e200`.
Posterior inference is conditioned on validity, but the reported evidence
includes the excluded prior volume. Account for this convention before
interpreting a Bayes factor against the climate-constrained retrieval. No
energy-balance or convective-stability cut is imposed.

The stellar prior is the two-dimensional marginal of the existing
`JointSystemPrior`, including its original 385–477 K conditioning in Teq.
Teq and chi are omitted from this free model; they do not set its temperatures.
The Teff/Rp-Rstar marginal is evaluated with conditional CDF inversion and
192-point Gaussian quadrature, without fitting another copula. Tests compare
it to draws from the original prior and check quadrature convergence.
The existing caveat remains: the source system posterior includes eclipses,
and its weak correlations with the fitted eclipse depths are neglected.

The planetary surface spectrum is rebinned to R=100 over 4–25 microns. The
eclipse ratio uses the sampled SPHINX Teff and Rp/Rstar. LRS predictions use
rectangular wavelength-bin averages. F1500W uses the stellar-photon-weighted
instrument response and the same 153.1200 +/- 12.0113 ppm Gaussian datum.
The likelihood is -chi2/2, with the same omitted normalization as the joint run.

## Files required on the HPC

Copy the Python sources and these actual inputs (resolve local symlinks):

- `data/LTT1445Ab_Sparta_16.txt`
- `data/LTT1445_Joint__Alltransiteclipse_binDDT60s_v2_e_w/MCMCResults*.csv`
- `inputs/sphinx.h5`
- `inputs/F1500W_response.txt`
- `picasofiles/opacities_photochem_0.1_250.0_R15000.db`
- The PICASO reference-data directory at `picasofiles/reference`.

The large climate-grid HDF5 files and previous retrieval outputs are **not
required**. Optional `--database`, `--chain`, and `--response` arguments accept
alternate locations. `input_files.py` configures the repository-local PICASO
reference-data directory, so it must exist on the HPC.

The environment needs numpy, scipy, pandas, h5py, astropy, photochem (with the
hotrocks extension), PICASO, PyMultiNest, and mpi4py. Native MultiNest must be
built with MPI support; MPI, mpi4py, and MultiNest must use compatible MPI
libraries. Load the appropriate HPC modules/conda environment first.

## Preflight and MPI run

From the repository root, first run:

```bash
python retrieval_run_free.py --check-only --report validation/free_setup.json
```

This checks representative profiles and spectra and estimates how often prior
draws stay within the opacity domain. The first PICASO call includes compilation
overhead. A report path must be new to prevent accidental overwrites.

To check your MPI setup, run the same check under your launch command:

```bash
mpiexec -n 2 python retrieval_run_free.py --check-only
```

Then start the retrieval, for example:

```bash
mpiexec -n 16 python retrieval_run_free.py --tag free_v1 --n-live-points 1000
```

Use your cluster's `srun` equivalent if required. Request one CPU per MPI rank;
the script sets native thread counts to one. Each rank holds its own PICASO
opacity data in memory, so choose the rank count with memory and shared-filesystem
I/O in mind. No Python multiprocessing is used. The seed defaults to 1445.

All ranks participate in the same native MultiNest run. Rank 0 creates the
output directory and writes metadata and final summaries; native MultiNest
handles its own parallel output. The script refuses an existing case directory
and uses short relative native filenames to avoid Fortran filename-buffer
limits. It does not resume earlier runs.

Outputs are under `pymultinest_free/free_v1/atmfree_16_F1500W_obs/`:

- `inputs.json`: priors, data, physical assumptions, versions, input hashes,
  opacity database path/size/mtime, MPI rank count, and run settings.
- `free_*`: native MultiNest outputs, including `free_post_equal_weights.dat`
  (the 11 parameter columns in `param_names` order, then log likelihood).
- `free_.pkl`: evidence estimates and equal-weight posterior samples.
- `fit_summary.json`: evidence, best sampled chi-square, AIC, and sample count.

Run the focused tests with:

```bash
python -m unittest discover -s tests -p test_free_retrieval.py -v
```
