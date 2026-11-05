# First-pass retrieval with the non-RV-GP global fit

Use `retrieval_run_joint.py` for the new analysis. `retrieval_run.py`, its
148 +/- 11 ppm helper, its independent legacy system priors, and all existing
files under `pymultinest/` are preserved.

## Adopted observations and system constraints

The selected source is
`data/LTT1445_Joint__Alltransiteclipse_binDDT60s_v2_e_w/MCMCResults_LTT1445_Joint__Alltransiteclipse_binDDT60s_v2_e_w.csv`.
The runner reads this file directly, using planet b (`__0`). It has 10,000
post-warmup draws and agrees with the four saved traces. It is the non-RV-GP
fit, not the similarly named RVGP folder or the older standalone rounded CSV.

F1500W uses the full chain's eclipse-depth mean and sample standard deviation:
**153.1200228 +/- 12.0113084 ppm**, approximated by a Gaussian likelihood.
The published SPARTA 16-bin LRS spectrum is retained unchanged. The combined
likelihood assumes independent spectral bins and independent instruments.
The LRS broadband eclipse depth in the global chain is not added as another
datum, since it would reuse the spectral information.

The system distribution preserves joint constraints rather than treating Teq
and Teff as independent:

1. Compute `Teq = Teff / sqrt(2*aor__0)` for every chain draw.
2. Retain the complete joint rows satisfying the saved climate grid support,
   **385 <= Teq <= 477 K**. This keeps **9,768/10,000 draws**; 1.92% are below
   and 0.40% above the grid. Temperatures are never clipped to grid boundaries.
3. Fit empirical marginal quantile functions and a Gaussian copula to the
   retained `(Teq, a/Rstar, Rp/Rstar)` samples. Ordered samples define continuous,
   piecewise-linear marginal quantile functions; ranked normal scores define
   the copula correlation matrix. This preserves marginal asymmetry while
   approximating the dependence between parameters.
4. Transform three unit-cube coordinates jointly, then derive
   `Teff = Teq * sqrt(2*a/Rstar)`. The parameters passed to the spectral model
   and saved by MultiNest remain `(Teq, Teff, R_planet_star)`.

This is a normalized continuous approximation to the retained empirical
posterior, not an exact representation of its full joint density. Each marginal
has the finite support of the retained samples. A plain multivariate Gaussian
was insufficient to meet the validation threshold in the a/Rstar tail; the
copula construction passes the checks on marginals and correlations.

The code checks that the entire constructed support remains inside the SPHINX
Teff grid. Unit-cube endpoints are moved inward by machine epsilon to avoid
infinite normal scores; this does not clip physical Teq values. Validation
compares 100,000 generated draws with the retained chain using marginal
quantiles, KS distances and correlations. The forward models are checked at
unit-cube endpoints and at the center.

**Scientific approximation:** the source fit already used LRS and F1500W eclipse
observations. Its system marginal is reused with a separate eclipse-depth
likelihood, neglecting the measured weak dependence between system parameters
and depth. This is the agreed first-pass modular treatment, not an
independent-data prior or a full reanalysis of the light curves. The same
system distribution is used for the LRS-only, F1500W-only, and combined cases;
"F1500W-only" describes the emission likelihood, not the provenance of all
system constraints. No RV or transit likelihood is evaluated again.

## Spectral models and throughput

The physical spectral functions are reused from `retrieval_run.py`. No global
legacy prior or response settings are changed. The existing climate grid keeps
its original fixed mass (2.73 Mearth), radius (1.34 Rearth), stellar irradiation
shape and other assumptions. No new climate calculations or grid extension
are needed for this first pass.

F1500W uses `inputs/F1500W_response.txt` by default: the local Pandeia
filter/instrument transmission times imager QE times OTE throughput. A custom
file can be supplied with `--response`. The photon-count prediction is

`(Rp/Rstar)^2 * integral(lambda R Fplanet) / integral(lambda R Fstar)`.

The display bounds in the data dictionary are not used to truncate the filter
wings. The continuous spectrum, stellar flux, and full response enter the
photometric integral. There is no rectangular-bin override in the new runner.
Synthetic proposal cases in the legacy runner retain their historical treatment.

## Running and outputs

Run from the repository root in the existing environment:

```sh
conda activate ltt
# Optional writable caches for restricted environments:
export NUMBA_CACHE_DIR=/private/tmp/ltt_numba
export MPLCONFIGDIR=/private/tmp/ltt_mpl

# Check distribution and physical predictions without sampling:
python retrieval_run_joint.py --check-only

# Run the six new retrievals:
python retrieval_run_joint.py
```

The six default cases are `rock_16`, `atm_16`, `rock_F1500W_obs`,
`atm_F1500W_obs`, `rock_16_F1500W_obs`, and `atm_16_F1500W_obs`. Select a subset
with `--cases`, e.g. `--cases rock_F1500W_obs atm_F1500W_obs`.

Use `--lrs-bins 8` for the corresponding SPARTA 8-bin sensitivity test. To add
only the four LRS-dependent runs to an existing tag, reusing its completed
F1500W-only results:

```sh
python retrieval_run_joint.py --lrs-bins 8 --tag nonrv_gp_v1 \
  --cases rock_8 atm_8 rock_8_F1500W_obs atm_8_F1500W_obs
```

The default remains 16 bins. With a fresh tag and no `--cases` option, either
bin selection runs all six cases, including the same F1500W-only comparison.
Old case directories are never overwritten or implicitly resumed.

Outputs go to **`pymultinest_joint/nonrv_gp_v1/<case>/`**, separate from the
historical tree. `--tag another_name` creates another run namespace. The runner
uses 1,000 live points by default, refuses existing case directories, and starts
without resuming an earlier chain. Relative filename prefixes are passed to
native MultiNest to avoid filename-buffer limits in long checkout paths.
The 8-bin LRS-only and combined cases were subsequently run at 1,000 live
points under `nonrv_gp_v1`, reusing the original F1500W-only results.

Each case saves normal MultiNest files, a result pickle, an AIC diagnostic,
and `inputs.json`. The latter records the chain hash, truncation and copula
settings, validation results, response values/hash, data, model source hashes,
fixed grid inputs and sampler settings. The grid itself is identified by path,
size and modification time. Old and new analyses can be imported together.

To save an additional setup report, choose a new JSON path:

```sh
python retrieval_run_joint.py --check-only --report validation/joint_setup.json
python -m unittest discover -s tests -v
```

Compare rock and atmosphere evidences only for the same dataset and system
prior. As in the historical runner, Gaussian normalization constants are omitted
from the log likelihood; they cancel in those matched model comparisons.
Do not compare absolute evidence values across different datasets. AIC for a
single photometric datum is retained only as a file-format convenience, not as
the atmospheric-retention decision statistic.

Check posterior proximity to the Teq limits after sampling. A final analysis
may need a grid extension and a more complete treatment of the system/eclipse
joint inference.

## Comparison figure

Run `python figure_joint.py` in `ltt` to produce
`figures/joint/figure_joint.pdf` and `.png`. The top row compares the three
atmospheric posterior spectra (median and pointwise 1-sigma bands), a fixed
zero-albedo blackbody reference, both datasets and the response weighted by
stellar photon flux (`lambda * R * Fstar_lambda`), alongside
the total surface-pressure posteriors. Four bottom panels show the H2O, CO2,
O2 and SO2 partial-pressure posteriors. Colors identify the dataset throughout;
matching-color dotted lines mark the 97.5th-percentile total and partial
pressure upper limits. Partial-pressure markers use lighter opacity, and a
single explanatory legend appears in the total-pressure panel. There are no
prior curves or numerical upper-limit labels.

For the complementary 8-bin figure, run `python figure_joint.py --lrs-bins 8`.
This reads the 8-bin LRS-only/combined cases and the shared F1500W-only result
from the selected `--results` directory. Its default output prefix is
`figures/joint/figure_joint_8`, so the 16-bin PDF, PNG and cache are preserved.
The default `--lrs-bins 16` retains the original filenames.

Adding bin selection changed the joint runner's source hash. When an otherwise
unchanged older result records that runner hash, the plotter verifies the
saved observations and recomputes five posterior likelihoods before accepting
it. Changes to physical-model sources or the chain/stellar/response data still
cause an error. This keeps the original 16-bin and F1500W results usable without
editing their metadata.

All posterior draws are used by default. Spectral bands are cached in
`figure_joint_spectra.npz`, with input/model fingerprints checked before reuse.
Use `--refresh-cache` to recompute them, `--results` to select another result
directory, or `--output-prefix` to choose the output filenames. An optional
`--max-spectra N` limits only spectral draws for a quick preview; pressure
histograms always use every posterior draw. The `.json` sidecar records the
plotted cases, fingerprints, interval convention and bare-rock reference.
