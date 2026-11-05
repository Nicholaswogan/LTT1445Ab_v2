# Retrievals with the final GP system fit

The supported system fit is
`data/LTT1445_Joint__Alltransiteclipse_binDDT60s_v2_12758GP_RVGP_e_w/`.
Only its unrounded `MCMCResults*.csv` is required for inference. This fit
includes an RV GP and a photometric SHO GP plus exponential baseline for the
DD12758 F1500W visit. Planet b is column index `__0`.

## Observations and likelihood

The F1500W chain median is **157.472924 ppm**, with lower/upper uncertainties
**17.313102/17.816034 ppm**, from the central 68.268949% interval. Report this
as **157 +18/-17 ppm**. These statistics use all source draws, before Teq
conditioning. The two-piece Gaussian likelihood uses the lower uncertainty
for predictions below the median and the upper uncertainty above it. This
is a small-asymmetry approximation to the depth marginal, not its exact PDF.

The SPARTA LRS spectra are retained unchanged, with independent symmetric
Gaussian bin errors. The combined likelihood adds LRS and F1500W terms,
assuming independent instruments. The broadband LRS depth in the system
chain is not added as another observation. `utils.observation_loglike` is
shared by the climate-constrained and free-temperature retrievals.
Constant likelihood normalization terms are omitted; they cancel between
rock/atmosphere fits to the same observations. Do not compare absolute
log-evidences across different datasets.

`err_lower` and `err_upper` contain the measurement errors used for inference.
`err` is their average for compatibility only. New plots use
`utils.observation_yerr` to retain asymmetric error bars. The legacy
148 +/- 11 ppm helper remains confined to `retrieval_run.py`'s older cases.

## Correlated system prior and grid

1. Compute Teq = Teff / sqrt(2 a/Rstar) for every source draw.
2. Retain complete rows with **385 <= Teq <= 477 K**: **9,791/10,000** draws.
   The excluded fractions are 1.75% below and 0.34% above the grid. Values
   are never clipped to its boundaries.
3. Fit empirical marginal quantile functions and a Gaussian copula in
   (Teq, a/Rstar, Rp/Rstar), then derive Teff in the prior transform.
4. Validate generated marginals and correlations and check stellar-grid support.

The source includes the eclipse observations. Reusing its system marginal
with a separate eclipse likelihood neglects weak parameter/depth correlations;
this remains a modular approximation, not an independent-data prior or a new
light-curve fit. "F1500W only" describes the emission likelihood, not the
provenance of the system constraints.

The provided traces contain 71 post-warmup divergence flags across 10,000
samples (11, 58, 2, 0 by chain). They are retained; assess these diagnostics
with the fit author before publication. Small shifts after diagnostic removal
do not establish unbiased posterior tails.

The climate grid keeps its fixed 2.73 Earth masses and 1.34 Earth radii.
The final system fit gives approximately 2.38 Earth masses and 1.30 Earth
radii. Seven paired climate calculations found maximum mass/radius-induced
changes of **0.71 ppm in F1500W** and **1.26 ppm in an LRS bin**. All converged.
The existing grid is retained given these small changes relative to the errors;
this is a representative sensitivity check, not an exhaustive guarantee.
See `results/final_gp_v1/new_data_assessment.json` after running the assessment.

Atmospheric priors are unchanged: independent uniform log10 partial pressures
in bar, H2O/CO2/SO2 [-7,2], O2 [-5,2]; uniform log10 chi over [log10(0.05),
log10(0.8)]; and surface albedo uniform [0,0.4].

F1500W always uses `inputs/F1500W_response.txt`, a product of filter/instrument,
QE, and telescope throughputs. The prediction is
`(Rp/Rstar)^2 integral(lambda R Fplanet) / integral(lambda R Fstar)`.
LRS bins retain rectangular averages. Display bounds do not truncate the
photometric response wings.

## Validate and run

Use conda environment `ltt` from the repository root:

```sh
python assess_new_data.py --sensitivity
python retrieval_run_joint.py --check-only --report validation/final_joint_setup.json
python -m unittest discover -s tests -v
python retrieval_run_joint.py
```

The default six cases are `rock_16`, `atm_16`, `rock_F1500W_obs`,
`atm_F1500W_obs`, `rock_16_F1500W_obs`, and `atm_16_F1500W_obs`.
To add the four 8-bin LRS-dependent cases while sharing F1500W-only results:

```sh
python retrieval_run_joint.py --lrs-bins 8 \
  --cases rock_8 atm_8 rock_8_F1500W_obs atm_8_F1500W_obs
```

Results go to **`pymultinest_joint/final_gp_v1/<case>/`**. The default is
1,000 live points. Existing case directories are refused and runs are not
implicitly resumed. `--tag` selects a fresh output namespace. The joint
runner's commands above are serial; use the MPI-aware free runner documented
in `FREE_RETRIEVAL.md` for free-temperature inference on the HPC.

Each case writes native MultiNest outputs, a result pickle, AIC diagnostics,
and `inputs.json` containing source hashes, fit identity, observations,
asymmetric errors, priors, response and grid provenance. Sampling summaries
and plots must be regenerated from these new results; old figures are not
updated simply by changing their labels.

## Figures and derived climates

```sh
python figure_joint.py
python figure_joint.py --lrs-bins 8
python figures_presentation.py --figure retrieval
python figures_presentation.py --figure joint-posteriors
python figures_presentation.py --figure joint-spectra
python figures_presentation.py --figure f1500w-posteriors
python figures_presentation.py --figure f1500w-spectra
python posterior_climate.py
python posterior_climate.py --case f1500w
```

Comparison figures and spectral caches are under `figures/joint/`;
presentation outputs and caches are under `figures/present/`. New plots overwrite
the previous figures at these paths.
Derived climate HDF5/JSON files are under `results/final_gp_v1/`.
Pressure upper limits are consistently the **97.5th posterior percentiles**,
with matching annotations and metadata. Spectral shading remains the pointwise
central one-sigma interval. Joint-density contour probabilities (68%, 95%)
are separate from the upper-limit convention.

Plotting validates that posterior metadata identify the supported source fit.
It does not fall back to archived results. `figure_joint.py --results` can
select another output tag from the same final fit. `--output-prefix` selects
an alternate figure location. Fingerprints invalidate caches when inputs
change. `--refresh-cache`/`--refresh-spectra`/`--refresh-models` force their
respective recalculations.
