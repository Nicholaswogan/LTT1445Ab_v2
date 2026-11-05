# Observed F1500W and the joint system fit

**Historical assessment:** the notes below describe the initial rounded-summary
assessment and legacy-prior setup. The subsequently selected full-chain analysis
is implemented separately; see [JOINT_RETRIEVAL.md](JOINT_RETRIEVAL.md).

The existing atmospheric grids can be reused for preliminary LRS + F1500W and
F1500W-only inference. A full replacement grid is not justified by the changes
in the supplied system summary. Final inference still needs an unrounded system
fit and a decision about the LRS light-curve geometry.

## What was checked

`assess_new_data.py` reads the saved HDF5 axes and the collaborator's CSV.
`results/new_data_assessment.json` records the diagnostics and direct climate
comparisons performed in conda environment `ltt`.

Both saved grids cover 4.000–24.999 microns, including the full F1500W response.
The nominal grid has 684,000 completed, converged models; the hot grid has
228,000. The saved `fp` is planet **surface flux**, not the eclipse depth:
`LTT1445Ab_grid.model` takes the second return value of `fpfs_picaso`, whose
implementation returns `(wavelength_edges, planet_flux, eclipse_ratio)`.
Consequently Rp/Rstar and the observed stellar surface flux can change in the
retrieval without regenerating `fp`.

| Quantity | Legacy grid / inference | New summary diagnostic |
| --- | --- | --- |
| Zero-albedo Teq | 431 +/- 23 K; grid 385, 431, 477 K | 424.79 K; approximate sd 19.16 K |
| Planet mass | 2.73 Earth masses | 2.685 +/- 0.218 Earth masses |
| Planet radius | 1.34 Earth radii | ~1.294 Earth radii from mean mass/density |
| Planet gravity | fixed | ~5.55% higher |
| Stellar radius | 0.271 solar radii | 0.265 +/- 0.005 solar radii |
| Stellar Teff | 3340 +/- 150 K | 3340.163 +/- 149.546 K |
| Stellar logg | 4.97 | ~5.003 from mean mass/radius |
| Rp/Rstar | 0.0454 +/- 0.0012 | 0.045; sd rounded to 0.000 |

These are diagnostics, not newly measured priors. In particular, the radius is
computed from marginal mean mass and density, not the mean of the radius
posterior. The CSV's `r_pl[0] = 0.012` is too coarsely rounded to infer a precise
Earth radius directly. The Teq estimate uses `Teff/sqrt(2*aor[0])`, avoiding
independent propagation of correlated Rstar and semimajor axis. Its quoted sd
ignores covariance, which the summary does not provide. No change to stellar
metallicity is supplied.

At Teq = 424.79 K, direct old/new mass-radius climate comparisons gave:

| Atmosphere | Change in F1500W (ppm) | Max absolute LRS change (ppm) | Absolute F1500W interpolation error (ppm) |
| --- | ---: | ---: | ---: |
| Thin limit | -0.004 | 0.121 | 0.112 |
| 1 bar O2 + 0.01 bar CO2 | +0.220 | 0.530 | 0.283 |
| 0.1 bar CO2 | -0.049 | 0.404 | 0.284 |
| 0.001 bar H2O | -0.074 | 0.056 | 0.119 |
| 0.0001 bar SO2 | -0.080 | 0.413 | 0.076 |
| O2/CO2 between grid nodes | +0.101 | 0.435 | 0.670 |
| H2O between grid nodes | -0.049 | 0.063 | 0.390 |

All 14 direct calculations converged. These use the local Pandeia photon
response and the rounded radius ratio 0.045 for a common eclipse conversion.
The two between-node cases also use chi = 0.125 and albedo = 0.15; the other
cases use chi = 0.2 and albedo = 0.2. The largest LRS interpolation error in
these cases was 1.063 ppm. R=100 versus R=1000 direct spectra differed by
0.040 and 0.037 ppm in F1500W for the O2/CO2 case and its between-node case.
These are targeted checks, not a bound over the full grid or a posterior test.
They support reusing the nominal grid at the present 11 ppm F1500W precision.
The hot grid's wavelength coverage and convergence were audited; its
interpolation and mass/radius sensitivity were not separately validated.

After retrieval, validate representative posterior samples directly, especially
near pressure limits or grid edges. Extend/refine the grid if those checks show
errors material relative to the measurement uncertainty. A change in Rp/Rstar
alone does not require new climate runs.

## Priors and the cool edge

Keep the existing `planets.py` values as the provenance of the saved grid.
The run script defaults to the legacy independent truncated-normal priors.
It does **not** interpret the rounded CSV as exact system parameters.

A revised explicit independent-prior configuration can be passed with
`--priors system_priors.json`. The file must contain exactly `Teq`, `Teff`,
and `R_planet_star`, each as `[mean, sd, lower, upper]`. For example, this
reproduces the legacy priors:

```json
{
  "Teq": [431.0, 23.0, 385.0, 477.0],
  "Teff": [3340.0, 150.0, 3040.0, 3640.0],
  "R_planet_star": [0.0454, 0.0012, 0.0430, 0.0478]
}
```

This interface explicitly assumes independence. It is not an implementation of
the collaborator's joint posterior. For final results, obtain unrounded
posterior samples for Teff, a/Rstar, Rp/Rstar, Rstar, planet mass, and their
correlations (plus confirmation of planet index and radius units). Construct
Teq sample by sample and retain its correlation with Teff and Rp/Rstar.
Prefer constraints from transit/RV/stellar data independent of the eclipse
measurements being fitted. Reusing a posterior already conditioned on those
same eclipses as a prior can count the observations twice.

A Gaussian approximation centered at 424.79 K is well covered by the current
Teq grid. However, the rectangular +/-2 sd support in the new Teff and a/Rstar
marginals permits Teq as low as 382.62 K. An illustrative independent truncated
normal simulation puts ~0.05% below 385 K; the actual fraction requires the
joint posterior. If the adopted final prior needs that tail, extend the grid
below 385 K, or explicitly justify and report conditioning the prior on grid
support. The code rejects out-of-range Teq priors rather than silently clipping
models. No full rerun is needed solely because the new Teq mean is lower.

## Running

Run from the repository root:

```sh
conda activate ltt
# Writable caches are useful under a restricted execution environment:
export NUMBA_CACHE_DIR=/private/tmp/ltt_numba
export MPLCONFIGDIR=/private/tmp/ltt_mpl
python retrieval_run.py --list-cases
python retrieval_run.py --tag preliminary_legacy_v1
```

The default selection runs rock and atmosphere for both the 16-bin LRS +
observed F1500W data and observed F1500W alone. These are **preliminary legacy
system-prior runs**. No full nested sampling runs were launched during this
update. To select a case, add e.g. `--cases atm_F1500W_obs`.
`atm_hot_16_F1500W_obs` and `atm_hot_F1500W_obs` are also available as optional
no-redistribution comparisons. Default runs use 1000 live points.

Outputs go to `pymultinest/<tag>/<case>/`, with the solver output, result pickle,
AIC diagnostic, and `inputs.json` recording data, response, priors and settings.
Existing case directories are refused, preventing accidental resume with
changed inputs. Use a fresh tag for a changed dataset, response, or prior.
Existing notebooks that assume `pymultinest/<case>/` need to point to the tagged
output directory when reading these new results.

Compare atmospheric and rock log evidences **within the same dataset and
configuration**; do not compare the absolute log evidence of the F1500W-only
run with the combined-data run. The likelihood retains the legacy omitted
Gaussian normalization, which cancels for model comparisons on identical data.
For one photometric point, parameter-count AIC is not a useful atmospheric
retention diagnostic; inspect Bayes factors and prior-to-posterior pressure
constraints instead.

The bundled `inputs/F1500W_response.txt` is loaded by default for both script
runs and notebook imports. Use `--response path/to/response.txt` to override it.
For notebook predictions, call `case['predict'](parameters)`.
This applies the same photometric response as the likelihood. `model_raw`
continues to return the wavelength-resolved ratio; the legacy `model` function
continues to average that ratio into arbitrary rectangular bins.

Observed F1500W is 148 +/- 11 ppm. Its display bounds remain 13.6–16.5 microns,
but its likelihood now uses

`integral(lambda * response * Fplanet) / integral(lambda * response * Fstar) * (Rp/Rstar)^2`.

Equivalently, the code integrates `fpfs * Fstar` in the numerator, since `fpfs`
already includes the radius ratio. It does not average eclipse ratios uniformly
across the filter. `--allow-tophat` explicitly permits the legacy approximation
instead of the default response; it cannot be combined with `--response`.
Synthetic proposal cases retain their original
rectangular bins.

`inputs/F1500W_response.txt` was exported from the Pandeia reference files
already used for plotting in `retrival.py`, multiplying the F1500W
filter/instrument transmission, imager QE, and OTE throughput. Component paths
and hashes are in `inputs/F1500W_response.json`. Confirm this calibration choice
with the observer; these are local reference files, not a newly downloaded
calibration release. See the [STScI filter documentation](https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-instrumentation/miri-filters-and-dispersers).
To rebuild from local reference data:

```sh
python prepare_f1500w_response.py /path/to/pandeia_data
python assess_new_data.py --sensitivity --response inputs/F1500W_response.txt
python -m unittest discover -s tests -v
```

Published LRS depths and uncertainties are retained as supplied. The reference
paper fixed several geometric parameters in its light-curve fits. A new global
geometry does not justify simply multiplying published depths by a radius
factor: check with the observers whether re-extraction/refitting changes the
spectral depths. The combined likelihood currently assumes independent Gaussian
LRS bins and F1500W, with no covariance or inter-instrument offset.

The gray bare-rock treatment is preserved: emissivity is `1-albedo`, so the
absorption/emission factors cancel in the surface temperature, as in the
paper's bare-rock equation. Changing this prescription would change the
scientific model and its comparison with the atmospheric thin limit.
