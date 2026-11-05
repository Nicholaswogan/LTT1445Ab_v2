"""Continuous, correlated system constraints for the preliminary joint retrieval.

The chain is conditioned on grid support before constructing empirical marginal
quantile functions and a Gaussian copula. This retains asymmetric marginals
without double truncation or artificial probability spikes at grid boundaries.
"""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, ks_2samp

ROOT = Path(__file__).resolve().parent
DEFAULT_CHAIN = ROOT / 'data/LTT1445_Joint__Alltransiteclipse_binDDT60s_v2_e_w'
COLUMNS = ['teff', 'aor__0', 'ror__0', 'EclipseDepth_F1500W__0']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class JointSystemPrior:
    """Empirical marginals and Gaussian copula in (Teq, a/Rstar, Rp/Rstar).

    Transform returns (Teq, Teff, Rp/Rstar) for the historical spectral models.
    Teff = Teq*sqrt(2*a/Rstar), preserving the dominant physical correlation.
    Marginal quantile functions interpolate between ordered retained samples;
    a Gaussian copula approximates their joint dependence. This is a normalized
    continuous approximation, not discrete resampling of whole chain rows.
    """
    def __init__(self, samples, teq_bounds=(385., 477.), teff_bounds=(2000., 4000.)):
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 2 or samples.shape[1] != 3 or len(samples) < 10:
            raise ValueError('Expected joint samples with columns Teq, a/Rstar, Rp/Rstar')
        if not np.all(np.isfinite(samples)) or np.any(samples <= 0):
            raise ValueError('System samples must be finite and positive')
        self.samples = samples
        self.teq_bounds = tuple(map(float, teq_bounds))
        self.teff_bounds = tuple(map(float, teff_bounds))
        if not 0 < self.teq_bounds[0] < self.teq_bounds[1]:
            raise ValueError('Invalid Teq support')
        if not 0 < self.teff_bounds[0] < self.teff_bounds[1]:
            raise ValueError('Invalid Teff support')
        self.retained = samples[(samples[:, 0] >= self.teq_bounds[0])
                                & (samples[:, 0] <= self.teq_bounds[1])]
        if len(self.retained) < 10:
            raise ValueError('Too few chain samples within grid support')
        self.quantiles = np.sort(self.retained, axis=0)
        n = len(self.retained)
        self.probabilities = np.linspace(0., 1., n)
        latent = norm.ppf((rankdata(self.retained, axis=0)-.5)/n)
        self.copula_correlation = np.corrcoef(latent.T)
        self.cholesky = np.linalg.cholesky(self.copula_correlation)
        # Every combination permitted by the copula must be inside the stellar
        # grid too; reject unsupported inputs rather than clipping temperatures.
        min_teff = self.quantiles[0, 0]*np.sqrt(2*self.quantiles[0, 1])
        max_teff = self.quantiles[-1, 0]*np.sqrt(2*self.quantiles[-1, 1])
        if min_teff < self.teff_bounds[0] or max_teff > self.teff_bounds[1]:
            raise ValueError('Retained marginal supports extend beyond the stellar grid')
        if self.quantiles[-1, 2] >= 1:
            raise ValueError('Invalid planet/star radius ratio')
        self.source = None
        self.eclipse_mean = None
        self.eclipse_sd = None

    @classmethod
    def from_chain(cls, directory=DEFAULT_CHAIN, **kwargs):
        paths = sorted(Path(directory).glob('MCMCResults*.csv'))
        if len(paths) != 1:
            raise ValueError('Expected exactly one MCMCResults CSV in chain directory')
        data = pd.read_csv(paths[0], usecols=COLUMNS)
        samples = np.c_[data.teff/np.sqrt(2*data.aor__0), data.aor__0, data.ror__0]
        result = cls(samples, **kwargs)
        result.source = paths[0]
        # Use the unconditioned observed depth summary; do not change the
        # measurement by filtering on an unrelated grid boundary.
        depth = data.EclipseDepth_F1500W__0.to_numpy()
        result.eclipse_mean = float(depth.mean())
        result.eclipse_sd = float(depth.std(ddof=1))
        return result

    def transform(self, unit):
        unit = np.asarray(unit, dtype=float)
        if unit.shape[-1:] != (3,) or not np.all(np.isfinite(unit)) or np.any((unit < 0) | (unit > 1)):
            raise ValueError('Expected three unit-cube coordinates in [0,1]')
        # Avoid infinite normal scores at exact numerical unit endpoints. This
        # affects only machine-precision endpoint mass, not physical Teq values.
        eps = np.finfo(float).eps
        independent = norm.ppf(np.clip(unit, eps, 1-eps))
        correlated = independent@self.cholesky.T
        probabilities = norm.cdf(correlated)
        values = np.stack([np.interp(probabilities[..., i], self.probabilities, self.quantiles[:, i])
                           for i in range(3)], axis=-1)
        q, a, r = values[..., 0], values[..., 1], values[..., 2]
        teff = q*np.sqrt(2*a)
        return np.stack([q, teff, r], axis=-1)

    def validate(self, draws=100000, seed=1445):
        rng = np.random.default_rng(seed)
        transformed = self.transform(rng.random((draws, 3)))
        generated = np.c_[transformed[:, 0], (transformed[:, 1]/transformed[:, 0])**2/2,
                          transformed[:, 2]]
        reference = self.retained
        quantiles = [.025, .16, .5, .84, .975]
        ref_sd = reference.std(axis=0, ddof=1)
        q_error = (np.quantile(generated, quantiles, axis=0)
                   - np.quantile(reference, quantiles, axis=0))/ref_sd
        correlation_error = np.corrcoef(generated.T)-np.corrcoef(reference.T)
        ks = [float(ks_2samp(generated[:, i], reference[:, i]).statistic) for i in range(3)]
        return {
            'draws': draws, 'seed': seed, 'comparison_coordinates': ['Teq', 'aor', 'R_planet_star'],
            'empirical_retained_mean': reference.mean(axis=0).tolist(),
            'generated_mean': generated.mean(axis=0).tolist(),
            'empirical_retained_sd': ref_sd.tolist(),
            'generated_sd': generated.std(axis=0, ddof=1).tolist(),
            'empirical_retained_correlation': np.corrcoef(reference.T).tolist(),
            'generated_correlation': np.corrcoef(generated.T).tolist(),
            'max_abs_correlation_error': float(np.max(np.abs(correlation_error))),
            'quantiles': quantiles, 'quantile_errors_in_empirical_sd': q_error.tolist(),
            'max_abs_quantile_error_in_sd': float(np.max(np.abs(q_error))),
            'marginal_KS_distances': ks,
            'passes_preliminary_checks': bool(max(ks) < .035 and np.max(np.abs(q_error)) < .12
                                             and np.max(np.abs(correlation_error)) < .035),
            'note': 'Distribution approximation checks, not a proof of equality or MCMC convergence.',
        }

    def metadata(self):
        q = self.samples[:, 0]
        return {
            'source_csv': None if self.source is None else str(self.source),
            'source_sha256': None if self.source is None else sha256(self.source),
            'planet_index': 0, 'chain_draws': len(q), 'retained_draws': len(self.retained),
            'below_grid_fraction': float(np.mean(q < self.teq_bounds[0])),
            'above_grid_fraction': float(np.mean(q > self.teq_bounds[1])),
            'Teq_bounds_K': self.teq_bounds, 'Teff_bounds_K': self.teff_bounds,
            'distribution': 'empirical marginal quantile functions with Gaussian copula, fitted to retained (Teq,aor,ror) samples',
            'marginal_support': np.c_[self.quantiles[0], self.quantiles[-1]].tolist(),
            'copula_correlation': self.copula_correlation.tolist(),
            'marginal_interpolation': 'linear quantiles, equally spaced probabilities from zero to one',
            'derivation': 'Teq=Teff/sqrt(2*aor); transformed samples return (Teq,Teff,ror)',
            'F1500W_depth': self.eclipse_mean, 'F1500W_sd': self.eclipse_sd,
            'approximation': 'All-data system marginal reused with a separate Gaussian eclipse likelihood; correlations with eclipse depth are neglected. This is not an eclipse-independent prior.',
        }
