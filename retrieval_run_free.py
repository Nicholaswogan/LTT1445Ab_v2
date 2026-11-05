"""Free-temperature retrieval of 16-bin LRS + observed F1500W.

Run in the ltt environment (or its HPC equivalent):
  python retrieval_run_free.py --check-only --report validation/free_setup.json
  mpiexec -n 16 python retrieval_run_free.py --tag final_gp_v1

No climate solver or precomputed emission grid is used. See FREE_RETRIEVAL.md.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import importlib
import json
from pathlib import Path
import pickle
import sqlite3
import sys
import time
from tempfile import NamedTemporaryFile

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri, roots_hermitenorm

from joint_system_prior import JointSystemPrior, ROOT, DEFAULT_CHAIN, FIT_TAG, sha256

PARAMETERS = ['log10PH2O', 'log10PCO2', 'log10PO2', 'log10PSO2', 'albedo',
              'T_ref', 'log10P_break', 'slope_up', 'slope_lo', 'Teff', 'R_planet_star']
BOUNDS = np.array([[-7., 2.], [-7., 2.], [-5., 2.], [-7., 2.], [0., .4],
                   [100., 2000.], [-6., 1.], [-400., 400.], [0., 1000.]])
CASE = 'atmfree_16_F1500W_obs'


def pt_profile(pressure, temperature_ref, pressure_break, slope_up, slope_lo, pressure_ref=.1):
    """Continuous broken-linear T(log10 P), with T(P_ref) exactly T_ref."""
    pressure = np.asarray(pressure, dtype=float)
    if np.any(pressure <= 0) or pressure_break <= 0 or pressure_ref <= 0:
        raise ValueError('Pressures must be positive')
    log_break = np.log10(pressure_break)
    def primitive(logp):
        d = logp - log_break
        return np.where(d <= 0, slope_up*d, slope_lo*d)
    return temperature_ref + primitive(np.log10(pressure)) - primitive(np.log10(pressure_ref))


class MarginalStellarPrior:
    """Numerically marginalize the *existing* copula over Teq and a/Rstar.

    Draw r from its empirical marginal, then invert the conditional Teff CDF.
    A Gaussian quadrature integrates over latent a/Rstar; latent Teq is
    integrated analytically using its empirical quantile inverse. This avoids
    adding a new copula approximation or sampling an unused Teq coordinate.
    """
    def __init__(self, system, quadrature_order=192):
        self.system = system
        self.order = quadrature_order
        self.nodes, weights = roots_hermitenorm(quadrature_order)
        self.weights = weights/np.sqrt(2*np.pi)
        correlation = system.copula_correlation
        self.ar = correlation[1, 2]
        self.beta = np.linalg.solve(correlation[1:, 1:], correlation[1:, 0])
        self.q_sd = np.sqrt(1 - correlation[0, 1:] @ self.beta)
        self.lower = system.quantiles[0, 0]*np.sqrt(2*system.quantiles[0, 1])
        self.upper = system.quantiles[-1, 0]*np.sqrt(2*system.quantiles[-1, 1])

    def transform(self, unit):
        unit = np.asarray(unit, dtype=float)
        if unit.shape != (2,) or not np.isfinite(unit).all() or np.any((unit < 0) | (unit > 1)):
            raise ValueError('Expected two unit coordinates')
        u_t, u_r = np.clip(unit, np.finfo(float).eps, 1-np.finfo(float).eps)
        zr = ndtri(u_r)
        za = self.ar*zr + np.sqrt(1-self.ar**2)*self.nodes
        q_mean = self.beta[0]*za + self.beta[1]*zr
        s = self.system
        a = np.interp(ndtr(za), s.probabilities, s.quantiles[:, 1])
        def cdf(teff):
            q = teff/np.sqrt(2*a)
            prob = np.interp(q, s.quantiles[:, 0], s.probabilities, left=0., right=1.)
            return float(self.weights @ ndtr((ndtri(prob)-q_mean)/self.q_sd))
        teff = brentq(lambda t: cdf(t)-u_t, self.lower, self.upper, xtol=1.e-7)
        radius_ratio = np.interp(u_r, s.probabilities, s.quantiles[:, 2])
        return np.array([teff, radius_ratio])


class FreeRetrieval:
    def __init__(self, chain=DEFAULT_CHAIN, database=None, response=None, nz=60):
        # input_files configures the local PICASO reference-data paths.
        import input_files
        import planets
        from photochem.extensions import hotrocks
        from photochem.utils import stars
        from utils import make_lrs_data, make_joint_photometry, append_data_dicts
        self.stars = stars
        self.star, self.planet = planets.LTT1445A, planets.LTT1445Ab
        self.database = Path(database or ROOT/'picasofiles/opacities_photochem_0.1_250.0_R15000.db').resolve()
        self.response_file = Path(response or ROOT/'inputs/F1500W_response.txt').resolve()
        self.stellar_file = ROOT/'inputs/sphinx.h5'
        self.lrs_file = ROOT/'data/LTT1445Ab_Sparta_16.txt'
        self.nz = nz
        # Intersection of molecular-table domains for the four retrieved gases.
        with sqlite3.connect(self.database.as_uri()+'?mode=ro', uri=True) as connection:
            rows = [connection.execute('SELECT MIN(temperature), MAX(temperature), MIN(pressure), '
                    'MAX(pressure) FROM molecular WHERE molecule=?', (m,)).fetchone()
                    for m in ('H2O', 'CO2', 'O2', 'SO2')]
        if any(any(v is None for v in row) for row in rows):
            raise ValueError('Opacity database lacks a required gas')
        self.temperature_bounds = (max(r[0] for r in rows), min(r[1] for r in rows))
        self.pressure_bounds = (max(r[2] for r in rows), min(r[3] for r in rows))
        self.pressure_top = self.pressure_bounds[0]
        with h5py.File(self.stellar_file, 'r') as saved:
            teff_bounds = (float(saved['Teff'][:].min()), float(saved['Teff'][:].max()))
        self.system = JointSystemPrior.from_chain(chain, teq_bounds=(385., 477.), teff_bounds=teff_bounds)
        self.stellar_prior = MarginalStellarPrior(self.system)
        self.sphinx = hotrocks.sphinx_interpolator(str(self.stellar_file))
        st, pl = self.star, self.planet
        wave, flux = self.sphinx(st.Teff, st.metal, st.logg, rescale_to_Teff=False)
        # PICASO needs an incident stellar spectrum even though we use its
        # planetary surface flux, and compute the eclipse ratio ourselves.
        with NamedTemporaryFile('w') as star_file:
            np.savetxt(star_file, np.c_[wave, flux*1.e-8])  # per cm -> per Angstrom
            star_file.flush()
            self.picaso = hotrocks.PicasoThermalEmission(
                str(self.database), pl.mass, pl.radius, st.radius,
                opannection_kwargs={'wave_range': [4., 25.]},
                star_kwargs={'filename': star_file.name, 'w_unit': 'um', 'f_unit': 'flam'})
        self.wavelength = stars.grid_at_resolution(float(self.picaso.opa.wave.min()),
                                                   float(self.picaso.opa.wave.max()), 100)
        self.response = np.loadtxt(self.response_file).T
        photometry = make_joint_photometry(self.system)
        self.data = append_data_dicts(make_lrs_data(self.lrs_file), photometry)
        from utils import photon_weighted_depth
        photon_weighted_depth(self.wavelength, np.ones(len(self.wavelength)-1),
                             np.ones(len(self.wavelength)-1), *self.response)

    def prior(self, cube):
        cube = np.asarray(cube)
        return np.r_[BOUNDS[:, 0] + cube[:9]*np.diff(BOUNDS, axis=1)[:, 0],
                     self.stellar_prior.transform(cube[9:])]

    def atmosphere(self, x):
        partial = 10.**np.asarray(x[:4])
        pressure_surface = float(partial.sum())
        if not self.pressure_top < pressure_surface <= self.pressure_bounds[1]:
            return None
        pressure = np.geomspace(self.pressure_top, pressure_surface, self.nz)
        temperature = pt_profile(pressure, x[5], 10.**x[6], x[7], x[8])
        if not np.isfinite(temperature).all() or np.any(temperature < self.temperature_bounds[0]) \
                or np.any(temperature > self.temperature_bounds[1]):
            return None
        return pd.DataFrame(dict(pressure=pressure, temperature=temperature,
            **{m: np.full(self.nz, fraction) for m, fraction in
               zip(('H2O', 'CO2', 'O2', 'SO2'), partial/pressure_surface)}))

    def predict(self, x):
        from utils import photon_weighted_depth
        atmosphere = self.atmosphere(x)
        if atmosphere is None:
            return None
        self.picaso.set_custom_albedo(np.array([1.]), np.array([x[4]]))
        _, planet_flux, _ = self.picaso.fpfs(atmosphere, wavl=self.wavelength)
        wave, flux = self.sphinx(x[-2], self.star.metal, self.star.logg, rescale_to_Teff=False)
        stellar_flux = self.stars.rebin(self.stars.make_bins(wave), flux, self.wavelength)
        ratio = planet_flux/stellar_flux*x[-1]**2
        prediction = np.array([self.stars.rebin(self.wavelength, ratio, b)[0] for b in self.data['bins']])
        prediction[self.data['photometric']] = photon_weighted_depth(
            self.wavelength, ratio, stellar_flux, *self.response)
        if not np.isfinite(prediction).all() or np.any(prediction < 0):
            raise FloatingPointError('Invalid PICASO eclipse spectrum')
        return prediction

    def loglike(self, x):
        prediction = self.predict(x)
        if prediction is None:
            # An identical floor for every invalid point creates a likelihood
            # plateau, which can bias nested-sampling prior-volume accounting.
            # A varying, numerically zero floor orders invalid profiles without
            # allowing them to contribute any measurable likelihood/evidence.
            pressure = np.geomspace(self.pressure_top, float(np.sum(10.**np.asarray(x[:4]))), self.nz)
            temperature = pt_profile(pressure, x[5], 10.**x[6], x[7], x[8])
            violation = np.maximum(self.temperature_bounds[0]-temperature, 0.)**2 \
                        + np.maximum(temperature-self.temperature_bounds[1], 0.)**2
            return float(-1.e10-np.sum(violation))
        from utils import observation_loglike
        return observation_loglike(self.data, prediction)

    def metadata(self):
        from importlib.metadata import version
        stat = self.database.stat()
        return dict(case=CASE, fit_tag=FIT_TAG, param_names=PARAMETERS, independent_uniform_bounds=BOUNDS.tolist(),
            system_prior=self.system.metadata(), stellar_marginal='Conditional CDF inversion with Gaussian quadrature',
            quadrature_order=self.stellar_prior.order, temperature_bounds_K=self.temperature_bounds,
            pressure_bounds_bar=self.pressure_bounds, pressure_top_bar=self.pressure_top, nz=self.nz,
            temperature_profile='Continuous broken-linear T(log10 P), T_ref at fixed 0.1 bar; surface T is bottom T',
            invalid_profiles='Zero likelihood outside molecular opacity T/P domain; no clipping or prior renormalization',
            invalid_profile_loglike='-1e10 minus sum of squared temperature-bound violations (K^2)',
            multinest_log_zero=-1.e200,
            omitted_parameters=['Teq', 'log10chi'], equilibrium_constraint=False,
            composition='Vertically uniform, fixed by sampled surface partial-pressure fractions; no condensation',
            planet_mass_earth=self.planet.mass, planet_radius_earth=self.planet.radius,
            database=str(self.database), database_size=stat.st_size, database_mtime_ns=stat.st_mtime_ns,
            response=str(self.response_file), response_sha256=sha256(self.response_file),
            stellar_file=str(self.stellar_file), stellar_sha256=sha256(self.stellar_file),
            lrs_file=str(self.lrs_file), lrs_sha256=sha256(self.lrs_file),
            source_sha256={name: sha256(ROOT/name) for name in
                           ('retrieval_run_free.py', 'joint_system_prior.py', 'utils.py', 'planets.py')},
            versions={name: version(name) for name in ('numpy', 'scipy', 'picaso', 'photochem')},
            python_version=sys.version,
            likelihood='Gaussian LRS and two-piece Gaussian F1500W; constant normalization omitted as in joint runner',
            data={k: v.tolist() for k, v in self.data.items()})

    def check(self):
        rng = np.random.default_rng(1445)
        valid_count = sum(self.atmosphere(self.prior(u)) is not None for u in rng.random((1000, 11)))
        samples = []
        for surface_pressure, t_ref, up, lo in ((1.e-4, 600., 0., 0.),
                                               (.01, 600., -20., 50.),
                                               (1., 700., 20., 100.)):
            x = self.prior(np.full(11, .5))
            x[:4] = np.log10(surface_pressure/4.)
            x[5:9] = [t_ref, -3., up, lo]
            started = time.perf_counter()
            prediction = self.predict(x)
            if prediction is None or not np.isfinite(prediction).all():
                raise ValueError('Forward check failed')
            samples.append(dict(parameters=x.tolist(), prediction_ppm=(prediction*1.e6).tolist(),
                                seconds=time.perf_counter()-started, loglike=self.loglike(x)))
        return dict(valid_prior_fraction=valid_count/1000., forward_checks=samples)


def mpi_context():
    try:
        from mpi4py import MPI
        size = MPI.COMM_WORLD.Get_size()
        launched_size = max(int(os.environ.get(key, '1')) for key in
                            ('OMPI_COMM_WORLD_SIZE', 'PMI_SIZE'))
        if launched_size > 1 and size != launched_size:
            raise RuntimeError('MPI launcher and mpi4py communicator disagree; check MPI library compatibility')
        return MPI.COMM_WORLD, MPI.COMM_WORLD.Get_rank(), size
    except ImportError:
        if any(int(os.environ.get(key, '1')) > 1 for key in ('OMPI_COMM_WORLD_SIZE', 'PMI_SIZE', 'SLURM_NTASKS')):
            raise RuntimeError('Parallel launch detected but mpi4py is unavailable')
        return None, 0, 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chain', type=Path, default=DEFAULT_CHAIN)
    parser.add_argument('--database', type=Path)
    parser.add_argument('--response', type=Path)
    parser.add_argument('--tag', default=FIT_TAG)
    parser.add_argument('--n-live-points', type=int, default=1000)
    parser.add_argument('--nz', type=int, default=60)
    parser.add_argument('--seed', type=int, default=1445)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if Path(args.tag).name != args.tag or args.tag in ('', '.', '..'):
        parser.error('--tag must be one directory name')
    if args.n_live_points < 1 or args.nz < 3:
        parser.error('Require positive live points and at least three atmospheric levels')
    if args.report and not args.check_only:
        parser.error('--report requires --check-only')
    comm, rank, size = mpi_context()
    try:
        # Verify the MPI-linked native library during MPI preflight as well.
        if size > 1:
            native = importlib.import_module('pymultinest.run')
            if getattr(native, 'lib_mpi', None) is None:
                raise RuntimeError('MPI MultiNest library unavailable; refusing independent serial runs')
        run = FreeRetrieval(args.chain, args.database, args.response, args.nz)
        if args.check_only:
            if rank == 0:
                report = dict(run.metadata(), checks=run.check())
                if args.report:
                    args.report.parent.mkdir(parents=True, exist_ok=True)
                    with args.report.open('x') as stream:
                        json.dump(report, stream, indent=2)
                print(json.dumps(report['checks'], indent=2), flush=True)
            if comm:
                comm.Barrier()
            return
        # All ranks call native MultiNest, but only rank 0 prepares/analyzes files.
        native = importlib.import_module('pymultinest.run')
        if size > 1 and getattr(native, 'lib_mpi', None) is None:
            raise RuntimeError('MPI MultiNest library unavailable; refusing independent serial runs')
        directory = ROOT/'pymultinest_free'/args.tag/CASE
        if rank == 0:
            directory.mkdir(parents=True, exist_ok=False)
            metadata = dict(run.metadata(), mpi_size=size, n_live_points=args.n_live_points, seed=args.seed)
            (directory/'inputs.json').write_text(json.dumps(metadata, indent=2)+'\n')
        if comm:
            comm.Barrier()
        # MultiNest has short native filename buffers. All paths in run are absolute.
        os.chdir(directory)
        from pymultinest.solve import solve
        result = solve(LogLikelihood=run.loglike, Prior=run.prior, n_dims=len(PARAMETERS),
                       outputfiles_basename='free_', n_live_points=args.n_live_points,
                       seed=args.seed, verbose=(rank == 0), resume=False, init_MPI=False, use_MPI=True,
                       log_zero=-1.e200)
        if rank == 0:
            with open('free_.pkl', 'wb') as stream:
                pickle.dump(result, stream)
            posterior = np.atleast_2d(np.loadtxt('free_post_equal_weights.dat'))
            best = float(posterior[:, -1].max())
            (directory/'fit_summary.json').write_text(json.dumps(dict(
                logZ=result['logZ'], logZerr=result['logZerr'], best_sampled_chi2=-2*best,
                AIC=2*len(PARAMETERS)-2*best, posterior_samples=len(posterior)), indent=2)+'\n')
        if comm:
            comm.Barrier()
    except BaseException:
        if comm and size > 1:
            import traceback
            traceback.print_exc()
            comm.Abort(1)
        raise


if __name__ == '__main__':
    main()
