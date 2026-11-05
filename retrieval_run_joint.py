"""First-pass retrievals with the non-RV-GP joint fit and truncated Teq support.

Run in conda environment ltt from the repository root. Historical cases and
outputs in retrieval_run.py / pymultinest are never modified by this runner.
"""
import argparse
from functools import partial
import json
import os
from pathlib import Path
import pickle

import h5py
import numpy as np

import retrieval_run as legacy
import planets
from photochem.utils import stars
from utils import make_lrs_data, make_F1500W_data_real, append_data_dicts, photon_weighted_depth
from joint_system_prior import JointSystemPrior, DEFAULT_CHAIN, ROOT, sha256

def case_names(lrs_bins=16):
    if lrs_bins not in (8, 16):
        raise ValueError('LRS bin count must be 8 or 16')
    return (f'rock_{lrs_bins}', f'atm_{lrs_bins}', 'rock_F1500W_obs', 'atm_F1500W_obs',
            f'rock_{lrs_bins}_F1500W_obs', f'atm_{lrs_bins}_F1500W_obs')


CASE_NAMES = case_names()
OUTPUT_ROOT = ROOT / 'pymultinest_joint'


class JointRetrieval:
    def __init__(self, chain=DEFAULT_CHAIN, response=legacy.DEFAULT_RESPONSE_FILE, lrs_bins=16):
        self.case_names = case_names(lrs_bins)
        self.lrs_bins = lrs_bins
        self.grid_file = ROOT / 'results/LTT1445Ab.h5'
        self.stellar_file = ROOT / 'inputs/sphinx.h5'
        self.lrs_file = ROOT / f'data/LTT1445Ab_Sparta_{lrs_bins}.txt'
        with h5py.File(self.grid_file) as f:
            names = list(f['gridnames'].asstr()[:])
            teq_axis = f['gridvals'][str(names.index('Teq'))][:]
            self.wavelength = f['common/wavl'][:]
        with h5py.File(self.stellar_file) as f:
            teff_axis = f['Teff'][:]
        self.system = JointSystemPrior.from_chain(
            chain, teq_bounds=(teq_axis.min(), teq_axis.max()),
            teff_bounds=(teff_axis.min(), teff_axis.max()))
        self.response_file = Path(response).resolve()
        self.response = np.loadtxt(self.response_file).T
        photon_weighted_depth(self.wavelength, np.ones(len(self.wavelength)-1),
                              np.ones(len(self.wavelength)-1), *self.response)
        # Copy the display/bin convention, without changing the legacy 148+/-11
        # helper or any of its case definitions.
        f1500 = make_F1500W_data_real()
        f1500['fpfs'] = np.array([self.system.eclipse_mean])
        f1500['err'] = np.array([self.system.eclipse_sd])
        lrs = make_lrs_data(self.lrs_file)
        combined = append_data_dicts(lrs, f1500)
        datasets = {str(lrs_bins): lrs, 'F1500W_obs': f1500, f'{lrs_bins}_F1500W_obs': combined}
        self.cases = {}
        for name in self.case_names:
            kind, dataset = name.split('_', 1)
            data = datasets[dataset]
            raw = legacy.model_atm_raw if kind == 'atm' else legacy.model_rock_raw
            params = (['log10PH2O', 'log10PCO2', 'log10PO2', 'log10PSO2', 'log10chi', 'albedo']
                      if kind == 'atm' else ['albedo'])+['Teq', 'Teff', 'R_planet_star']
            predict = partial(self.predict, raw=raw, data=data)
            self.cases[name] = {'param_names': params, 'data_dict': data,
                                'prior': partial(self.prior, kind=kind), 'predict': predict,
                                'model_raw': raw, 'loglike': partial(self.loglike, predict=predict, data=data)}

    def prior(self, cube, kind):
        cube = np.asarray(cube, dtype=float)
        size = 9 if kind == 'atm' else 4
        if cube.shape != (size,):
            raise ValueError(f'Expected {size} coordinates for {kind}')
        result = np.empty(size)
        if kind == 'atm':
            bounds = [(-7., 2.), (-7., 2.), (-5., 2.), (-7., 2.),
                      (np.log10(.05), np.log10(.8)), (0., .4)]
            for i, (lo, hi) in enumerate(bounds):
                result[i] = lo+cube[i]*(hi-lo)
        else:
            result[0] = .4*cube[0]
        result[-3:] = self.system.transform(cube[-3:])
        return result

    def predict(self, parameters, *, raw, data):
        # Reuse historical physical spectra, with an instance-local observation
        # operator. Never change legacy.SYSTEM_PRIORS or legacy.F1500W_RESPONSE.
        ratio = raw(parameters, self.wavelength)
        prediction = np.array([stars.rebin(self.wavelength, ratio, b)[0] for b in data['bins']])
        photometric = data.get('photometric', np.zeros(len(prediction), dtype=bool))
        if np.any(photometric):
            st = planets.LTT1445A
            wave, flux = legacy.SPHINX(parameters[-2], st.metal, st.logg, rescale_to_Teff=False)
            flux = stars.rebin(stars.make_bins(wave), flux, self.wavelength)
            prediction[photometric] = photon_weighted_depth(
                self.wavelength, ratio, flux, *self.response)
        return prediction

    @staticmethod
    def loglike(parameters, *, predict, data):
        prediction = predict(parameters)
        if not np.all(np.isfinite(prediction)):
            return -1.e100
        # Same convention as historical runs; constants cancel for matched data.
        return float(-.5*np.sum(((data['fpfs']-prediction)/data['err'])**2))

    def metadata(self, validation):
        st = planets.LTT1445A
        return {
            'system_prior': self.system.metadata(), 'prior_validation': validation,
            'system_fit': 'non-RV-GP, all transits and eclipses, planet b',
            'photometry': 'photon-weighted planet/star flux integrals',
            'response_file': str(self.response_file), 'response_sha256': sha256(self.response_file),
            'response': self.response.tolist(),
            'lrs_file': str(self.lrs_file), 'lrs_sha256': sha256(self.lrs_file),
            'lrs_bins': self.lrs_bins,
            'stellar_spectrum': {'file': str(self.stellar_file), 'sha256': sha256(self.stellar_file),
                                 'metallicity': st.metal, 'logg': st.logg},
            'grid_file': str(self.grid_file),
            'grid_stat': {'size': self.grid_file.stat().st_size,
                          'mtime_ns': self.grid_file.stat().st_mtime_ns},
            'grid_fixed_mass_earth': planets.LTT1445Ab.mass,
            'grid_fixed_radius_earth': planets.LTT1445Ab.radius,
            'grid_reused_without_new_climate_runs': True,
            'likelihood': 'independent Gaussian errors; additive normalization omitted, as in legacy runs',
            'source_sha256': {name: sha256(ROOT/name) for name in
                              ['retrieval_run_joint.py', 'joint_system_prior.py', 'retrieval_run.py',
                               'utils.py', 'planets.py', 'gridutils.py']},
        }

    def check_forward_models(self):
        checks = {}
        for name, case in self.cases.items():
            n = len(case['param_names'])
            for q in (0., .5, 1.):
                parameters = case['prior'](np.full(n, q))
                value = case['loglike'](parameters)
                if not np.isfinite(value) or value <= -1.e99:
                    raise ValueError(f'Invalid forward model for {name}, cube quantile {q}')
            parameters = case['prior'](np.full(n, .5))
            checks[name] = {'center_parameters': parameters.tolist(),
                            'center_prediction_ppm': (case['predict'](parameters)*1.e6).tolist(),
                            'center_loglike': case['loglike'](parameters),
                            'finite_at_unit_endpoints_and_center': True}
        return checks


def write_fit_diagnostics(case, basename):
    """Retain the historical diagnostic format without adding new legacy cases."""
    samples = np.atleast_2d(np.loadtxt(str(basename)+'post_equal_weights.dat'))
    k = len(case['param_names'])
    if samples.shape[1] != k+1:
        raise ValueError('Unexpected posterior shape for fit diagnostics')
    max_loglike = float(samples[:, -1].max())
    with h5py.File(str(basename)+'_aic.h5', 'w') as f:
        for key, value in dict(max_loglike=max_loglike, k=k,
                               chi2_min=-2*max_loglike, AIC=2*k-2*max_loglike).items():
            f.create_dataset(key, data=value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chain', type=Path, default=DEFAULT_CHAIN)
    parser.add_argument('--response', type=Path, default=legacy.DEFAULT_RESPONSE_FILE)
    parser.add_argument('--lrs-bins', type=int, choices=(8, 16), default=16)
    parser.add_argument('--cases', nargs='+', choices=sorted(set(case_names(8)+case_names(16))))
    parser.add_argument('--list-cases', action='store_true')
    parser.add_argument('--check-only', action='store_true', help='Validate prior and forward models without sampling')
    parser.add_argument('--report', type=Path, help='Optional JSON validation report for --check-only')
    parser.add_argument('--tag', default='nonrv_gp_v1')
    parser.add_argument('--n-live-points', type=int, default=1000)
    args = parser.parse_args()
    available_cases = case_names(args.lrs_bins)
    if args.cases is None:
        args.cases = list(available_cases)
    if any(name not in available_cases for name in args.cases):
        parser.error('Selected case does not match --lrs-bins')
    if args.list_cases:
        print('\n'.join(available_cases))
        return
    if not args.tag or Path(args.tag).name != args.tag or args.tag in ('.', '..'):
        parser.error('--tag must be a single directory name')
    if args.n_live_points < 1:
        parser.error('--n-live-points must be positive')
    if args.report and not args.check_only:
        parser.error('--report requires --check-only')
    if len(set(args.cases)) != len(args.cases):
        parser.error('Duplicate cases are not allowed')
    run = JointRetrieval(args.chain, args.response, lrs_bins=args.lrs_bins)
    validation = run.system.validate()
    if not validation['passes_preliminary_checks']:
        raise ValueError('Joint copula approximation failed validation: '+json.dumps(validation))
    metadata = run.metadata(validation)
    if args.check_only:
        metadata['forward_model_checks'] = run.check_forward_models()
        if args.report:
            # A validation report must not overwrite inputs or historical files.
            report = args.report.resolve()
            if report.suffix != '.json' or report.exists():
                parser.error('--report must be a new .json file')
            if ROOT/'pymultinest' in report.parents or report in [run.system.source, run.response_file]:
                parser.error('--report cannot target historical outputs or inputs')
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps(metadata, indent=2)+'\n')
        summary = {'depth_ppm': run.system.eclipse_mean*1.e6,
                   'error_ppm': run.system.eclipse_sd*1.e6,
                   'system_prior': run.system.metadata(), 'prior_validation': validation,
                   'forward_model_checks': metadata['forward_model_checks']}
        print(json.dumps(summary, indent=2))
        return
    root = OUTPUT_ROOT / args.tag
    for name in args.cases:
        if (root/name).exists():
            parser.error(f'Output exists: {root/name}; choose a fresh --tag')
    from pymultinest.solve import solve
    for name in args.cases:
        case = run.cases[name]
        directory = root/name
        directory.mkdir(parents=True)
        # Short relative names avoid native MultiNest filename-buffer limits
        # under a long absolute checkout path. Files stay in the same directory.
        basename = os.path.relpath(directory/name)
        inputs = dict(metadata, case=name, param_names=case['param_names'], n_live_points=args.n_live_points,
                      data={k: v.tolist() for k, v in case['data_dict'].items()})
        (directory/'inputs.json').write_text(json.dumps(inputs, indent=2)+'\n')
        result = solve(LogLikelihood=case['loglike'], Prior=case['prior'],
                       n_dims=len(case['param_names']), outputfiles_basename=basename,
                       n_live_points=args.n_live_points, verbose=True, resume=False)
        with open(basename+'.pkl', 'wb') as f:
            pickle.dump(result, f)
        write_fit_diagnostics(case, basename)


if __name__ == '__main__':
    main()
