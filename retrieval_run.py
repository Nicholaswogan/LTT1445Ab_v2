import input_files

import numpy as np
from photochem.extensions import hotrocks
from photochem.utils import stars
import planets
from scipy.stats import truncnorm
import os
import pickle
import h5py
from pathlib import Path
import gridutils
from utils import photon_weighted_depth, make_lrs_data, make_lrs_and_F1500W_data, make_lrs_and_F1500W_data_real, make_F1500W_data_real

def make_interpolators(filename):
    g = gridutils.GridInterpolator(filename)

    wavl = g.common['wavl']
    spectra = g.make_interpolator('fp', bounds_mode='error')
    press = g.make_interpolator('P',logspace=True, bounds_mode='error')
    temp = g.make_interpolator('T', bounds_mode='error')

    return wavl, spectra, press, temp

def quantile_to_uniform(quantile, lower_bound, upper_bound):
    return quantile*(upper_bound - lower_bound) + lower_bound

def model_atm_raw(x, wavl):
    _ensure_initialized()

    log10PH2O, log10PCO2, log10PO2, log10PSO2, log10chi, albedo, Teq, Teff, R_planet_star = x

    chi = 10.0**log10chi
    # Roundoff in 10**log10(bound) must not invalidate a prior endpoint.
    for bound in (0.05, 0.8):
        if abs(chi-bound) <= 4*np.spacing(bound):
            chi = bound

    # Inputs to model grid
    y = np.array([log10PH2O, log10PCO2, log10PO2, log10PSO2, chi, albedo, Teq])
    F_planet = SPECTRA(y)
    F_planet = stars.rebin(WAVL, F_planet, wavl)

    # Stellar flux
    st = planets.LTT1445A
    wv_star, F_star = SPHINX(Teff, st.metal, st.logg, rescale_to_Teff=False) # CGS units
    wavl_star = stars.make_bins(wv_star)
    F_star = stars.rebin(wavl_star, F_star, wavl) # rebin

    # fpfs
    fpfs = F_planet/F_star * (R_planet_star)**2

    return fpfs

def model_atm(x, wv_bins):
    _ensure_initialized()
    wavl = WAVL
    fpfs1 = model_atm_raw(x, wavl)

    fpfs = np.empty(len(wv_bins))
    for i,b in enumerate(wv_bins):
        fpfs[i] = stars.rebin(wavl, fpfs1, b)[0]

    return fpfs

def prior_atm(cube):
    params = np.zeros_like(cube)
    params[0] = quantile_to_uniform(cube[0], -7, 2) # log10PH2O
    params[1] = quantile_to_uniform(cube[1], -7, 2) # log10PCO2
    params[2] = quantile_to_uniform(cube[2], -5, 2) # log10PO2
    params[3] = quantile_to_uniform(cube[3], -7, 2) # log10PSO2
    params[4] = quantile_to_uniform(cube[4], np.log10(0.05), np.log10(0.8)) # log10chi
    params[5] = quantile_to_uniform(cube[5], 0, 0.4) # albedo
    params[6] = system_prior("Teq", cube[6]) # Teq
    params[7] = system_prior("Teff", cube[7]) # Teff
    params[8] = system_prior("R_planet_star", cube[8]) # R_planet_star
    return params

def model_atm_hot_raw(x, wavl):
    _ensure_initialized()

    log10PH2O, log10PCO2, log10PO2, log10PSO2, albedo, Teq, Teff, R_planet_star = x

    # Inputs to model grid
    y = np.array([log10PH2O, log10PCO2, log10PO2, log10PSO2, albedo, Teq])
    F_planet = SPECTRA_HOT(y)
    F_planet = stars.rebin(WAVL, F_planet, wavl)

    # Stellar flux
    st = planets.LTT1445A
    wv_star, F_star = SPHINX(Teff, st.metal, st.logg, rescale_to_Teff=False) # CGS units
    wavl_star = stars.make_bins(wv_star)
    F_star = stars.rebin(wavl_star, F_star, wavl) # rebin

    # fpfs
    fpfs = F_planet/F_star * (R_planet_star)**2

    return fpfs

def model_atm_hot(x, wv_bins):
    _ensure_initialized()
    wavl = WAVL
    fpfs1 = model_atm_hot_raw(x, wavl)

    fpfs = np.empty(len(wv_bins))
    for i,b in enumerate(wv_bins):
        fpfs[i] = stars.rebin(wavl, fpfs1, b)[0]

    return fpfs

def prior_atm_hot(cube):
    params = np.zeros_like(cube)
    params[0] = quantile_to_uniform(cube[0], -7, 2) # log10PH2O
    params[1] = quantile_to_uniform(cube[1], -7, 2) # log10PCO2
    params[2] = quantile_to_uniform(cube[2], -5, 2) # log10PO2
    params[3] = quantile_to_uniform(cube[3], -7, 2) # log10PSO2
    params[4] = quantile_to_uniform(cube[4], 0, 0.4) # albedo
    params[5] = system_prior("Teq", cube[5]) # Teq
    params[6] = system_prior("Teff", cube[6]) # Teff
    params[7] = system_prior("R_planet_star", cube[7]) # R_planet_star
    return params

def model_rock_raw(x, wavl):
    _ensure_initialized()

    albedo, Teq, Teff, R_planet_star = x

    # Gray surface: epsilon = 1 - albedo, so absorption and emission
    # factors cancel in Tday. The helper applies (1-albedo), hence this
    # compensating inverse call; see the bare-rock equation in the paper.
    flux = stars.equilibrium_temperature_inverse(Teq, albedo)
    Tday = hotrocks.bare_rock_dayside_temperature(flux, albedo, 2/3)

    # Planet flux
    wv_av = (wavl[1:] + wavl[:-1])/2
    F_planet = (1 - albedo)*stars.blackbody_cgs(Tday, wv_av/1e4)*np.pi

    # Stellar flux
    st = planets.LTT1445A
    wv_star, F_star = SPHINX(Teff, st.metal, st.logg, rescale_to_Teff=False) # CGS units
    wavl_star = stars.make_bins(wv_star)
    F_star = stars.rebin(wavl_star, F_star, wavl) # rebin

    # fpfs
    fpfs = F_planet/F_star * (R_planet_star)**2

    return fpfs

def model_rock(x, wv_bins):
    _ensure_initialized()
    wavl = WAVL
    fpfs1 = model_rock_raw(x, wavl)

    fpfs = np.empty(len(wv_bins))
    for i,b in enumerate(wv_bins):
        fpfs[i] = stars.rebin(wavl, fpfs1, b)[0]

    return fpfs

def prior_rock(cube):
    params = np.zeros_like(cube)
    params[0] = quantile_to_uniform(cube[0], 0, 0.4) # albedo
    params[1] = system_prior("Teq", cube[1]) # Teq
    params[2] = system_prior("Teff", cube[2]) # Teff
    params[3] = system_prior("R_planet_star", cube[3]) # R_planet_star
    return params

def predict_data(model, cube, data_dict):
    """Predictions in data order, including the F1500W observation operator."""
    resulty = model(cube, data_dict['bins'])
    if 'photometric' in data_dict and np.any(data_dict['photometric']):
        if F1500W_RESPONSE is None:
            if not ALLOW_TOPHAT:
                raise ValueError('Observed F1500W requires --response or explicit --allow-tophat.')
        else:
            raw = {model_atm: model_atm_raw, model_atm_hot: model_atm_hot_raw,
                   model_rock: model_rock_raw}[model]
            ratio = raw(cube, WAVL)
            st = planets.LTT1445A
            wv, fs = SPHINX(cube[-2], st.metal, st.logg, rescale_to_Teff=False)
            fs = stars.rebin(stars.make_bins(wv), fs, WAVL)
            resulty[data_dict['photometric']] = photon_weighted_depth(
                WAVL, ratio, fs, *F1500W_RESPONSE)
    return resulty


def make_loglike(model, data_dict):
    def loglike(cube):
        y = data_dict['fpfs']
        e = data_dict['err']
        resulty = predict_data(model, cube, data_dict)
        if not np.all(np.isfinite(resulty)):
            return -1.0e100 # outside implicit priors
        loglikelihood = -0.5*np.sum((y - resulty)**2/e**2)
        return loglikelihood
    return loglike

def make_loglike_prior(data_dict, param_names, model, model_raw, prior):

    loglike = make_loglike(model, data_dict)

    out = {
        'loglike': loglike,
        'prior': prior,
        'param_names': param_names,
        'data_dict': data_dict,
        'model': model,
        'model_raw': model_raw,
        'predict': lambda cube: predict_data(model, cube, data_dict),
    }

    return out

def make_cases():

    cases = {}

    param_names_atm = [
        'log10PH2O', 'log10PCO2', 'log10PO2', 'log10PSO2', 'log10chi',
        'albedo', 'Teq', 'Teff', 'R_planet_star'
    ]
    param_names_atm_hot = [
        'log10PH2O', 'log10PCO2', 'log10PO2', 'log10PSO2',
        'albedo', 'Teq', 'Teff', 'R_planet_star'
    ]
    param_names_rock = [
        'albedo', 'Teq', 'Teff', 'R_planet_star'
    ]

    # 8 bin data
    data_dict = make_lrs_data('data/LTT1445Ab_Sparta_8.txt')
    cases['rock_8'] = make_loglike_prior(data_dict, param_names_rock, model_rock, model_rock_raw, prior_rock)
    cases['atm_8'] = make_loglike_prior(data_dict, param_names_atm, model_atm, model_atm_raw, prior_atm)
    cases['atm_hot_8'] = make_loglike_prior(data_dict, param_names_atm_hot, model_atm_hot, model_atm_hot_raw, prior_atm_hot)

    # 16 bin data
    data_dict = make_lrs_data('data/LTT1445Ab_Sparta_16.txt')
    cases['rock_16'] = make_loglike_prior(data_dict, param_names_rock, model_rock, model_rock_raw, prior_rock)
    cases['atm_16'] = make_loglike_prior(data_dict, param_names_atm, model_atm, model_atm_raw, prior_atm)
    cases['atm_hot_16'] = make_loglike_prior(data_dict, param_names_atm_hot, model_atm_hot, model_atm_hot_raw, prior_atm_hot)

    # 1 bar O2, 0.01 bar CO2, and 0.2 albedo.
    errors = np.arange(5, 41, 5)
    for err in errors:
        data_dict = make_lrs_and_F1500W_data(
            filename='data/LTT1445Ab_Sparta_16.txt',
            fpfs_15=56.49e-6,
            ntrans=1,
            err_one_transit=err*1.0e-6
        )
        cases[f'rock_16_F1500W_{err}ppm'] = make_loglike_prior(data_dict, param_names_rock, model_rock, model_rock_raw, prior_rock)
        cases[f'atm_16_F1500W_{err}ppm'] = make_loglike_prior(data_dict, param_names_atm, model_atm, model_atm_raw, prior_atm)

    # 16 bin data + observed 15 micron point
    data_dict = make_lrs_and_F1500W_data_real('data/LTT1445Ab_Sparta_16.txt')
    cases['rock_16_F1500W_obs'] = make_loglike_prior(data_dict, param_names_rock, model_rock, model_rock_raw, prior_rock)
    cases['atm_hot_16_F1500W_obs'] = make_loglike_prior(data_dict, param_names_atm_hot, model_atm_hot, model_atm_hot_raw, prior_atm_hot)
    cases['atm_16_F1500W_obs'] = make_loglike_prior(data_dict, param_names_atm, model_atm, model_atm_raw, prior_atm)

    data_dict = make_F1500W_data_real()
    cases['rock_F1500W_obs'] = make_loglike_prior(data_dict, param_names_rock, model_rock, model_rock_raw, prior_rock)
    cases['atm_hot_F1500W_obs'] = make_loglike_prior(data_dict, param_names_atm_hot, model_atm_hot, model_atm_hot_raw, prior_atm_hot)
    cases['atm_F1500W_obs'] = make_loglike_prior(data_dict, param_names_atm, model_atm, model_atm_raw, prior_atm)

    return cases

def compute_AIC(model_name, root="pymultinest"):
    # Compute maximum likelihood, k, and AIC for one retrieval, then save the
    # result as `pymultinest/{model_name}/{model_name}_aic.h5`.
    if model_name not in RETRIEVAL_CASES:
        raise ValueError(f"Unknown model_name `{model_name}`.")

    case = RETRIEVAL_CASES[model_name]
    dirname = os.path.join(root, model_name)
    post_file = os.path.join(dirname, f"{model_name}post_equal_weights.dat")
    if not os.path.exists(post_file):
        raise FileNotFoundError(f"Could not find posterior sample file: {post_file}")

    samples = np.loadtxt(post_file)
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    if samples.shape[1] < 2:
        raise ValueError(f"Unexpected shape for posterior samples in {post_file}: {samples.shape}")

    max_loglike = float(np.max(samples[:, -1]))
    k = len(case["param_names"])
    chi2_min = -2.0 * max_loglike
    aic = 2.0 * k - 2.0 * max_loglike

    summary = {
        "max_loglike": max_loglike,
        "k": k,
        "chi2_min": chi2_min,
        "AIC": aic,
    }

    outfile = os.path.join(dirname, f"{model_name}_aic.h5")
    with h5py.File(outfile, "w") as f:
        for key, val in summary.items():
            if isinstance(val, str):
                f.create_dataset(key, data=np.array(val, dtype=object), dtype=h5py.string_dtype())
            else:
                f.create_dataset(key, data=val)

    return summary


# Legacy priors remain explicit until a higher precision system fit is supplied.
SYSTEM_PRIORS = {'Teq': [431.0, 23.0, 385.0, 477.0],
                 'Teff': [3340.0, 150.0, 3040.0, 3640.0],
                 'R_planet_star': [0.0454, 0.0012, 0.0430, 0.0478]}
DEFAULT_RESPONSE_FILE = Path(__file__).resolve().parent / 'inputs/F1500W_response.txt'
F1500W_RESPONSE = np.loadtxt(DEFAULT_RESPONSE_FILE).T
ALLOW_TOPHAT = False


def system_prior(name, quantile):
    mean, sd, lower, upper = SYSTEM_PRIORS[name]
    return truncnorm((lower-mean)/sd, (upper-mean)/sd, loc=mean, scale=sd).ppf(quantile)


def configure_priors(priors):
    """Explicit independent truncated normals: [mean, sd, lower, upper]."""
    if set(priors) != set(SYSTEM_PRIORS):
        raise ValueError('Supply Teq, Teff and R_planet_star priors.')
    for name, values in priors.items():
        if len(values) != 4 or not np.all(np.isfinite(values)):
            raise ValueError(f'Invalid prior for {name}')
        mean, sd, lo, hi = values
        if sd <= 0 or not 0 < lo < hi or not lo <= mean <= hi:
            raise ValueError(f'Invalid prior for {name}; rounded zero errors are not valid.')
    # Check actual saved grid axes, never silently clip a revised prior.
    for filename in ('results/LTT1445Ab.h5', 'results/LTT1445Ab_hot.h5'):
        with h5py.File(filename) as f:
            axis = f['gridvals'][str(len(f['gridvals'])-1)][:]
        if priors['Teq'][2] < axis.min() or priors['Teq'][3] > axis.max():
            raise ValueError(f'Teq prior extends beyond {filename}: {axis.min()}–{axis.max()} K')
    SYSTEM_PRIORS.update(priors)


def _ensure_initialized():
    if 'WAVL' not in globals():
        initialize_retrieval()


def __getattr__(name):
    # Preserve notebook access to lazily loaded grid and SPHINX globals.
    if name in {'WAVL', 'SPECTRA', 'PRESS', 'TEMP', 'WAVL_HOT', 'SPECTRA_HOT',
                'PRESS_HOT', 'TEMP_HOT', 'SPHINX'}:
        _ensure_initialized()
        return globals()[name]
    raise AttributeError(name)


def initialize_retrieval():
    global WAVL, SPECTRA, PRESS, TEMP, WAVL_HOT, SPECTRA_HOT, PRESS_HOT, TEMP_HOT, SPHINX
    WAVL, SPECTRA, PRESS, TEMP = make_interpolators('results/LTT1445Ab.h5')
    WAVL_HOT, SPECTRA_HOT, PRESS_HOT, TEMP_HOT = make_interpolators('results/LTT1445Ab_hot.h5')
    if not np.array_equal(WAVL, WAVL_HOT):
        raise ValueError('Nominal and hot grids have different wavelength bins.')
    SPHINX = hotrocks.sphinx_interpolator('inputs/sphinx.h5')


RETRIEVAL_CASES = make_cases()


def main():
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description='LTT 1445 Ab eclipse inference')
    parser.add_argument('--cases', nargs='+', choices=sorted(RETRIEVAL_CASES),
                        default=['rock_16_F1500W_obs', 'atm_16_F1500W_obs',
                                 'rock_F1500W_obs', 'atm_F1500W_obs'])
    parser.add_argument('--list-cases', action='store_true')
    parser.add_argument('--priors', type=Path, help='JSON with explicit independent system priors')
    photometry = parser.add_mutually_exclusive_group()
    photometry.add_argument('--response', type=Path, default=DEFAULT_RESPONSE_FILE,
                            help='Photon response file (default: inputs/F1500W_response.txt)')
    photometry.add_argument('--allow-tophat', action='store_true', help='Use the preliminary rectangular-bin approximation instead of the response')
    parser.add_argument('--tag', default='preliminary_legacy', help='Separate output namespace; use a new tag for changed inputs')
    parser.add_argument('--n-live-points', type=int, default=1000)
    args = parser.parse_args()
    if args.list_cases:
        print('\n'.join(sorted(RETRIEVAL_CASES)))
        return
    if not args.tag or Path(args.tag).name != args.tag or args.tag in ('.', '..'):
        parser.error('--tag must be a single directory name')
    if args.n_live_points < 1:
        parser.error('--n-live-points must be positive')
    if args.priors:
        configure_priors(json.loads(args.priors.read_text()))
    global F1500W_RESPONSE, ALLOW_TOPHAT
    ALLOW_TOPHAT = args.allow_tophat
    F1500W_RESPONSE = None
    if not ALLOW_TOPHAT:
        F1500W_RESPONSE = np.loadtxt(args.response).T
        # Validate shape, support, and response before starting sampling.
        with h5py.File('results/LTT1445Ab.h5') as f:
            edges = f['common/wavl'][:]
        photon_weighted_depth(edges, np.ones(len(edges)-1), np.ones(len(edges)-1), *F1500W_RESPONSE)
    if any('obs' in name for name in args.cases) and F1500W_RESPONSE is None and not ALLOW_TOPHAT:
        parser.error('Observed F1500W requires --response, or --allow-tophat for preliminary runs')
    root = Path('pymultinest') / args.tag
    for name in args.cases:
        if (root / name).exists():
            parser.error(f'Output directory already exists: {root / name}. Choose a fresh --tag.')
    from pymultinest.solve import solve
    initialize_retrieval()
    for name in args.cases:
        case = RETRIEVAL_CASES[name]
        dirname = root / name
        dirname.mkdir(parents=True)
        basename = str(dirname / name)
        metadata = {'case': name, 'system_priors': SYSTEM_PRIORS,
                    'prior_assumption': 'independent truncated normals',
                    'response': None if F1500W_RESPONSE is None else F1500W_RESPONSE.tolist(),
                    'photometry': 'rectangular ratio average' if F1500W_RESPONSE is None else 'photon weighted flux ratio',
                    'data': {k: v.tolist() for k, v in case['data_dict'].items()},
                    'n_live_points': args.n_live_points,
                    'grid_files': ['results/LTT1445Ab.h5', 'results/LTT1445Ab_hot.h5']}
        (dirname / 'inputs.json').write_text(json.dumps(metadata, indent=2)+'\n')
        results = solve(LogLikelihood=case['loglike'], Prior=case['prior'],
                        n_dims=len(case['param_names']), outputfiles_basename=basename,
                        verbose=True, n_live_points=args.n_live_points, resume=False)
        with open(basename+'.pkl', 'wb') as f:
            pickle.dump(results, f)
        compute_AIC(name, root=str(root))


if __name__ == '__main__':
    main()
