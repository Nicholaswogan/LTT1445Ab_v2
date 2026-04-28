import input_files

import numpy as np
from photochem.extensions import hotrocks
from photochem.utils import stars
import planets
import LTT1445Ab_grid
from scipy.stats import truncnorm
import os 
import pickle
import h5py
from pymultinest.solve import solve
import gridutils
from utils import make_lrs_data, make_F1500W_data

def make_interpolators(filename):
    g = gridutils.GridInterpolator(filename)

    wavl = g.common['wavl']
    spectra = g.make_interpolator('fp')
    press = g.make_interpolator('P',logspace=True)
    temp = g.make_interpolator('T')

    return wavl, spectra, press, temp

def quantile_to_uniform(quantile, lower_bound, upper_bound):
    return quantile*(upper_bound - lower_bound) + lower_bound

def model_atm_raw(x, wavl):

    log10PH2O, log10PCO2, log10PO2, log10PSO2, log10chi, albedo, Teq, Teff, R_planet_star = x

    chi = 10.0**log10chi

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
    wavl = LTT1445Ab_grid.WAVL
    fpfs1 = model_atm_raw(x, wavl)

    fpfs = np.empty(len(wv_bins))
    for i,b in enumerate(wv_bins):
        fpfs[i] = stars.rebin(wavl, fpfs1, b)

    return fpfs

def prior_atm(cube):
    params = np.zeros_like(cube)
    params[0] = quantile_to_uniform(cube[0], -7, 2) # log10PH2O
    params[1] = quantile_to_uniform(cube[1], -7, 2) # log10PCO2
    params[2] = quantile_to_uniform(cube[2], -5, 2) # log10PO2
    params[3] = quantile_to_uniform(cube[3], -7, 2) # log10PSO2
    params[4] = quantile_to_uniform(cube[4], np.log10(0.05), np.log10(0.8)) # log10chi
    params[5] = quantile_to_uniform(cube[5], 0, 0.4) # albedo
    params[6] = truncnorm(-2, 2, loc=431, scale=23).ppf(cube[6]) # Teq
    params[7] = truncnorm(-2, 2, loc=3340, scale=150).ppf(cube[7]) # Teff
    params[8] = truncnorm(-2, 2, loc=0.0454, scale=0.0012).ppf(cube[8]) # R_planet_star
    return params  

def model_atm_hot_raw(x, wavl):

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
    wavl = LTT1445Ab_grid.WAVL
    fpfs1 = model_atm_hot_raw(x, wavl)

    fpfs = np.empty(len(wv_bins))
    for i,b in enumerate(wv_bins):
        fpfs[i] = stars.rebin(wavl, fpfs1, b)

    return fpfs

def prior_atm_hot(cube):
    params = np.zeros_like(cube)
    params[0] = quantile_to_uniform(cube[0], -7, 2) # log10PH2O
    params[1] = quantile_to_uniform(cube[1], -7, 2) # log10PCO2
    params[2] = quantile_to_uniform(cube[2], -5, 2) # log10PO2
    params[3] = quantile_to_uniform(cube[3], -7, 2) # log10PSO2
    params[4] = quantile_to_uniform(cube[4], 0, 0.4) # albedo
    params[5] = truncnorm(-2, 2, loc=431, scale=23).ppf(cube[5]) # Teq
    params[6] = truncnorm(-2, 2, loc=3340, scale=150).ppf(cube[6]) # Teff
    params[7] = truncnorm(-2, 2, loc=0.0454, scale=0.0012).ppf(cube[7]) # R_planet_star
    return params   

def model_rock_raw(x, wavl):

    albedo, Teq, Teff, R_planet_star = x

    # Compute the dayside temperature
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
    wavl = LTT1445Ab_grid.WAVL
    fpfs1 = model_rock_raw(x, wavl)

    fpfs = np.empty(len(wv_bins))
    for i,b in enumerate(wv_bins):
        fpfs[i] = stars.rebin(wavl, fpfs1, b)

    return fpfs

def prior_rock(cube):
    params = np.zeros_like(cube)
    params[0] = quantile_to_uniform(cube[0], 0, 0.4) # albedo
    params[1] = truncnorm(-2, 2, loc=431, scale=23).ppf(cube[1]) # Teq
    params[2] = truncnorm(-2, 2, loc=3340, scale=150).ppf(cube[2]) # Teff
    params[3] = truncnorm(-2, 2, loc=0.0454, scale=0.0012).ppf(cube[3]) # R_planet_star
    return params   

def make_loglike(model, data_dict):
    def loglike(cube):
        data_bins = data_dict['bins']
        y = data_dict['fpfs']
        e = data_dict['err']
        resulty = model(cube, data_bins)
        if np.any(np.isnan(resulty)):
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

    # # F1500W eclipse centered on instant re-radiation
    # data_dict = make_F1500W_data(184.845e-6, 1)
    # cases['rock_F1500W_hot'] = make_loglike_prior(data_dict, param_names_rock, model_rock, model_rock_raw, prior_rock)
    # cases['atm_F1500W_hot'] = make_loglike_prior(data_dict, param_names_atm, model_atm, model_atm_raw, prior_atm)

    # # F1500W eclipse with half instant re-radiation
    # data_dict = make_F1500W_data(184.845e-6/2, 1)
    # cases['rock_F1500W_cool'] = make_loglike_prior(data_dict, param_names_rock, model_rock, model_rock_raw, prior_rock)
    # cases['atm_F1500W_cool'] = make_loglike_prior(data_dict, param_names_atm, model_atm, model_atm_raw, prior_atm)

    return cases

def compute_AIC(model_name):
    # Compute maximum likelihood, k, and AIC for one retrieval, then save the
    # result as `pymultinest/{model_name}/{model_name}_aic.h5`.
    if model_name not in RETRIEVAL_CASES:
        raise ValueError(f"Unknown model_name `{model_name}`.")

    case = RETRIEVAL_CASES[model_name]
    dirname = os.path.join("pymultinest", model_name)
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


WAVL, SPECTRA, PRESS, TEMP = make_interpolators('results/LTT1445Ab.h5')
WAVL_HOT, SPECTRA_HOT, PRESS_HOT, TEMP_HOT = make_interpolators('results/LTT1445Ab_hot.h5')
SPHINX = hotrocks.sphinx_interpolator('inputs/sphinx.h5')
RETRIEVAL_CASES = make_cases()

if __name__ == '__main__':

    models_to_run = list(RETRIEVAL_CASES.keys())
    for model_name in models_to_run:
        # Setup directories
        outputfiles_basename = f'pymultinest/{model_name}/{model_name}'
        try:
            os.mkdir(f'pymultinest/{model_name}')
        except FileExistsError:
            pass

        # Do nested sampling
        results = solve(
            LogLikelihood=RETRIEVAL_CASES[model_name]['loglike'], 
            Prior=RETRIEVAL_CASES[model_name]['prior'], 
            n_dims=len(RETRIEVAL_CASES[model_name]['param_names']), 
            outputfiles_basename=outputfiles_basename, 
            verbose=True,
            n_live_points=1000
        )
        # Save pickle
        pickle.dump(results, open(outputfiles_basename+'.pkl','wb'))

        # Compute AIC
        compute_AIC(model_name)
