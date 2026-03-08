import input_files

import numpy as np
from photochem.extensions import hotrocks
from photochem.utils import stars
import planets
import LTT1445Ab_grid
from scipy.stats import truncnorm
import os 
import pickle
from pymultinest.solve import solve
import gridutils

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

def make_lrs_data(filename):
    wv1, wv2, wv, fpfs, fpfs_err = np.loadtxt(filename,skiprows=1).T

    wv = (wv1 + wv2)/2
    wv_err = (wv2 - wv1)/2
    bins = np.empty((len(wv),2))
    for i in range(wv.shape[0]):
        bins[i,:] = np.array([wv1[i],wv2[i]])

    assert np.all(np.isclose(np.mean(bins,axis=1), wv))
    assert np.all(np.isclose((bins[:,1] - bins[:,0])/2, wv_err))
    
    data_dict = {}
    data_dict['bins'] = bins
    data_dict['fpfs'] = fpfs/1e6
    data_dict['err'] = fpfs_err/1e6
    data_dict['wv'] = wv
    data_dict['wv_err'] = wv_err

    return data_dict

def make_F1500W_data(fpfs, ntrans, err_one_transit=None):

    if err_one_transit == None:
        err_one_transit = 36e-6 # from proposal
    fpfs_err = err_one_transit/np.sqrt(ntrans)

    bins = np.empty((1,2))
    bins[0,:] = np.array([13.6, 16.5])
    wv = np.array([np.mean([13.6, 16.5])])
    wv_err = np.array([(16.5 - 13.6)/2])

    data_dict = {}
    data_dict['bins'] = bins
    data_dict['fpfs'] = np.array([fpfs])
    data_dict['err'] = np.array([fpfs_err])
    data_dict['wv'] = wv
    data_dict['wv_err'] = wv_err

    return data_dict

def make_cases():

    cases = {}

    param_names_atm = [
        'log10PH2O', 'log10PCO2', 'log10PO2', 'log10PSO2', 'log10chi', 
        'albedo', 'Teq', 'Teff', 'R_planet_star'
    ]
    param_names_rock = [
        'albedo', 'Teq', 'Teff', 'R_planet_star'
    ]

    # 8 bin data
    data_dict = make_lrs_data('data/LTT1445Ab_Sparta_8.txt')
    cases['rock_8'] = make_loglike_prior(data_dict, param_names_rock, model_rock, model_rock_raw, prior_rock)
    cases['atm_8'] = make_loglike_prior(data_dict, param_names_atm, model_atm, model_atm_raw, prior_atm)

    # 16 bin data
    data_dict = make_lrs_data('data/LTT1445Ab_Sparta_16.txt')
    cases['rock_16'] = make_loglike_prior(data_dict, param_names_rock, model_rock, model_rock_raw, prior_rock)
    cases['atm_16'] = make_loglike_prior(data_dict, param_names_atm, model_atm, model_atm_raw, prior_atm)

    # # F1500W eclipse centered on instant re-radiation
    # data_dict = make_F1500W_data(184.845e-6, 1)
    # cases['rock_F1500W_hot'] = make_loglike_prior(data_dict, param_names_rock, model_rock, model_rock_raw, prior_rock)
    # cases['atm_F1500W_hot'] = make_loglike_prior(data_dict, param_names_atm, model_atm, model_atm_raw, prior_atm)

    # # F1500W eclipse with half instant re-radiation
    # data_dict = make_F1500W_data(184.845e-6/2, 1)
    # cases['rock_F1500W_cool'] = make_loglike_prior(data_dict, param_names_rock, model_rock, model_rock_raw, prior_rock)
    # cases['atm_F1500W_cool'] = make_loglike_prior(data_dict, param_names_atm, model_atm, model_atm_raw, prior_atm)

    return cases

WAVL, SPECTRA, PRESS, TEMP = make_interpolators('results/LTT1445Ab.h5')
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