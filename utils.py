import numpy as np
import re
from photochem.utils import stars
import scipy.stats as stats
from functools import lru_cache

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

def make_F1500W_data_real():
    """Observed eclipse depth; bin edges are plotting/legacy top-hat bounds.

    Use the photon response for inference, rather than these display bounds.
    """

    fpfs = 148.0e-6
    fpfs_err = 11.0e-6

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
    data_dict['photometric'] = np.array([True])

    return data_dict

def make_F1500W_data(fpfs, ntrans, err_one_transit=None):

    if err_one_transit is None:
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

def make_lrs_and_F1500W_data(filename, fpfs_15, ntrans, err_one_transit):

    data_dict1 = make_lrs_data(filename)
    data_dict2 = make_F1500W_data(fpfs_15, ntrans, err_one_transit)

    return append_data_dicts(data_dict1, data_dict2)

def make_lrs_and_F1500W_data_real(filename):

    data_dict1 = make_lrs_data(filename)
    data_dict2 = make_F1500W_data_real()

    return append_data_dicts(data_dict1, data_dict2)

def append_data_dicts(data_dict1, data_dict2):

    data_dict = {}
    data_dict['bins'] = np.vstack((data_dict1['bins'], data_dict2['bins']))
    for key in ['fpfs', 'err', 'wv', 'wv_err']:
        data_dict[key] = np.concatenate((data_dict1[key], data_dict2[key]))
    for key in ('err_lower', 'err_upper'):
        if key in data_dict1 or key in data_dict2:
            data_dict[key] = np.concatenate([d.get(key, d['err']) for d in (data_dict1, data_dict2)])
    if 'photometric' in data_dict1 or 'photometric' in data_dict2:
        data_dict['photometric'] = np.concatenate([
            d.get('photometric', np.zeros(len(d['wv']), dtype=bool))
            for d in (data_dict1, data_dict2)
        ])

    order = np.argsort(data_dict['wv'])
    for key in data_dict:
        data_dict[key] = data_dict[key][order]

    return data_dict


def make_joint_photometry(system):
    """Final system-fit measurement; asymmetric errors remain in flux-ratio units."""
    data = make_F1500W_data_real()  # Reuse display bounds, not the legacy depth.
    data['fpfs'] = np.array([system.eclipse_median])
    data['err_lower'] = np.array([system.eclipse_err_lower])
    data['err_upper'] = np.array([system.eclipse_err_upper])
    # Compatibility summary only; new likelihoods and plots use both sides.
    data['err'] = (data['err_lower']+data['err_upper'])/2
    return data


def observation_yerr(data, mask=None):
    """Matplotlib error array, retaining asymmetric measurement uncertainties."""
    errors = np.array([data.get('err_lower', data['err']), data.get('err_upper', data['err'])])
    return errors if mask is None else errors[:, mask]


def observation_loglike(data, prediction):
    """Independent normal/two-piece normal errors, with constant normalization omitted.

    Below the measured central depth use the lower uncertainty, above it use
    the upper uncertainty. The split-normal center is the chain median, a
    small-asymmetry approximation, not an exact reconstruction of its density.
    Constants cancel only between models fitted to the same observations.
    """
    prediction = np.asarray(prediction, dtype=float)
    if prediction.shape != np.asarray(data['fpfs']).shape:
        raise ValueError('Prediction shape does not match observations')
    if not np.isfinite(prediction).all():
        return -1.e100
    sigma = np.where(prediction < data['fpfs'], data.get('err_lower', data['err']),
                     data.get('err_upper', data['err']))
    if not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError('Measurement uncertainties must be finite and positive')
    return float(-.5*np.sum(((prediction-data['fpfs'])/sigma)**2))


@lru_cache(maxsize=4)
def _photon_weights(edges, wavelength, response):
    edges, wavelength, response = map(np.asarray, (edges, wavelength, response))
    if (edges.ndim != 1 or wavelength.ndim != 1 or response.shape != wavelength.shape
            or len(edges) < 2 or len(wavelength) < 2
            or not all(np.all(np.isfinite(a)) for a in (edges, wavelength, response))
            or np.any(np.diff(edges) <= 0) or np.any(np.diff(wavelength) <= 0)
            or np.any(wavelength <= 0) or np.any(response < 0) or not np.any(response > 0)):
        raise ValueError('Invalid wavelength grid or photon response')
    # Require the entire tabulated passband; do not silently discard filter wings.
    if wavelength[0] < edges[0] or wavelength[-1] > edges[-1]:
        raise ValueError('Response extends outside model wavelength coverage')
    weights = np.zeros(len(edges)-1)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        lo, hi = max(lo, wavelength[0]), min(hi, wavelength[-1])
        if lo >= hi:
            continue
        nodes = np.r_[lo, wavelength[(wavelength > lo) & (wavelength < hi)], hi]
        left, right = nodes[:-1], nodes[1:]
        mid = (left+right)/2
        # Simpson is exact for lambda times a piecewise-linear throughput.
        weights[i] = np.sum((right-left)/6 * (
            left*np.interp(left, wavelength, response)
            + 4*mid*np.interp(mid, wavelength, response)
            + right*np.interp(right, wavelength, response)))
    return weights


def photon_weighted_depth(edges, fpfs, stellar_flux, wavelength, response):
    """Photon-count eclipse ratio: integral(lambda R Fp)/integral(lambda R Fs).

    Fluxes are F_lambda bin averages on ``edges`` (microns); fpfs includes
    the squared radius ratio. Response is a photon throughput, not an
    energy-weighted response. Integrates the stored piecewise-constant spectra.
    """
    weights = _photon_weights(tuple(edges), tuple(wavelength), tuple(response))
    fpfs, stellar_flux = np.asarray(fpfs), np.asarray(stellar_flux)
    if fpfs.shape != weights.shape or stellar_flux.shape != weights.shape:
        raise ValueError('Flux arrays must have one entry per model bin')
    denominator = np.sum(weights*stellar_flux)
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError('Nonpositive stellar photon flux')
    return np.sum(weights*stellar_flux*fpfs)/denominator

def species_to_latex(sp):
    sp1 = re.sub(r'([0-9]+)', r"_\1", sp)
    sp1 = r'$\mathrm{'+sp1+'}$'
    if sp == 'O1D':
        sp1 = r'$\mathrm{O(^1D)}$'
    elif sp == 'N2D':
        sp1 = r'$\mathrm{N(^2D)}$'
    elif sp == '1CH2':
        sp1 = r'$\mathrm{^1CH_2}$'
    elif sp == 'H2SO4aer':
        sp1 = r'H$_2$SO$_2$ cloud'
    return sp1

def residuals(data_y, err, expected_y):
    return (data_y - expected_y)/err

def chi_squared(data_y, err, expected_y):
    R = residuals(data_y, err, expected_y)
    return np.sum(R**2)

def reduced_chi_squared(data_y, err, expected_y, dof):
    chi2 = chi_squared(data_y, err, expected_y)
    return chi2/dof

def compute_sigma(data_y, err, expected_y, dof):
    chi2_value = chi_squared(data_y, err, expected_y)
    p_value = stats.chi2.sf(chi2_value, dof)
    sigma_value = stats.norm.ppf(1 - p_value/2)
    return sigma_value

def rebin_spectra_to_data(wavl, fpfs1, data_dict):
    wv_bins = data_dict['bins']
    fpfs = np.empty(len(wv_bins))
    for i,b in enumerate(wv_bins):
        fpfs[i] = stars.rebin(wavl.copy(), fpfs1.copy(), b.copy())[0]
    return fpfs

def compute_stats(wavl, fpfs, data_dict):
    fpfs1 = rebin_spectra_to_data(wavl, fpfs, data_dict)
    args = (data_dict['fpfs'], data_dict['err'], fpfs1, len(fpfs1))
    rchi2 = reduced_chi_squared(*args)
    sigma = compute_sigma(*args)
    return rchi2, sigma
