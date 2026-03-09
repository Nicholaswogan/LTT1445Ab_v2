import numpy as np
import re

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