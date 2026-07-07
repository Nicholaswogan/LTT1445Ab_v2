import input_files

import numpy as np
import retrieval_run
import pickle
from matplotlib import pyplot as plt
from ultranest.plot import PredictionBand
import LTT1445Ab_grid

def latex_interval(samples, precision=2, mode="median"):
    samples = np.asarray(samples)
    
    fmt = f"{{:.{precision}f}}"
    
    if mode == "median":
        median = np.percentile(samples, 50)
        lower = np.percentile(samples, 16)
        upper = np.percentile(samples, 84)
        
        err_plus = upper - median
        err_minus = median - lower
        
        return f"${fmt.format(median)}^{{+{fmt.format(err_plus)}}}_{{-{fmt.format(err_minus)}}}$"
    
    elif mode == "upper":
        upper95 = np.percentile(samples, 97.5)
        return f"$< {fmt.format(upper95)}$"
    elif mode== 'lower':
        lower = np.percentile(samples, 2.5)
        return f"$> {fmt.format(lower)}$"
    
    else:
        raise ValueError("mode must be 'median' or 'upper'")

def spectra_16():

    results = RESULTS

    np.random.seed(2)
    n_draws = 500
    wavl = LTT1445Ab_grid.WAVL
    wv = (wavl[1:] + wavl[:-1])/2

    case = retrieval_run.RETRIEVAL_CASES['atm_16']
    result = results['atm_16']

    data_dict = case['data_dict']

    samples = result['samples']
    draws = np.random.randint(0, samples.shape[0]-1, n_draws)
    band_atm = PredictionBand(wv)
    for idraw in draws:
        band_atm.add(1e6*case['model_raw'](samples[idraw,:], wavl))

    case = retrieval_run.RETRIEVAL_CASES['rock_16']
    result = results['rock_16']

    samples = result['samples']
    draws = np.random.randint(0, samples.shape[0]-1, n_draws)
    band_rock = PredictionBand(wv)
    for idraw in draws:
        band_rock.add(1e6*case['model_raw'](samples[idraw,:], wavl))

    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(1,1,figsize=[10,3.0])
    fig.patch.set_facecolor("w")

    wavl = LTT1445Ab_grid.WAVL
    wv = (wavl[1:] + wavl[:-1])/2
    c = LTT1445Ab_grid.NOMINAL_CLIMATE_MODEL
    _,_,fpfs = c.fpfs_instant_reradiation(wavl, 0.0)
    ax.plot(wv, fpfs*1e6,lw=1,c='k',ls='-',label='Dark bare rock')

    band_rock.line(ax=ax,color='0.5',lw=1, label='Bare-rock model\n'+r'($\log Z = '+f'{results['rock_16']['logZ']:.1f}$)')
    band_rock.shade(q=0.341,ax=ax, color='0.5', alpha=0.3, label='')

    band_atm.line(ax=ax,color='C0',lw=1, label='Atmospheric model\n'+r'($\log Z = '+f'{results['atm_16']['logZ']:.1f}$)')
    band_atm.shade(q=0.341,ax=ax, color='C0', alpha=0.3, label='')

    ax.errorbar(data_dict['wv'][:], data_dict['fpfs'][:]*1e6, yerr=data_dict['err'][:]*1e6, xerr=data_dict['wv_err'][:], 
                elinewidth=.5, marker='o', capsize=1.5, ms=1.5, ls='', c='k')
    ax.errorbar([-1], [-1], yerr=[1], xerr=[1], elinewidth=1, marker='o', ms=2, capsize=1.5, ls='', c='k',
                label="JWST MIRI LRS")

    ax.grid(alpha=0.4)
    ax.set_xlim(4,18)
    ax.grid(alpha=0.4)
    ax.set_xlim(4.8,17)
    ax.set_ylim(0,210)
    ax.set_yticks([0,50,100,150,200])
    ax.set_xticks(np.arange(5,18,1))
    ax.set_xlabel('Wavelength (microns)')
    ax.set_ylabel('Planet-to-star flux (ppm)')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], ncol=1,bbox_to_anchor=(0.0, 1), loc='upper left',fontsize=10, frameon=False)

    plt.savefig('figures/spectra_16.pdf',bbox_inches='tight')

def retrieval_16():

    results = RESULTS

    # Spectrum
    key = 'atm_16'
    case = retrieval_run.RETRIEVAL_CASES[key]
    result = results[key]

    np.random.seed(2)
    n_draws = 500
    samples = result['samples']
    draws = np.random.randint(0, samples.shape[0]-1, n_draws)
    wavl = LTT1445Ab_grid.WAVL

    Ps = []
    Ts = []
    wv = (wavl[1:] + wavl[:-1])/2
    band = PredictionBand(wv)
    for idraw in draws:
        band.add(1e6*case['model_raw'](samples[idraw,:], wavl))
        Ps.append(retrieval_run.PRESS(samples[idraw,:7]))
        Ts.append(retrieval_run.TEMP(samples[idraw,:7]))

    n_params = 4

    wv = (wavl[1:] + wavl[:-1])/2
    band1 = PredictionBand(wv)
    cube = np.random.uniform(size=n_draws*n_params).reshape((n_draws,n_params))
    for i in range(cube.shape[0]):
        x = retrieval_run.prior_rock(cube[i,:])
        x[0] = 0
        fpfs = retrieval_run.model_rock_raw(x, wavl)
        band1.add(1e6*fpfs)

    n_draws = 5000
    n_params = len(case['param_names'])
    cube = np.random.uniform(size=n_draws*n_params).reshape((n_draws,n_params))
    samples_prior = np.empty_like(cube)
    for i in range(cube.shape[0]):
        x = case['prior'](cube[i,:])
        samples_prior[i,:] = x

    param_names = case['param_names']

    plt.rcParams.update({'font.size': 11})
    fig = plt.figure(constrained_layout=False,figsize=[12,8])
    fig.patch.set_facecolor("w")

    gs = fig.add_gridspec(100, 100)

    sep = 8
    w = int((100-(1*sep))/2)
    axs1 = []
    start = 0
    end = w
    for i in range(2):
        ax = fig.add_subplot(gs[:27, start:end])
        axs1.append(ax)
        start = end+sep
        end = end+sep+w

    sep = 2
    w = int((100-(3*sep))/4)
    axs2 = []

    start = 0
    end = w
    for i in range(4):
        ax = fig.add_subplot(gs[27+10:(27+10)+27, start:end])
        axs2.append(ax)
        start = end+sep
        end = end+sep+w

    sep = 2
    w = int((100-(4*sep))/5)
    axs3 = []

    start = 0
    end = w
    for i in range(5):
        ax = fig.add_subplot(gs[(27+10)+27+10:, start:end])
        axs3.append(ax)
        start = end+sep
        end = end+sep+w


    # 
    # Surface pressure
    # 
    ax = axs1[0]
    P_surf = np.sum(10.0**samples[:,:4],axis=1)
    P_surf_prior = np.sum(10.0**samples_prior[:,:4],axis=1)
    bins = np.arange(-4.5,3.5,.25)
    ax.hist(np.log10(P_surf), bins=bins, alpha=0.3, facecolor='k', density=True)
    ax.hist(np.log10(P_surf), bins=bins, alpha=1, color='k', lw=2, density=True, histtype='step', label='Posterior')
    ax.hist(np.log10(P_surf_prior), bins=bins, alpha=1, color='k', lw=1, density=True, ls=':', histtype='step', label='Prior')
    # ax.axvline(np.quantile(np.log10(P_surf),.95), c='k', lw=2)
    ax.set_xlim(-3,2.5)
    ax.set_xticks(np.arange(-3,3,1))
    ax.set_xlabel(r'$\log_{10}$ Tot. Surf. Pressure (bar)')
    ax.grid(alpha=0.4)
    ax.set_yticks([])
    ax.set_ylabel('Probability Density',labelpad=10)
    upper_limit = np.quantile(np.log10(P_surf),.975)
    ax.axvline(upper_limit, c='k', lw=1)
    ax.text(upper_limit,ax.get_ylim()[1]*.5,'95% UL',rotation=-90,size=8)
    note = latex_interval(np.log10(P_surf), precision=1, mode='upper')
    ax.text(.4,.01,note,size=13,c='k', transform=ax.transAxes, ha='center', va='bottom')
    ax.legend(ncol=2, columnspacing=0.9, bbox_to_anchor=(-0.0, 1.0), loc='upper left',fontsize=9.5, frameon=True)
    ax.set_ylim(0,ax.get_ylim()[1]*1.1)

    # P T profile
    ax = axs1[1]
    for i in range(150):
        ax.plot(Ts[i][0], np.log10(Ps[i][0]/1e6),c='k',ls='',marker='o',ms=1,lw=0.1, alpha=0.3)
        ax.plot(Ts[i], np.log10(Ps[i]/1e6),c='k',lw=0.1, alpha=0.3)
    ax.plot([],[],c='k',lw=0.5, alpha=0.7,label='Estimate')
    ax.grid(alpha=0.4)
    ax.set_ylabel(r'Pressure ($\log_{10}$ bar)', labelpad=3)
    # ax.set_yscale('log')
    ax.set_ylim(np.log10(10),np.log10(2e-6))
    ax.set_yticks([1,0,-1,-2,-3,-4,-5])
    ax.set_xlabel('Dayside Temperature (K)')

    props = {
        'keys': ['log10PH2O','log10PCO2','log10PO2','log10PSO2'],
        'labels': [
            r'$\log_{10}P_\mathrm{H_2O}$ (bar)',
            r'$\log_{10}P_\mathrm{CO_2}$ (bar)',
            r'$\log_{10}P_\mathrm{O_2}$ (bar)',
            r'$\log_{10}P_\mathrm{SO_2}$ (bar)',
        ],
        'labels1': [
            'H$_2$O',
            'CO$_2$',
            'O$_2$',
            'SO$_2$',
        ],
        'labels1x': [
            'right',
            'left',
            'right',
            'right',
        ],
        'colors': ['C0','C2','C4','C5'],
        'upper_limits': [True,True,True,True],
        'upper_limits_xlabel': [.3,.5,.3,.3],
        'xticks': [
            np.arange(-7,3,2),
            np.arange(-7,3,2),
            np.arange(-4,3,2),
            np.arange(-7,-1,2),
        ],
    }
    for i,ax in enumerate(axs2):
        
        Pi = samples[:,param_names.index(props['keys'][i])]
        color = props['colors'][i]
        ax.hist(Pi, alpha=0.3, facecolor=color, density=True)
        ax.hist(Pi, alpha=1, color=color, lw=2, density=True, histtype='step')
        ax.set_xlim(np.min(Pi),np.max(Pi))
        ax.set_xlabel(props['labels'][i])
        if props['xticks'][i] is not None:
            ax.set_xticks(props['xticks'][i])
        if props['labels1'][i] is not None:
            if props['labels1x'][i] == 'right':
                ha = 'right'
                x = .98
            else:
                ha = 'left'
                x = 0.02
            ax.text(x,.98,props['labels1'][i],size=13,c=color, transform=ax.transAxes, ha=ha, va='top')
        if props['upper_limits'][i]:
            upper_limit = np.quantile(Pi,.975)
            ax.axvline(upper_limit, c='k', lw=1)
            ax.text(upper_limit,ax.get_ylim()[1]*.65,'95% UL',rotation=-90,size=8)
            note = latex_interval(Pi, precision=1, mode='upper')
            ax.text(props['upper_limits_xlabel'][i],.01,note,size=13,c=color, transform=ax.transAxes, ha='center', va='bottom')

    for ax in axs2:
        ax.grid(alpha=0.4)
        ax.set_yticks([])
    ax = axs2[0]
    ax.set_ylabel('Probability Density',labelpad=10)


    props = {
        'keys': ['log10chi','albedo','Teq','Teff','R_planet_star'],
        'colors': ['C6','C7','C8','C9','C1'],
        'labels': [r'$\log_{10}\chi$','Surface albedo','$T_{eq}$ (K)','$T_{*,eff}$ (K)','$R_p/R_*$'],
        'xticks': [
            None,
            None,
            None,
            None,
            None
        ],
        'precisions': [None, None, None, None, None]
    }
    for i,ax in enumerate(axs3):
        Pi = samples[:,param_names.index(props['keys'][i])]
        color = props['colors'][i]
        ax.hist(Pi, alpha=0.3, facecolor=color, density=True)
        ax.hist(Pi, alpha=1, color=color, lw=2, density=True, histtype='step')
        ax.set_xlim(np.min(Pi),np.max(Pi))
        ax.set_xlabel(props['labels'][i])
        if props['xticks'][i] is not None:
            ax.set_xticks(props['xticks'][i])
        if props['precisions'][i] is not None:
            note = latex_interval(Pi, precision=props['precisions'][i])
            ax.text(.5,.01,note,size=13,c=color, transform=ax.transAxes, ha='center', va='bottom')


    for ax in axs3:
        # ax.set_xlim(-4,2)
        ax.grid(alpha=0.4)
        ax.set_yticks([])

    ax = axs3[0]
    ax.set_ylabel('Probability Density',labelpad=10)


    plt.savefig('figures/retrieval_16.pdf',bbox_inches='tight')


def get_results():
    results = {}
    for key in ['rock_16', 'atm_16']:
        with open('pymultinest/'+key+'/'+key+'.pkl','rb') as f:
            result = pickle.load(f)
        results[key] = result
    return results

RESULTS = get_results()

if __name__ == "__main__":
    spectra_16()
    retrieval_16()
    