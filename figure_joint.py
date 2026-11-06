"""Plot the three atmospheric joint retrievals, spectra and pressure posteriors.

Run from the repository root in conda env ltt:
    python figure_joint.py
    python figure_joint.py --lrs-bins 8
The first run evaluates all posterior spectra; subsequent runs reuse a checked
cache so that layout edits are quick. No retrieval outputs are modified.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import qmc

import retrieval_run as legacy
import planets
from retrieval_run_joint import JointRetrieval
from joint_system_prior import ROOT, RESULTS_DIR, sha256, require_current_fit
from utils import observation_yerr

# Figures overwrite their established paths, independently of the fit tag.
JOINT_FIGURE_DIR = ROOT / 'figures/joint'

def scenario_cases(lrs_bins=16):
    return ((f'atm_{lrs_bins}', 'LRS alone', 'C0'),
            ('atm_F1500W_obs', 'F1500W alone', 'C1'),
            (f'atm_{lrs_bins}_F1500W_obs', 'LRS & F1500W combined', 'C2'))


CASES = scenario_cases()
PARAMETERS = ['log10PH2O', 'log10PCO2', 'log10PO2', 'log10PSO2', 'log10chi',
              'albedo', 'Teq', 'Teff', 'R_planet_star']
SPECIES = [('H_2O', 'log10PH2O', -7.), ('CO_2', 'log10PCO2', -7.),
           ('O_2', 'log10PO2', -5.), ('SO_2', 'log10PSO2', -7.)]


def load_posteriors(directory, cases=CASES):
    result = {}
    for name, _, _ in cases:
        folder = directory/name
        metadata_file = folder/'inputs.json'
        posterior_file = folder/(name+'post_equal_weights.dat')
        metadata = json.loads(metadata_file.read_text())
        require_current_fit(metadata)
        data = np.atleast_2d(np.loadtxt(posterior_file))
        if data.shape[1] != len(metadata['param_names'])+1 or not np.all(np.isfinite(data)):
            raise ValueError(f'Invalid equal-weight posterior: {posterior_file}')
        order = [metadata['param_names'].index(k) for k in PARAMETERS]
        result[name] = {'samples': data[:, order], 'metadata': metadata,
                        'posterior_file': posterior_file, 'metadata_file': metadata_file, 'loglike': data[:, -1]}
    return result


def spectral_bands(run, posteriors, cache, max_spectra, seed, refresh, cases=CASES):
    fingerprint = {
        'posterior_sha256': {k: sha256(v['posterior_file']) for k, v in posteriors.items()},
        'inputs_sha256': {k: sha256(v['metadata_file']) for k, v in posteriors.items()},
        'model_source_sha256': {name: sha256(ROOT/name) for name in
                                ['retrieval_run.py', 'retrieval_run_joint.py', 'planets.py', 'gridutils.py']},
        'grid_stat': [run.grid_file.stat().st_size, run.grid_file.stat().st_mtime_ns],
        'stellar_sha256': sha256(run.stellar_file),
        'max_spectra': max_spectra, 'seed': seed,
        'quantiles': [.15865525393145707, .5, .8413447460685429],
    }
    key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    if cache.exists() and not refresh:
        with np.load(cache, allow_pickle=False) as saved:
            if saved['fingerprint'].item() == key:
                print(f'Using spectral cache: {cache}', flush=True)
                return {name: saved[name].copy() for name, _, _ in cases}, fingerprint
    bands = {}
    rng = np.random.default_rng(seed)
    for name, _, _ in cases:
        samples = posteriors[name]['samples']
        if max_spectra and len(samples) > max_spectra:
            samples = samples[rng.choice(len(samples), max_spectra, replace=False)]
        print(f'Evaluating {len(samples)} posterior spectra for {name}...', flush=True)
        spectra = np.array([legacy.model_atm_raw(x, run.wavelength) for x in samples])
        if not np.all(np.isfinite(spectra)):
            raise ValueError(f'Nonfinite posterior spectrum in {name}')
        bands[name] = np.quantile(spectra, fingerprint['quantiles'], axis=0)*1.e6
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, fingerprint=np.array(key), **bands)
    return bands, fingerprint


def posterior_panel(ax, values, bins, xlabel, label=None, ylabel=False, cases=CASES):
    for (_, _, color), series in zip(cases, values):
        ax.hist(series, bins=bins, density=True, histtype='step', color=color, lw=2)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(bottom=0)
    ax.set_xlabel(xlabel)
    ax.set_yticks([])
    if label:
        ax.text(.97, .97, label, transform=ax.transAxes, ha='right', va='top')
    if ylabel:
        ax.set_ylabel('Probability Density')


def make_figure(run, posteriors, bands, cases=CASES, lrs_bins=16):
    plt.rcdefaults()
    plt.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(12., 6.4))
    fig.patch.set_facecolor('w')
    grid = fig.add_gridspec(2, 1, height_ratios=(1.35, 1.), hspace=.28)
    top = grid[0].subgridspec(1, 2, width_ratios=(3., 1.15), wspace=.10)
    bottom = grid[1].subgridspec(1, 4, wspace=.08)
    spectrum_ax = fig.add_subplot(top[0])
    total_ax = fig.add_subplot(top[1])
    gas_axes = [fig.add_subplot(bottom[i]) for i in range(4)]
    fig.subplots_adjust(left=.075, right=.985, bottom=.12, top=.97)
    wavelength = .5*(run.wavelength[:-1]+run.wavelength[1:])

    # The effective weighting of a monochromatic eclipse ratio is the stellar
    # photon-count density: lambda * throughput * stellar F_lambda. Display its
    # normalized shape in axes-height units, without changing the flux scale.
    system_center = run.system.transform(np.full(3, .5))
    star = planets.LTT1445A
    stellar_wavelength, stellar_flux = legacy.SPHINX(
        system_center[1], star.metal, star.logg, rescale_to_Teff=False)
    response_wavelength, response = run.response
    if (response_wavelength.min() < stellar_wavelength.min()
            or response_wavelength.max() > stellar_wavelength.max()):
        raise ValueError('Stellar spectrum does not cover the full F1500W response')
    response = response_wavelength*response*np.interp(
        response_wavelength, stellar_wavelength, stellar_flux)
    response = response/response.max()
    spectrum_ax.fill_between(response_wavelength, 0, .105*response,
                             transform=spectrum_ax.get_xaxis_transform(),
                             color='0.45', alpha=.20, linewidth=0, zorder=0)
    spectrum_ax.text(15.1, .12, 'F1500W', fontsize=11, color='0.4',
                     ha='center', transform=spectrum_ax.get_xaxis_transform())
    for name, label, color in cases:
        lo, median, hi = bands[name]
        spectrum_ax.fill_between(wavelength, lo, hi, color=color, alpha=.18, linewidth=0, zorder=1)
        spectrum_ax.plot(wavelength, median, color=color, lw=2, label=label, zorder=3)

    # A fixed, zero-albedo reference at the central retained system constraints;
    # this is not a fitted bare-rock posterior or an evidence-weighted average.
    rock_parameters = np.r_[0., system_center]
    bare = legacy.model_rock_raw(rock_parameters, run.wavelength)*1.e6
    flux = legacy.stars.equilibrium_temperature_inverse(system_center[0], 0.)
    blackbody_temperature = float(legacy.hotrocks.bare_rock_dayside_temperature(flux, 0., 2/3))
    spectrum_ax.plot(wavelength, bare, color='k', lw=1.5, ls='-',
                     label=f'Zero albedo blackbody ({blackbody_temperature:.0f} K)', zorder=2)

    lrs = run.cases[f'atm_{lrs_bins}']['data_dict']
    f1500 = run.cases['atm_F1500W_obs']['data_dict']
    spectrum_ax.errorbar(lrs['wv'], lrs['fpfs']*1.e6, yerr=lrs['err']*1.e6,
                         xerr=lrs['wv_err'], fmt='o', color='0.1', ms=3.1,
                         mew=.7, elinewidth=.8, capsize=1.6, label='MIRI LRS (Wachiraphan+25)', zorder=5)
    spectrum_ax.errorbar(f1500['wv'], f1500['fpfs']*1.e6, yerr=observation_yerr(f1500)*1.e6,
                         xerr=f1500['wv_err'], fmt='D', color='0.1', mfc='white', ms=4.8,
                         mew=.9, elinewidth=1., capsize=2., label='MIRI F1500W (this work)', zorder=6)
    spectrum_ax.set(xlim=(4.8, 17.5), ylim=(0., 240.),
                    xlabel=r'Wavelength ($\mu$m)', ylabel='Planet-to-star flux (ppm)')
    spectrum_ax.set_xticks(np.arange(5., 18., 2.))
    spectrum_ax.set_yticks([0, 50, 100, 150, 200])
    spectrum_ax.legend(loc='upper left', ncol=2, frameon=False, fontsize=12,
                       columnspacing=1.5, handlelength=2.4, bbox_to_anchor=(0.0, 1.03))

    # All pressure densities are per dex, evaluated from every equal-weight
    # posterior draw (even when --max-spectra limits only spectral evaluation).
    samples = [posteriors[name]['samples'] for name, _, _ in cases]
    total_pressure = [np.log10(np.sum(10.**s[:, :4], axis=1)) for s in samples]
    posterior_panel(total_ax, total_pressure, np.arange(-5., 2.751, .25),
                    r'$\log_{10}(P_{\rm surf}/\mathrm{bar})$',
                    ylabel=True, cases=cases)
    prior = run.cases[f'atm_{lrs_bins}_F1500W_obs']['prior']
    low, high = prior(np.zeros(9))[:4], prior(np.ones(9))[:4]
    unit = qmc.Sobol(4, scramble=True, seed=1445).random_base2(20)
    total_prior = np.log10(np.sum(10.**(low+unit*(high-low)), axis=1))
    total_ax.hist(total_prior, bins=np.arange(-5., 2.751, .25), density=True,
                  histtype='step', color='k', ls='--', lw=1.5)
    total_ax.text(.06, .97, 'Total\nsurface\npressure', transform=total_ax.transAxes,
                  ha='left', va='top', multialignment='center',
                  bbox=dict(facecolor='white', edgecolor='none', pad=1.5), fontsize=12)
    total_upper_limits = {}
    for (name, _, color), pressure in zip(cases, total_pressure):
        upper_limit = float(np.quantile(10.**pressure, .975))
        total_upper_limits[name] = upper_limit
        total_ax.axvline(np.log10(upper_limit), color=color, ls=':', lw=1.5, alpha=0.8)
    total_ax.legend(handles=[Line2D([], [], color='k', ls=':', lw=2,
                                    label='97.5%\nupper lim.'),
                            Line2D([], [], color='k', ls='--', lw=1.5, label='Prior')],
                    loc='upper right', bbox_to_anchor=(1.035, 1.02),
                    frameon=True, facecolor='none', edgecolor='none', framealpha=1.,
                    fontsize=10.5, handlelength=2.)
    total_ax.set_xlim(left=-3, right=3)
    if lrs_bins == 16:
        total_ax.set_ylim(total_ax.get_ylim()[0], total_ax.get_ylim()[1]*1.1)
    total_ax.set_xticks([-3, -2, -1, 0, 1, 2])
    partial_upper_limits = {}
    for i, (ax, (formula, parameter, lower)) in enumerate(zip(gas_axes, SPECIES)):
        column = PARAMETERS.index(parameter)
        posterior_panel(ax, [s[:, column] for s in samples], np.arange(lower, 2.001, .25),
                        rf'$\log_{{10}}(P_{{\mathrm{{{formula}}}}}/\mathrm{{bar}})$',
                        label=rf'$\mathrm{{{formula}}}$', ylabel=i == 0, cases=cases)
        partial_upper_limits[parameter] = {}
        for (name, _, color), sample in zip(cases, samples):
            upper_limit = float(np.quantile(10.**sample[:, column], .975))
            partial_upper_limits[parameter][name] = upper_limit
            ax.axvline(np.log10(upper_limit), color=color, ls=':', lw=1.5, alpha=.8)
        ax.set_xticks(np.arange(lower, 2.01, 2.))
    return fig, {'bare_rock_reference_parameters': dict(zip(
        ['albedo', 'Teq', 'Teff', 'R_planet_star'], rock_parameters.tolist())),
        'bare_rock_temperature_K': blackbody_temperature,
        'spectral_interval': 'pointwise central 68.268949% posterior interval',
        'posterior_density': 'unfilled step histograms per log10 pressure (per dex), bin width 0.25 dex',
        'F1500W_horizontal_bar': 'legacy display bounds 13.6--16.5 um; likelihood uses full photon response',
        'F1500W_display_response': 'lambda * R(lambda) * Fstar_lambda, normalized to its peak, using the reference stellar parameters',
        'posterior_annotations': 'matching-color dotted 97.5th-percentile total and partial pressure upper limits; dashed black induced total-pressure prior',
        'total_pressure_prior': {'samples': len(total_prior), 'method': 'Scrambled Sobol, seed 1445; sum of four independent log-uniform partial pressures',
                                 'log10_partial_pressure_bounds': np.column_stack((low, high)).tolist()},
        'total_pressure_upper_limit_quantile': .975,
        'total_pressure_upper_limits_bar': total_upper_limits,
        'partial_pressure_upper_limit_quantile': .975,
        'partial_pressure_upper_limits_bar': partial_upper_limits}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=RESULTS_DIR)
    parser.add_argument('--lrs-bins', type=int, choices=(8, 16), default=16)
    parser.add_argument('--output-prefix', type=Path, help='Default: figures/joint/figure_joint[_8]')
    parser.add_argument('--max-spectra', type=int, default=0,
                        help='Optional spectral subsample size; default 0 uses every posterior draw')
    parser.add_argument('--seed', type=int, default=1445)
    parser.add_argument('--refresh-cache', action='store_true')
    args = parser.parse_args()
    if args.max_spectra < 0:
        parser.error('--max-spectra must be nonnegative')
    cases = scenario_cases(args.lrs_bins)
    if args.output_prefix is None:
        suffix = '_8' if args.lrs_bins == 8 else ''
        args.output_prefix = JOINT_FIGURE_DIR / f'figure_joint{suffix}'
    posteriors = load_posteriors(args.results, cases)
    combined_metadata = posteriors[f'atm_{args.lrs_bins}_F1500W_obs']['metadata']
    run = JointRetrieval(chain=Path(combined_metadata['system_prior']['source_csv']).parent,
                         response=combined_metadata['response_file'], lrs_bins=args.lrs_bins)
    # Fail clearly if the plotting model no longer matches the saved fit.
    for name, entry in posteriors.items():
        for filename, expected in entry['metadata']['source_sha256'].items():
            if sha256(ROOT/filename) != expected:
                raise ValueError(f'Model source changed since {name}: {filename}')
        if sha256(run.response_file) != entry['metadata']['response_sha256']:
            raise ValueError(f'Response changed since {name}')
        if sha256(run.system.source) != entry['metadata']['system_prior']['source_sha256']:
            raise ValueError(f'System chain changed since {name}')
        if name != 'atm_F1500W_obs' and sha256(run.lrs_file) != entry['metadata']['lrs_sha256']:
            raise ValueError(f'LRS data changed since {name}')
        if sha256(run.stellar_file) != entry['metadata']['stellar_spectrum']['sha256']:
            raise ValueError(f'Stellar spectra changed since {name}')
        case = run.cases[name]
        for key, values in case['data_dict'].items():
            if not np.array_equal(values, np.asarray(entry['metadata']['data'][key])):
                raise ValueError(f'Observation changed since {name}: {key}')
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    cache = args.output_prefix.parent/(args.output_prefix.name+'_spectra.npz')
    bands, fingerprint = spectral_bands(run, posteriors, cache, args.max_spectra,
                                       args.seed, args.refresh_cache, cases)
    fig, description = make_figure(run, posteriors, bands, cases=cases, lrs_bins=args.lrs_bins)
    for extension in ('pdf', 'png'):
        path = args.output_prefix.with_suffix('.'+extension)
        fig.savefig(path, dpi=220, bbox_inches='tight', pad_inches=.06)
        print(f'Saved {path}', flush=True)
    plt.close(fig)
    metadata = dict(description, results_directory=str(args.results), lrs_bins=args.lrs_bins, fingerprint=fingerprint,
                    cases=[dict(name=n, label=l, color=c, posterior_draws=len(posteriors[n]['samples']))
                           for n, l, c in cases])
    args.output_prefix.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')


if __name__ == '__main__':
    main()
