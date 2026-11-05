"""Simple presentation figures; run in conda environment ltt."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from joint_system_prior import (ROOT, FIT_TAG, RESULTS_DIR,
                                JointSystemPrior, require_current_fit)
from utils import make_joint_photometry, observation_yerr, observation_loglike

JOINT_FIGURE_DIR = ROOT / 'figures/joint'
PRESENT_FIGURE_DIR = ROOT / 'figures/present'


def _final_photometry():
    return make_joint_photometry(JointSystemPrior.from_chain())


F1500W_COLOR = 'red'

def figure_data():
    plt.rcParams.update({'font.size': 13})
    fig, ax = plt.subplots(1, 1, figsize=[6, 4])

    lower, upper, _, depth, error = np.loadtxt(
        ROOT/'data/LTT1445Ab_Sparta_16.txt', skiprows=1).T
    ax.errorbar((lower+upper)/2, depth, xerr=(upper-lower)/2, yerr=error,
                fmt='o', color='black', markersize=3, elinewidth=1, capsize=2,
                label='MIRI LRS (Wachiraphan+25)')

    # Use the same updated F1500W measurement as the joint retrieval.
    photometry = _final_photometry()
    ax.errorbar(photometry['wv'], photometry['fpfs']*1.e6,
                xerr=photometry['wv_err'], yerr=observation_yerr(photometry)*1.e6,
                fmt='s', color=F1500W_COLOR, markersize=6, elinewidth=1, capsize=2,
                label='MIRI F1500W (Ih+2026; DDT)')

    ax.set_xlabel(r'Wavelength ($\mu$m)')
    ax.set_ylabel('Planet-to-star flux (ppm)')
    ax.set_ylim(0, 200)
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1), frameon=False, fontsize=13)

    fig.tight_layout()
    output = PRESENT_FIGURE_DIR/'figure_data.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    print(f'Saved {output}')
    return fig, ax


def _nominal_models(refresh=False):
    """Cache RCE profiles and PICASO spectra; plotting edits reuse the cache."""
    import planets
    from photochem.utils import stars

    pl, st = planets.LTT1445Ab, planets.LTT1445A
    system = JointSystemPrior.from_chain()
    nominal_teq, nominal_teff, radius_ratio = system.transform(np.full(3, .5))
    database = ROOT/'picasofiles/opacities_photochem_0.1_250.0_R15000.db'
    stellar_file = ROOT/'inputs/sphinx.h5'
    config = dict(version=2, Teq=float(nominal_teq), Teff=float(nominal_teff),
                  radius_ratio=float(radius_ratio), mass=pl.mass, radius=pl.radius,
                  system_source=str(system.source), system_source_sha256=system.metadata()['source_sha256'],
                  stellar_irradiation_Teff=st.Teff,
                  star_radius=st.radius, metallicity=st.metal, logg=st.logg, chi=.2,
                  albedo=0., pressure_top_dyn_cm2=1., resolution=100,
                  atmospheres=[['H2O', .1], ['CO2', 0.001], ['SO2', 0.01]],
                  database_stat=[database.stat().st_size, database.stat().st_mtime_ns],
                  stellar_stat=[stellar_file.stat().st_size, stellar_file.stat().st_mtime_ns])
    fingerprint = json.dumps(config, sort_keys=True)
    cache = PRESENT_FIGURE_DIR/'nominal_models.npz'
    if cache.exists() and not refresh:
        with np.load(cache, allow_pickle=False) as saved:
            if saved['fingerprint'].item() == fingerprint:
                print(f'Using cached models: {cache}')
                return {name: saved[name].copy() for name in saved.files if name != 'fingerprint'}

    import input_files
    from photochem.extensions import hotrocks
    # Initialize only one climate/PICASO object, without loading emission grids.
    c = hotrocks.AdiabatClimateThermalEmission(
        Teq=nominal_teq, M_planet=pl.mass, R_planet=pl.radius, R_star=st.radius,
        Teff=st.Teff, metal=st.metal, logg=st.logg, catdir='sphinx',
        sphinx_filename=str(stellar_file))
    c.verbose = False
    c.P_top = 1.
    c.tidally_locked_dayside = True
    c.chi = .2
    c.rad.set_bolometric_flux(stars.equilibrium_temperature_inverse(nominal_teq, 0.))
    c.set_custom_albedo(np.array([1.]), np.array([0.]))
    c.initialize_picaso_from_clima(str(database), opannection_kwargs={'wave_range': [4., 25.]})
    wavelength = stars.grid_at_resolution(float(c.ptherm.opa.wave.min()),
                                          float(c.ptherm.opa.wave.max()), R=100)
    sphinx = hotrocks.sphinx_interpolator(str(stellar_file))
    wave, stellar_flux = sphinx(nominal_teff, st.metal, st.logg, rescale_to_Teff=False)
    stellar_flux = stars.rebin(stars.make_bins(wave), stellar_flux, wavelength)
    result = dict(wavelength=wavelength, stellar_flux=stellar_flux,
                  nominal_parameters=np.array(fingerprint))
    for species, pressure in config['atmospheres']:
        print(f'Computing {pressure:g} bar {species}...', flush=True)
        partial = np.full(len(c.species_names), 1.e-15)
        partial[c.species_names.index(species)] = pressure
        if not c.RCE_robust(partial*1.e6):
            raise RuntimeError(f'Climate did not converge: {pressure:g} bar {species}')
        result[f'{species}_pressure'] = np.r_[c.P_surf, c.P]/1.e6
        result[f'{species}_temperature'] = np.r_[c.T_surf, c.T]
        result[f'{species}_redistribution'] = np.array(c.heat_redistribution_parameters())
        _, flux, _ = c.fpfs_picaso(wavl=wavelength)
        result[f'{species}_ratio'] = flux/stellar_flux*radius_ratio**2
    rock_temperature = hotrocks.bare_rock_dayside_temperature(
        stars.equilibrium_temperature_inverse(nominal_teq, 0.), 0., 2/3)
    centers = (wavelength[1:]+wavelength[:-1])/2
    result['rock_temperature'] = np.array(rock_temperature)
    result['rock_ratio'] = np.pi*stars.blackbody_cgs(rock_temperature, centers/1.e4)/stellar_flux*radius_ratio**2
    if not all(np.isfinite(value).all() for value in result.values() if value.dtype.kind != 'U'):
        raise ValueError('Nonfinite model outputs')
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, fingerprint=np.array(fingerprint), **result)
    print(f'Saved model cache: {cache}')
    return result


def figure_models(refresh=False):
    """Five frames: data only, then add CO2, H2O, SO2, and bare rock."""
    from photochem.utils import stars
    from utils import make_lrs_data, photon_weighted_depth

    models = _nominal_models(refresh=refresh)
    data = make_lrs_data(ROOT/'data/LTT1445Ab_Sparta_16.txt')
    photometry = _final_photometry()
    observed_f15 = float(photometry['fpfs'][0])
    response = np.loadtxt(ROOT/'inputs/F1500W_response.txt').T
    wavelength = models['wavelength']
    centers = (wavelength[1:]+wavelength[:-1])/2

    plt.rcParams.update({'font.size': 13})
    fig, (pt_ax, spectrum_ax) = plt.subplots(1, 2, figsize=[10, 5],
                                            gridspec_kw={'width_ratios': [1, 1.4]})
    fig.subplots_adjust(left=.11, right=.98, bottom=.15, top=.76, wspace=.23)
    legend_handles = []
    model_artists = []
    degrees_of_freedom = len(data['fpfs']) + 1  # Fixed models: no fitted parameters.
    summary = dict(fit_tag=FIT_TAG, nominal_parameters=json.loads(models['nominal_parameters'].item()),
                   chi2_definition='Sum over 16 LRS bins and one F1500W point; parameters fixed, not fitted',
                   degrees_of_freedom=degrees_of_freedom,
                   F1500W_depth_ppm=float(observed_f15*1.e6),
                   F1500W_error_lower_ppm=float(photometry['err_lower'][0]*1.e6),
                   F1500W_error_upper_ppm=float(photometry['err_upper'][0]*1.e6),
                   models={})
    for name, label, color in [('CO2', r'0.001 bar CO$_2$', 'C2'),
                                ('H2O', r'0.1 bar H$_2$O', 'C0'),
                                ('SO2', r'0.01 bar SO$_2$', 'C5'),
                                ('rock', 'Zero-albedo bare rock', 'black')]:
        ratio = models[f'{name}_ratio']
        prediction_lrs = np.array([stars.rebin(wavelength, ratio, b)[0] for b in data['bins']])
        prediction_f15 = photon_weighted_depth(wavelength, ratio, models['stellar_flux'], *response)
        chi2_lrs = np.sum(((prediction_lrs-data['fpfs'])/data['err'])**2)
        chi2_f15 = -2*observation_loglike(photometry, np.array([prediction_f15]))
        chi2 = float(chi2_lrs+chi2_f15)
        reduced_chi2 = chi2/degrees_of_freedom
        handle, = spectrum_ax.plot(centers, ratio*1.e6, color=color, lw=1.7,
                                   label=label+rf'; $\chi^2_\nu={reduced_chi2:.1f}$')
        legend_handles.append(handle)
        artists = [handle]
        if name != 'rock':
            profile, = pt_ax.plot(models[f'{name}_temperature'], models[f'{name}_pressure'], color=color, lw=1.7)
            artists.append(profile)
        else:
            # No pressure coordinate exists for an airless surface. Mark its
            # temperature on the axis, rather than inventing a P–T profile.
            t_rock = float(models['rock_temperature'])
            marker, = pt_ax.plot(t_rock, 0., '^', color='black', markersize=7,
                                 transform=pt_ax.get_xaxis_transform(), clip_on=False)
            annotation = pt_ax.annotate(r'$T_\mathrm{rock}$', (t_rock, .02),
                                        xycoords=pt_ax.get_xaxis_transform(), xytext=(-5, 0),
                                        textcoords='offset points', fontsize=11, ha="right")
            artists.extend([marker, annotation])
        model_artists.append(artists)
        summary['models'][name] = dict(chi2=chi2, reduced_chi2=reduced_chi2, chi2_lrs=float(chi2_lrs),
            chi2_f1500w=float(chi2_f15), F1500W_prediction_ppm=float(prediction_f15*1.e6),
            surface_temperature_K=float(models['rock_temperature'] if name == 'rock'
                                        else models[f'{name}_temperature'][0]))

    lrs = spectrum_ax.errorbar(data['wv'], data['fpfs']*1.e6, xerr=data['wv_err'], yerr=data['err']*1.e6,
                               fmt='o', color='black', markersize=3, elinewidth=1, capsize=2,
                               label='MIRI LRS (Wachiraphan+25)', zorder=5)
    f15 = spectrum_ax.errorbar((13.6+16.5)/2, observed_f15*1.e6, xerr=(16.5-13.6)/2,
                               yerr=observation_yerr(photometry)*1.e6, fmt='s', color=F1500W_COLOR, markersize=6,
                               elinewidth=1, capsize=2, label='MIRI F1500W (Ih+2026; DDT)', zorder=5)
    pt_ax.set(xlabel='Temperature (K)', ylabel='Pressure (bar)', yscale='log', ylim=(1.5e-1, 1.e-6))
    spectrum_ax.set(xlabel=r'Wavelength ($\mu$m)', ylabel='Planet-to-star flux (ppm)', xlim=(4.8, 17.5))
    spectrum_ax.set_ylim(bottom=0)
    spectrum_ax.legend(handles=[lrs, f15], loc='upper left', bbox_to_anchor=(0., 1.),
                       frameon=False, fontsize=13)
    model_legend = fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(.5, 0.92),
                              ncol=2, frameon=False, fontsize=13)
    # Determine limits using all models, then freeze them before hiding artists.
    # Keep the complete legend layout too: unrevealed entries occupy blank slots.
    pt_ax.set_xlim(pt_ax.get_xlim())
    spectrum_ax.set_ylim(spectrum_ax.get_ylim())
    # Measure the complete legend once so even the data-only frame receives
    # the same crop. Keep its position and trim the unused canvas above it.
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    content_top = fig.get_tightbbox(fig.canvas.get_renderer()).y1
    export_bbox = Bbox.from_bounds(0., 0., fig.get_figwidth(),
                                  min(fig.get_figheight(), content_top + .06))
    output = PRESENT_FIGURE_DIR/'figure_models.png'
    summary['frames'] = []
    for count in range(5):
        for i, artists in enumerate(model_artists):
            for artist in artists:
                artist.set_visible(i < count)
        for i, (handle, text) in enumerate(zip(model_legend.get_lines(), model_legend.get_texts())):
            handle.set_visible(i < count)
            text.set_visible(i < count)
        # Retain the original filename for the final frame: five PNGs in total.
        frame = output if count == 4 else output.with_name(f'figure_models_{count}.png')
        fig.savefig(frame, dpi=300, bbox_inches=export_bbox)
        summary['frames'].append(dict(file=frame.name, visible_models=list(summary['models'])[:count]))
        print(f'Saved {frame}')
    summary['layout'] = dict(figsize=fig.get_size_inches().tolist(), dpi=300,
                            export_bounds_inches=export_bbox.bounds,
                            axes=[dict(bounds=ax.get_position().bounds, xlim=ax.get_xlim(),
                                       ylim=ax.get_ylim()) for ax in (pt_ax, spectrum_ax)])
    output.with_suffix('.json').write_text(json.dumps(summary, indent=2)+'\n')
    for name, values in summary['models'].items():
        print(f"{name}: reduced chi2 = {values['reduced_chi2']:.2f}, Ts = {values['surface_temperature_K']:.1f} K")
    return fig, (pt_ax, spectrum_ax)


def figure_retrieval(refresh=False):
    """Combined 16-bin joint-system-prior retrieval and total-pressure prior."""
    from scipy.stats import qmc
    from matplotlib.lines import Line2D
    from figure_joint import load_posteriors, spectral_bands, scenario_cases
    from retrieval_run_joint import JointRetrieval

    directory = RESULTS_DIR
    cases = scenario_cases(16)
    posteriors = load_posteriors(directory, cases)
    name = 'atm_16_F1500W_obs'
    posterior = posteriors[name]
    metadata = posterior['metadata']
    run = JointRetrieval(chain=Path(metadata['system_prior']['source_csv']).parent,
                         response=metadata['response_file'], lrs_bins=16)
    # Reuse the original figure's fingerprint-checked spectral cache. It includes
    # all three datasets, but this figure draws only the combined-case bands.
    bands, fingerprint = spectral_bands(
        run, posteriors, JOINT_FIGURE_DIR/'figure_joint_spectra.npz',
        max_spectra=0, seed=1445, refresh=refresh, cases=cases)
    lower, median, upper = bands[name]  # Already in ppm; pointwise +/-1 sigma.
    wavelength = (run.wavelength[:-1]+run.wavelength[1:])/2
    data = run.cases[name]['data_dict']
    # Confirm that the displayed data match the observations used in the fit.
    for key, values in data.items():
        if not np.array_equal(values, metadata['data'][key]):
            raise ValueError(f'Saved retrieval data differ from current data: {key}')

    gas_columns = [metadata['param_names'].index(k) for k in
                   ('log10PH2O', 'log10PCO2', 'log10PO2', 'log10PSO2')]
    # load_posteriors returns a canonical parameter order, starting with gases.
    total_pressure = np.log10(np.sum(10.**posterior['samples'][:, :4], axis=1))
    prior = run.cases[name]['prior']
    prior_low = prior(np.zeros(9))[:4]
    prior_high = prior(np.ones(9))[:4]
    unit = qmc.Sobol(4, scramble=True, seed=1445).random_base2(20)
    total_pressure_prior = np.log10(np.sum(10.**(prior_low+unit*(prior_high-prior_low)), axis=1))

    plt.rcParams.update({'font.size': 14})
    fig, (spectrum_ax, pressure_ax) = plt.subplots(1, 2, figsize=[10, 4],
                                                  gridspec_kw={'width_ratios': [1.3, 1]})
    fig.subplots_adjust(left=.085, right=.98, bottom=.18, top=.96, wspace=.24)
    spectrum_ax.fill_between(wavelength, lower, upper, color='C0', alpha=.2, linewidth=0)
    spectrum_ax.plot(wavelength, median, color='C0', lw=2, label='Climate-constrained retrieval')
    for photometric, color, marker, size, label in (
            (False, 'black', 'o', 3, 'MIRI LRS (Wachiraphan+25)'),
            (True, F1500W_COLOR, 's', 6, 'MIRI F1500W (Ih+2026; DDT)')):
        mask = data['photometric'] == photometric
        spectrum_ax.errorbar(data['wv'][mask], data['fpfs'][mask]*1.e6,
                             xerr=data['wv_err'][mask], yerr=observation_yerr(data, mask)*1.e6,
                             fmt=marker, color=color, markersize=size, elinewidth=1,
                             capsize=2, label=label, zorder=5)
    spectrum_ax.set(xlabel=r'Wavelength ($\mu$m)', ylabel='Planet-to-star flux (ppm)',
                    xlim=(4.8, 17.5), ylim=(0., 240.))
    spectrum_ax.legend(loc='upper left', bbox_to_anchor=(0., 1.), frameon=False, fontsize=12)

    # Cover the full induced prior, rather than truncating its high-pressure end.
    bins = np.arange(-5., 2.801, .4)
    pressure_ax.hist(total_pressure, bins=bins, density=True, histtype='step',
                     color='C0', lw=3, label='')
    pressure_ax.axvline(np.quantile(total_pressure, .975), color='C0', ls=':', lw=2)
    pressure_ax.hist(total_pressure_prior, bins=bins, density=True, histtype='step',
                     color='black', ls='--', lw=2.5, label='')
    pressure_ax.set(xlabel=r'Total surface pressure (bar)',
                    ylabel='Probability density', xlim=(bins[0], bins[-1]), ylim=(0., None))
    pressure_ax.set_ylim(top=pressure_ax.get_ylim()[1])
    pressure_ax.set_xlim(-3.5, pressure_ax.get_xlim()[1])
    xticks = np.array([-3, -2, -1, 0, 1, 2])
    pressure_ax.set_xticks(xticks)
    pressure_ax.set_xticklabels([rf'$10^{{{a}}}$' for a in xticks])
    # pressure_ax.legend(handles=[Line2D([], [], color='C0', lw=2, label='Posterior'),
    #                            Line2D([], [], color='black', lw=1.5, ls='--', label='Prior')],
    #                    loc='upper right', bbox_to_anchor=(1., 1.), frameon=False)

    output = PRESENT_FIGURE_DIR/'figure_retrieval.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    output.with_suffix('.json').write_text(json.dumps(dict(
        case=name, results_directory=str(directory), spectral_fingerprint=fingerprint,
        posterior_samples=len(total_pressure), prior_draws=len(total_pressure_prior),
        prior_gas_columns=gas_columns, prior_log10_partial_pressure_bounds=np.c_[prior_low, prior_high].tolist(),
        total_pressure_definition='log10(sum of the four sampled partial pressures in bar)',
        spectral_quantiles=[.15865525393145707, .5, .8413447460685429],
        total_pressure_97p5_upper_limit_bar=float(10.**np.quantile(total_pressure, .975))),
        indent=2)+'\n')
    print(f'Saved {output}')
    print(f'97.5% upper limit on total surface pressure: '
          f'{10.**np.quantile(total_pressure, .975):.6g} bar')
    return fig, (spectrum_ax, pressure_ax)


def figure_gas_posteriors():
    """Match retrieval_16_present.pdf using the updated combined posterior."""
    name = 'atm_16_F1500W_obs'
    directory = RESULTS_DIR/name
    metadata = json.loads((directory/'inputs.json').read_text())
    require_current_fit(metadata)
    posterior_file = directory/(name+'post_equal_weights.dat')
    samples = np.atleast_2d(np.loadtxt(posterior_file))
    plt.rcParams.update({'font.size': 14})
    fig, axes = plt.subplots(1, 4, figsize=[12, 3])
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(wspace=.15)
    gas_settings = [
        ('log10PH2O', 'H_2O', 'C0', 'right', .3, np.arange(-7, 3, 2)),
        ('log10PCO2', 'CO_2', 'C2', 'center', .5, np.arange(-7, 3, 2)),
        ('log10PO2', 'O_2', 'C4', 'right', .3, np.arange(-4, 3, 2)),
        ('log10PSO2', 'SO_2', 'C5', 'right', .3, np.arange(-7, -1, 2)),
    ]
    limits = {}
    for ax, (key, formula, color, alignment, note_x, ticks) in zip(axes, gas_settings):
        values = samples[:, metadata['param_names'].index(key)]
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite posterior samples for {key}')
        bins = np.histogram_bin_edges(values, bins=10)
        ax.hist(values, bins=bins, density=True, color=color, alpha=.3)
        ax.hist(values, bins=bins, density=True, color=color, histtype='step', lw=2)
        ax.set_xlim(values.min(), values.max())
        # Keep tick locations inside the sampled range without extending axes.
        ax.set_xticks(ticks[(ticks >= values.min()-.02) & (ticks <= values.max()+.02)])
        ax.set_xlabel(rf'$\log_{{10}}P_\mathrm{{{formula}}}$ (bar)')
        ax.set_yticks([])
        label_x = {'right': .98, 'center': .5, 'left': .02}[alignment]
        ax.text(label_x, .98, rf'$\mathrm{{{formula}}}$',
                size=15, color=color, transform=ax.transAxes, ha=alignment, va='top')
        upper = float(np.quantile(values, .975))
        ax.axvline(upper, color='black', lw=1)
        ax.text(upper, ax.get_ylim()[1]*.55, '97.5% UL', rotation=-90, size=12,
                va='center', ha='left')
        ax.text(note_x, .01, rf'$<{upper:.1f}$', size=17, color=color,
                transform=ax.transAxes, ha='center', va='bottom')
        limits[key] = dict(log10_upper_limit=upper, upper_limit_bar=float(10.**upper))
    axes[0].set_ylabel('Probability Density', labelpad=10)
    output = PRESENT_FIGURE_DIR/'retrieval_joint_16_present'
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('.pdf', '.png'):
        path = output.with_suffix(extension)
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved {path}')
    output.with_suffix('.json').write_text(json.dumps(dict(
        case=name, posterior_file=str(posterior_file), posterior_samples=len(samples),
        upper_limit_probability=.975, partial_pressure_upper_limits=limits), indent=2)+'\n')
    return fig, axes


def _baseline_joint(name='atm_16_F1500W_obs'):
    """Load an atmospheric joint-system-prior case in canonical parameter order."""
    from figure_joint import load_posteriors
    from retrieval_run_joint import JointRetrieval
    directory = RESULTS_DIR
    entry = load_posteriors(directory, cases=[(name, '', 'C0')])[name]
    metadata = entry['metadata']
    run = JointRetrieval(chain=Path(metadata['system_prior']['source_csv']).parent,
                         response=metadata['response_file'], lrs_bins=16)
    for key, values in run.cases[name]['data_dict'].items():
        if not np.array_equal(values, metadata['data'][key]):
            raise ValueError(f'Baseline observations changed: {key}')
    return run, entry


def _joint_profiles(run, entry, refresh=False):
    """Cache 150 profiles interpolated in the same grid coordinates as the fit."""
    import h5py
    from scipy.interpolate import RegularGridInterpolator
    from joint_system_prior import sha256
    grid_stat = run.grid_file.stat()
    fingerprint = json.dumps(dict(version=1, posterior_sha256=sha256(entry['posterior_file']),
        grid_stat=[grid_stat.st_size, grid_stat.st_mtime_ns], count=150, seed=1445), sort_keys=True)
    suffix = 'f1500w' if entry['metadata']['case'] == 'atm_F1500W_obs' else '16'
    cache = PRESENT_FIGURE_DIR/f'joint_{suffix}_profiles.npz'
    if cache.exists() and not refresh:
        with np.load(cache, allow_pickle=False) as saved:
            if saved['fingerprint'].item() == fingerprint:
                print(f'Using profile cache: {cache}')
                return saved['pressure'].copy(), saved['temperature'].copy(), saved['indices'].copy()
    samples = entry['samples']
    indices = np.random.default_rng(1445).choice(len(samples), min(150, len(samples)), replace=False)
    selected = samples[indices]
    coordinates = np.c_[selected[:, :4], 10.**selected[:, 4], selected[:, 5:7]]
    with h5py.File(run.grid_file, 'r') as grid:
        axes = tuple(grid['gridvals'][str(i)][:] for i in range(7))
        # Read only these two fields, avoiding the large stored emission array.
        pressure = 10.**RegularGridInterpolator(axes, np.log10(grid['results/P'][:]),
                                               bounds_error=True)(coordinates)
        temperature = RegularGridInterpolator(axes, grid['results/T'][:],
                                              bounds_error=True)(coordinates)
    if not np.isfinite(pressure).all() or not np.isfinite(temperature).all() or np.any(pressure <= 0):
        raise ValueError('Invalid interpolated P–T profiles')
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, fingerprint=np.array(fingerprint), pressure=pressure,
                        temperature=temperature, indices=indices)
    print(f'Saved profile cache: {cache}')
    return pressure, temperature, indices


def _save_joint_figure(fig, stem, description):
    description = dict(description, fit_tag=FIT_TAG)
    output = PRESENT_FIGURE_DIR/stem
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('.pdf', '.png'):
        path = output.with_suffix(extension)
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved {path}')
    output.with_suffix('.json').write_text(json.dumps(description, indent=2)+'\n')


def _figure_fit_posteriors(atm_name, stem, refresh=False):
    """Shared total-pressure, P–T, and nine-marginal layout."""
    from scipy.stats import qmc
    from figure_joint import PARAMETERS

    run, entry = _baseline_joint(atm_name)
    samples = entry['samples']
    pressure, temperature, profile_indices = _joint_profiles(run, entry, refresh)
    plt.rcParams.update({'font.size': 11})
    fig = plt.figure(figsize=[12, 8], facecolor='white')
    grid = fig.add_gridspec(3, 1, hspace=.38)
    top = grid[0].subgridspec(1, 2, wspace=.18)
    middle = grid[1].subgridspec(1, 4, wspace=.10)
    bottom = grid[2].subgridspec(1, 5, wspace=.12)
    top_axes = [fig.add_subplot(top[i]) for i in range(2)]
    gas_axes = [fig.add_subplot(middle[i]) for i in range(4)]
    nuisance_axes = [fig.add_subplot(bottom[i]) for i in range(5)]
    total = np.log10(np.sum(10.**samples[:, :4], axis=1))
    unit = qmc.Sobol(4, scramble=True, seed=1445).random_base2(20)
    prior = run.cases[atm_name]['prior']
    low, high = prior(np.zeros(9))[:4], prior(np.ones(9))[:4]
    total_prior = np.log10(np.sum(10.**(low+unit*(high-low)), axis=1))

    def histogram(ax, values, color, bins=10, label=None):
        edges = np.histogram_bin_edges(values, bins=bins)
        ax.hist(values, bins=edges, density=True, color=color, alpha=.3)
        ax.hist(values, bins=edges, density=True, color=color, histtype='step', lw=2, label=label)
        ax.set_yticks([])
        ax.set_xlim(values.min(), values.max())
        ax.grid(alpha=.4)

    def upper_limit(ax, values, color, note_x=.3):
        limit = float(np.quantile(values, .975))
        ax.axvline(limit, color='black', lw=1)
        ax.text(limit, ax.get_ylim()[1]*.55, '97.5% UL', rotation=-90, size=8,
                va='center', ha='left')
        rounded_limit = round(limit, 1)
        if rounded_limit == 0:
            rounded_limit = 0.  # Avoid displaying a rounded negative zero.
        ax.text(note_x, .01, rf'$<{rounded_limit:.1f}$', size=13, color=color,
                transform=ax.transAxes, ha='center', va='bottom',
                bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=.2) if color == 'black' else None)
        return dict(log10_upper_limit=limit, upper_limit_bar=float(10.**limit))

    ax = top_axes[0]
    bins = np.arange(-5., 2.801, .25)
    histogram(ax, total, 'black', bins=bins, label='Posterior')
    ax.hist(total_prior, bins=bins, density=True, color='black', histtype='step',
            lw=1, ls=':', label='Prior')
    ax.set(xlim=(-3.5, 2.75), xlabel=r'$\log_{10}$ Tot. Surf. Pressure (bar)', ylabel='Probability Density')
    ax.set_xticks(np.arange(-3, 3))
    limits = {'total': upper_limit(ax, total, 'black', note_x=.4)}
    ax.legend(ncol=2, columnspacing=.9, loc='upper left', bbox_to_anchor=(0., 1.), fontsize=9.5)
    ax.set_ylim(0., ax.get_ylim()[1]*1.2)
    ax = top_axes[1]
    for p, t in zip(pressure, temperature):
        ax.plot(t, np.log10(p/1.e6), color='black', lw=.15, alpha=.3)
        ax.plot(t[0], np.log10(p[0]/1.e6), 'o', color='black', ms=1, alpha=.3)
    ax.set(xlabel='Dayside Temperature (K)', ylabel=r'Pressure ($\log_{10}$ bar)', ylim=(1., -5.7))
    ax.set_yticks([1, 0, -1, -2, -3, -4, -5])
    ax.grid(alpha=.4)

    gases = [('H_2O', 'C0', 'right', .3), ('CO_2', 'C2', 'center', .5),
             ('O_2', 'C4', 'right', .3), ('SO_2', 'C5', 'right', .3)]
    for i, (ax, (formula, color, alignment, note_x)) in enumerate(zip(gas_axes, gases)):
        values = samples[:, i]
        histogram(ax, values, color)
        ax.set_xlabel(rf'$\log_{{10}}P_\mathrm{{{formula}}}$ (bar)')
        ticks = np.arange(-7 if i != 2 else -4, 3, 2)
        ax.set_xticks(ticks[(ticks >= values.min()-.02) & (ticks <= values.max()+.02)])
        ax.text(.5 if alignment == 'center' else .98, .98, rf'$\mathrm{{{formula}}}$',
                transform=ax.transAxes, ha=alignment, va='top', color=color, fontsize=13)
        limits[PARAMETERS[i]] = upper_limit(ax, values, color, note_x)
    gas_axes[0].set_ylabel('Probability Density', labelpad=10)
    labels = [r'$\log_{10}\chi$', 'Surface albedo', r'$T_\mathrm{eq}$ (K)',
              r'$T_{\star,\mathrm{eff}}$ (K)', r'$R_p/R_\star$']
    for i, (ax, label, color) in enumerate(zip(nuisance_axes, labels, ['C6', 'C7', 'C8', 'C9', 'C1'])):
        histogram(ax, samples[:, i+4], color)
        ax.set_xlabel(label)
        ax.locator_params(axis='x', nbins=4)
    nuisance_axes[0].set_ylabel('Probability Density', labelpad=10)
    _save_joint_figure(fig, stem, dict(case=atm_name,
        posterior_file=str(entry['posterior_file']), posterior_samples=len(samples),
        profile_method='Grid interpolation: log pressure, linear temperature; same seven model coordinates',
        profile_posterior_indices=profile_indices.tolist(), upper_limit_probability=.975,
        pressure_upper_limits=limits, prior_draws=len(total_prior)))
    return fig, (top_axes, gas_axes, nuisance_axes)


def figure_joint_posteriors(refresh=False):
    """Full posterior layout for the combined 16-bin LRS + F1500W fit."""
    return _figure_fit_posteriors('atm_16_F1500W_obs', 'retrieval_joint_16', refresh)


def figure_f1500w_posteriors(refresh=False):
    """Same posterior layout for the F1500W-only fit."""
    return _figure_fit_posteriors('atm_F1500W_obs', 'retrieval_joint_f1500w', refresh)


def _figure_fit_spectra(atm_name, rock_name, stem, refresh=False):
    """Shared atmospheric/bare-rock spectral comparison for a chosen dataset."""
    import pickle
    from photochem.extensions import hotrocks
    from photochem.utils import stars
    from figure_joint import load_posteriors, spectral_bands, scenario_cases
    from joint_system_prior import sha256

    run, entry = _baseline_joint(atm_name)
    directory = RESULTS_DIR
    cases = scenario_cases(16)
    posteriors = load_posteriors(directory, cases)
    atmospheric, fingerprint = spectral_bands(run, posteriors,
        JOINT_FIGURE_DIR/'figure_joint_spectra.npz', 0, 1445, refresh, cases)
    rock_file = directory/rock_name/(rock_name+'post_equal_weights.dat')
    rock_metadata = json.loads((directory/rock_name/'inputs.json').read_text())
    rock_data = np.atleast_2d(np.loadtxt(rock_file))
    rock_samples = rock_data[:, [rock_metadata['param_names'].index(k)
                                for k in ('albedo', 'Teq', 'Teff', 'R_planet_star')]]
    sphinx = hotrocks.sphinx_interpolator(str(run.stellar_file))
    import planets
    st = planets.LTT1445A
    wavelength = (run.wavelength[:-1]+run.wavelength[1:])/2

    def rock_spectrum(x):
        albedo, teq, teff, ror = x
        day_temperature = hotrocks.bare_rock_dayside_temperature(
            stars.equilibrium_temperature_inverse(teq, albedo), albedo, 2/3)
        wave, flux = sphinx(teff, st.metal, st.logg, rescale_to_Teff=False)
        stellar_flux = stars.rebin(stars.make_bins(wave), flux, run.wavelength)
        return (1-albedo)*np.pi*stars.blackbody_cgs(day_temperature, wavelength/1.e4)/stellar_flux*ror**2*1.e6

    suffix = 'f1500w' if atm_name == 'atm_F1500W_obs' else '16'
    cache = PRESENT_FIGURE_DIR/f'joint_{suffix}_rock_spectra.npz'
    rock_fingerprint = json.dumps(dict(version=1, posterior_sha256=sha256(rock_file),
        stellar_sha256=sha256(run.stellar_file), planets_sha256=sha256(ROOT/'planets.py'),
        wavelength=run.wavelength.tolist(), quantiles=[.15865525393145707, .5, .8413447460685429]),
        sort_keys=True)
    rock_bands = None
    if cache.exists() and not refresh:
        with np.load(cache, allow_pickle=False) as saved:
            if saved['fingerprint'].item() == rock_fingerprint:
                rock_bands = saved['bands'].copy()
                print(f'Using bare-rock spectral cache: {cache}')
    if rock_bands is None:
        print(f'Evaluating {len(rock_samples)} bare-rock posterior spectra for {rock_name}...', flush=True)
        spectra = np.array([rock_spectrum(x) for x in rock_samples])
        if not np.isfinite(spectra).all():
            raise ValueError('Nonfinite bare-rock posterior spectra')
        rock_bands = np.quantile(spectra, [.15865525393145707, .5, .8413447460685429], axis=0)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, fingerprint=np.array(rock_fingerprint), bands=rock_bands)
    evidence = {}
    for name in (atm_name, rock_name):
        with (directory/name/(name+'.pkl')).open('rb') as stream:
            result = pickle.load(stream)
        evidence[name] = dict(logZ=float(result['logZ']), logZerr=float(result['logZerr']))

    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(1, 1, figsize=[10, 3], facecolor='white')
    handles = []
    for name, bands, color, label in ((atm_name, atmospheric[atm_name], 'C0', 'Atmospheric model'),
                                     (rock_name, rock_bands, '0.5', 'Bare-rock model')):
        low, median, high = bands
        ax.fill_between(wavelength, low, high, color=color, alpha=.3)
        handle, = ax.plot(wavelength, median, color=color, lw=1,
                          label=label+'\n'+rf'($\ln Z={evidence[name]["logZ"]:.1f}$)')
        handles.append(handle)
    center = run.system.transform(np.full(3, .5))
    zero_reference, = ax.plot(wavelength, rock_spectrum(np.r_[0., center]), color='black',
                              lw=1, label='Zero-albedo bare rock')
    data = run.cases[atm_name]['data_dict']
    data_handles = []
    for photometric, color, marker, label in ((False, 'black', 'o', 'MIRI LRS (Wachiraphan+25)'),
            (True, F1500W_COLOR, 's', 'MIRI F1500W (Ih+2026; DDT)')):
        mask = data['photometric'] == photometric
        if not np.any(mask):
            continue
        data_handles.append(ax.errorbar(data['wv'][mask], data['fpfs'][mask]*1.e6,
            xerr=data['wv_err'][mask], yerr=observation_yerr(data, mask)*1.e6, fmt=marker,
            color=color, ms=3 if not photometric else 5, elinewidth=.6, capsize=1.5, label=label, zorder=5))
    # The display response is weighted by stellar photon flux, matching the
    # photometric observation operator; its height is arbitrary for display.
    wave, flux = sphinx(center[1], st.metal, st.logg, rescale_to_Teff=False)
    response_wave, response = run.response
    weighted_response = response_wave*response*np.interp(response_wave, wave, flux)
    weighted_response /= weighted_response.max()
    ax.fill_between(response_wave, 0., .11*weighted_response, color='0.5', alpha=.5,
                    transform=ax.get_xaxis_transform(), lw=0)
    ax.text(15., .13, 'MIRI F1500W', ha='center', va='bottom', fontsize=9,
            transform=ax.get_xaxis_transform())
    ax.set(xlabel=r'Wavelength ($\mu$m)', ylabel='Planet-to-star flux (ppm)',
           xlim=(4.8, 17.5), ylim=(0., 240.))
    ax.set_xticks(np.arange(5, 18))
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.grid(alpha=.4)
    ax.legend(handles=data_handles+handles+[zero_reference], loc='upper left',
              bbox_to_anchor=(0., 1.), fontsize=9, frameon=False)
    _save_joint_figure(fig, stem, dict(
        atmospheric_case=atm_name, bare_rock_case=rock_name, evidence=evidence,
        atmospheric_posterior_samples=len(entry['samples']), bare_rock_posterior_samples=len(rock_samples),
        spectral_quantiles=[.15865525393145707, .5, .8413447460685429],
        atmospheric_fingerprint=fingerprint, bare_rock_fingerprint=rock_fingerprint,
        zero_albedo_reference_parameters=np.r_[0., center].tolist(),
        response_display='lambda * response * stellar flux, normalized; arbitrary plotted height'))
    return fig, ax


def figure_joint_spectra(refresh=False):
    """Spectral comparison for the combined 16-bin LRS + F1500W fit."""
    return _figure_fit_spectra('atm_16_F1500W_obs', 'rock_16_F1500W_obs', 'spectra_joint_16', refresh)


def figure_f1500w_spectra(refresh=False):
    """Same spectral layout using only F1500W fits, evidence, and observation."""
    return _figure_fit_spectra('atm_F1500W_obs', 'rock_F1500W_obs', 'spectra_joint_f1500w', refresh)


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    _ = threadpool_limits(limits=4)
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure', choices=['data', 'models', 'retrieval', 'gases',
                                           'joint-posteriors', 'joint-spectra',
                                           'f1500w-posteriors', 'f1500w-spectra', 'all'], default='all')
    parser.add_argument('--refresh-models', action='store_true')
    parser.add_argument('--refresh-spectra', action='store_true')
    args = parser.parse_args()
    if args.figure in ('data', 'all'):
        figure_data()
    if args.figure in ('models', 'all'):
        figure_models(refresh=args.refresh_models)
    if args.figure in ('retrieval', 'all'):
        figure_retrieval(refresh=args.refresh_spectra)
    if args.figure in ('gases', 'all'):
        figure_gas_posteriors()
    if args.figure in ('joint-posteriors', 'all'):
        figure_joint_posteriors(refresh=args.refresh_models)
    if args.figure in ('joint-spectra', 'all'):
        figure_joint_spectra(refresh=args.refresh_spectra)
    if args.figure in ('f1500w-posteriors', 'all'):
        figure_f1500w_posteriors(refresh=args.refresh_models)
    if args.figure in ('f1500w-spectra', 'all'):
        figure_f1500w_spectra(refresh=args.refresh_spectra)
    plt.close('all')
