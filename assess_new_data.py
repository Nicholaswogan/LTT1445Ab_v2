"""Audit grid coverage and optionally run paired climate sensitivity calculations.

Run from the repo root in conda env ltt. No sampling or grid files are changed.
Uses unrounded final-fit draws for system diagnostics and sensitivity inputs.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from joint_system_prior import DEFAULT_CHAIN, DERIVED_DIR, sha256


def audit():
    from astropy import constants as c
    source, = DEFAULT_CHAIN.glob('MCMCResults*.csv')
    draws = pd.read_csv(source)
    mean = lambda key: float(draws[key].mean())
    sd = lambda key: float(draws[key].std(ddof=1))
    temperatures = draws.teff/np.sqrt(2*draws.aor__0)
    teq, teq_sd = float(temperatures.mean()), float(temperatures.std(ddof=1))
    radius = mean('r_pl__0')
    ror = mean('ror__0')
    diagnostics = []
    for path in sorted(DEFAULT_CHAIN.glob('trace.trace/*/metadata.json')):
        trace = json.loads(path.read_text())
        stats = trace['_stats'][0]
        divergence = np.asarray(stats['diverging'], dtype=bool)
        tune = np.asarray(stats['tune'], dtype=bool)
        diagnostics.append(dict(chain=int(path.parent.name), draws=len(divergence),
            post_warmup_divergences=int(np.count_nonzero(divergence & ~tune))))
    grids = {}
    for filename in ('results/LTT1445Ab.h5', 'results/LTT1445Ab_hot.h5'):
        with h5py.File(filename) as f:
            names = f['gridnames'].asstr()[:]
            axes = {name: f['gridvals'][str(i)][:].tolist() for i, name in enumerate(names)}
            w = f['common/wavl'][:]
            done = f['completed'][:]
            converged = f['results/converged'][:]
            grids[filename] = dict(axes=axes, wavelength_um=[float(w[0]), float(w[-1])],
                                   completed=int(np.count_nonzero(done)), total=int(done.size),
                                   converged=int(np.count_nonzero(converged)))
    return dict(
        source=str(source), source_sha256=sha256(source), sampling_diagnostics=diagnostics,
        warning='All source draws retained, including flagged divergent transitions; review diagnostics with fit author.',
        teq_zero_albedo_K=teq, teq_sd_K=teq_sd,
        mass_earth=mean('m_pl__0'), radius_earth=radius, teff_K=mean('teff'),
        gravity_ratio_to_grid=(mean('m_pl__0')/2.73)*(1.34/radius)**2,
        radius_ratio=ror, radius_ratio_sd=sd('ror__0'),
        eclipse_area_ratio_to_legacy=(ror/0.0454)**2,
        stellar_logg_from_mean_mass_radius=float(np.log10(
            (c.G*c.M_sun*mean('m_star')/(c.R_sun*mean('r_star'))**2).to('cm/s2').value)),
        grids=grids)


def sensitivity(report, response_file=None):
    import LTT1445Ab_grid as grid
    from photochem.utils import stars
    from utils import make_lrs_data, rebin_spectra_to_data, photon_weighted_depth
    from photochem.extensions import hotrocks
    data = make_lrs_data('data/LTT1445Ab_Sparta_16.txt')
    teq = report['teq_zero_albedo_K']
    radius = report['radius_earth']
    models = [('legacy', grid.NOMINAL_CLIMATE_MODEL),
              ('new_mass_radius', grid.initialize_model(mass=report['mass_earth'], radius=radius)[0])]
    sphinx = hotrocks.sphinx_interpolator('inputs/sphinx.h5')
    w, fs = sphinx(report['teff_K'], -0.34, 4.97, rescale_to_Teff=False)
    fs = stars.rebin(stars.make_bins(w), fs, grid.WAVL)
    response = None if response_file is None else np.loadtxt(response_file).T
    with h5py.File('results/LTT1445Ab.h5') as f:
        axes = tuple(f['gridvals'][str(i)][:] for i in range(len(f['gridvals'])))
        interp = RegularGridInterpolator(axes, f['results/fp'][:], bounds_error=True)
    # Sample atmospheric transitions and the thin limit; not an exhaustive validation.
    cases = [('thin', [-7, -7, -5, -7]),
             ('O2_CO2', [-7, -2, 0, -7]),
             ('CO2', [-7, -1, -5, -7]),
             ('H2O', [-3, -7, -5, -7]),
             ('SO2', [-7, -7, -5, -4]),
             ('CO2_between', [-7, -1.75, 0, -7]),
             ('H2O_between', [-3.5, -7, -5, -7])]
    rows = []
    for name, pressures in cases:
        x = np.array(pressures+([0.125, 0.15, teq] if name.endswith('between') else [0.2, 0.2, teq]))
        depths = {}
        convergence = {}
        high_resolution_f15 = None
        for label, model in models:
            grid.NOMINAL_CLIMATE_MODEL = model
            out = grid.model(x)
            convergence[label] = bool(out['converged'])
            if not out['converged']:
                raise RuntimeError(f'Climate failed: {name}, {label}')
            depths[label] = out['fp']/fs*report['radius_ratio']**2
            if label == 'legacy' and name in ('O2_CO2', 'CO2_between') and response is not None:
                wh, fph, _ = model.fpfs_picaso(R=1000)
                ws, fsh = sphinx(report['teff_K'], -0.34, 4.97, rescale_to_Teff=False)
                fsh = stars.rebin(stars.make_bins(ws), fsh, wh)
                high_resolution_f15 = float(photon_weighted_depth(
                    wh, fph/fsh*report['radius_ratio']**2, fsh, *response))*1e6
        depths['interpolated'] = interp(x)[0]/fs*report['radius_ratio']**2
        binned = {key: rebin_spectra_to_data(grid.WAVL, val, data) for key, val in depths.items()}
        f15 = {key: (float(stars.rebin(grid.WAVL, val, np.array([13.6, 16.5]))[0])
                     if response is None else float(photon_weighted_depth(grid.WAVL, val, fs, *response)))
               for key, val in depths.items()}
        row = dict(case=name, x=x.tolist(), converged=convergence,
                   f1500w_method='legacy rectangular ratio average' if response is None else 'photon throughput',
                   f1500w_depth_ppm={k: v*1e6 for k, v in f15.items()},
                   mass_radius_max_abs_LRS_change_ppm=float(np.max(np.abs(binned['new_mass_radius']-binned['legacy']))*1e6),
                   mass_radius_F1500W_change_ppm=(f15['new_mass_radius']-f15['legacy'])*1e6,
                   interpolation_max_abs_LRS_error_ppm=float(np.max(np.abs(binned['interpolated']-binned['legacy']))*1e6),
                   interpolation_F1500W_error_ppm=(f15['interpolated']-f15['legacy'])*1e6)
        if high_resolution_f15 is not None:
            row['direct_R1000_F1500W_depth_ppm'] = high_resolution_f15
            row['saved_R100_minus_R1000_F1500W_ppm'] = f15['legacy']*1e6-high_resolution_f15
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sensitivity', action='store_true')
    parser.add_argument('--response', type=Path, default=Path('inputs/F1500W_response.txt'))
    parser.add_argument('--output', type=Path, default=DERIVED_DIR/'new_data_assessment.json')
    args = parser.parse_args()
    report = audit()
    if args.sensitivity:
        report['sensitivity'] = sensitivity(report, args.response)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
