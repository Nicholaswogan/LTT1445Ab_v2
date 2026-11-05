"""Audit grid coverage and optionally run paired climate sensitivity calculations.

Run from the repo root in conda env ltt. No sampling or grid files are changed.
The rounded joint-fit CSV is used for diagnostics, never to manufacture priors.
"""
import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator

SUMMARY = Path('data/LTT1445_Joint__Alltransiteclipse_binDDT60s_v2_e_w/ChainSummary_LTT1445_Joint__Alltransiteclipse_binDDT60s_v2_e_w.csv')


def audit():
    from astropy import constants as c
    with SUMMARY.open() as f:
        rows = {r['Parameter']: r for r in csv.DictReader(f)}
    mean = lambda key: float(rows[key]['mean'])
    sd = lambda key: float(rows[key]['sd'])
    teq = mean('teff')/np.sqrt(2*mean('aor[0]'))
    # Delta-method diagnostic only; covariance is unavailable in the summary.
    teq_sd = teq*np.hypot(sd('teff')/mean('teff'), 0.5*sd('aor[0]')/mean('aor[0]'))
    rho_earth = (c.M_earth/(4*np.pi*c.R_earth**3/3)).to('g/cm3').value
    radius = (mean('m_pl[0]')*rho_earth/mean('density_pl[0]'))**(1/3)
    ror = mean('ror[0]')
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
        source=str(SUMMARY),
        warning='Rounded marginal summaries: diagnostics only, not a joint posterior or valid radius-ratio prior.',
        teq_zero_albedo_K=teq, teq_sd_ignoring_covariance_K=teq_sd,
        mass_earth=mean('m_pl[0]'), radius_from_mean_mass_and_density_earth=radius,
        gravity_ratio_to_grid=(mean('m_pl[0]')/2.73)*(1.34/radius)**2,
        radius_ratio_rounded=ror, radius_ratio_sd_rounded=sd('ror[0]'),
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
    radius = report['radius_from_mean_mass_and_density_earth']
    models = [('legacy', grid.NOMINAL_CLIMATE_MODEL),
              ('new_mass_radius', grid.initialize_model(mass=report['mass_earth'], radius=radius)[0])]
    sphinx = hotrocks.sphinx_interpolator('inputs/sphinx.h5')
    w, fs = sphinx(3340., -0.34, 4.97, rescale_to_Teff=False)
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
            depths[label] = out['fp']/fs*report['radius_ratio_rounded']**2
            if label == 'legacy' and name in ('O2_CO2', 'CO2_between') and response is not None:
                wh, fph, _ = model.fpfs_picaso(R=1000)
                ws, fsh = sphinx(3340., -0.34, 4.97, rescale_to_Teff=False)
                fsh = stars.rebin(stars.make_bins(ws), fsh, wh)
                high_resolution_f15 = float(photon_weighted_depth(
                    wh, fph/fsh*report['radius_ratio_rounded']**2, fsh, *response))*1e6
        depths['interpolated'] = interp(x)[0]/fs*report['radius_ratio_rounded']**2
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
    parser.add_argument('--response', type=Path)
    parser.add_argument('--output', type=Path, default=Path('results/new_data_assessment.json'))
    args = parser.parse_args()
    report = audit()
    if args.sensitivity:
        report['sensitivity'] = sensitivity(report, args.response)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
