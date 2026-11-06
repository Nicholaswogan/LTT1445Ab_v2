"""Recompute joint-posterior climates and benchmark derived f/Bond albedo.

Run in conda env ltt, from the repository root:
    python posterior_climate.py                 # 1000 samples, four processes
    python posterior_climate.py --samples 10    # short benchmark
    python posterior_climate.py --plot-only     # plot saved 1000-sample run
    python posterior_climate.py --case f1500w   # F1500W-only, separate output
    python posterior_climate.py --extend results/final_gp_v1/posterior_climate_1000.h5 --samples 5000

Saves HDF5 plus JSON and three-column CSV sidecars in results/final_gp_v1/.
Checkpoints every 25 completed samples; failed samples are retained and flagged.
--extend retains completed calculations and adds posterior draws from the full
weighted MultiNest output (with replacement). Repeated draws reuse calculations;
they do not increase the independent information in the retrieval posterior.

Uses the original grid's fixed mass, radius, and stellar irradiation shape.
Sampled Teff and Rp/Rstar affect the observation operator, not these climates.
No PICASO calculations, retrieval changes, or grid writes are performed.
"""

import os
# One native thread per worker; four independent climate objects by default.
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import h5py

from joint_system_prior import ROOT, FIT_TAG, RESULTS_DIR, DERIVED_DIR, require_current_fit
JOINT_FIGURE_DIR = ROOT / 'figures/joint'
DEFAULT_POSTERIOR = RESULTS_DIR/'atm_16_F1500W_obs/atm_16_F1500W_obspost_equal_weights.dat'
F1500W_POSTERIOR = RESULTS_DIR/'atm_F1500W_obs/atm_F1500W_obspost_equal_weights.dat'
PARAMETERS = ['log10PH2O', 'log10PCO2', 'log10PO2', 'log10PSO2', 'log10chi',
              'albedo', 'Teq', 'Teff', 'R_planet_star']


def initialize_worker():
    global CLIMATE, STARS, INIT_SECONDS, THREAD_LIMIT
    started = time.perf_counter()
    os.chdir(ROOT)
    import input_files  # Establish local reference-data paths, as in grid runner.
    import planets
    from photochem.extensions import hotrocks
    from photochem.utils import stars
    from threadpoolctl import threadpool_limits
    THREAD_LIMIT = threadpool_limits(limits=1)
    pl, st = planets.LTT1445Ab, planets.LTT1445A
    CLIMATE = hotrocks.AdiabatClimateThermalEmission(
        Teq=pl.Teq, M_planet=pl.mass, R_planet=pl.radius, R_star=st.radius,
        Teff=st.Teff, metal=st.metal, logg=st.logg, catdir='sphinx',
        sphinx_filename='inputs/sphinx.h5')
    CLIMATE.verbose = False
    CLIMATE.P_top = 1.
    CLIMATE.tidally_locked_dayside = True
    STARS = stars
    INIT_SECONDS = time.perf_counter() - started


def compute_sample(index, sample):
    started = time.perf_counter()
    c = CLIMATE
    result = dict(posterior_index=int(index), parameters=sample.tolist(),
                  worker_pid=os.getpid(), worker_initialization_seconds=INIT_SECONDS,
                  converged=False)
    try:
        c.rad.set_bolometric_flux(STARS.equilibrium_temperature_inverse(sample[6], 0.))
        c.set_custom_albedo(np.array([1.]), np.array([sample[5]]))
        c.chi = 10.**sample[4]
        pressures = np.full(len(c.species_names), 1.e-15)
        for species, logp in zip(('H2O', 'CO2', 'O2', 'SO2'), sample[:4]):
            pressures[c.species_names.index(species)] = 10.**logp
        # Use the same robust solver and default guesses as the original grid.
        result['converged'] = bool(c.RCE_robust(pressures * 1.e6))
        if result['converged']:
            tau, k, f = c.heat_redistribution_parameters()
            bond = c.rad.wrk_sol.fup_n[-1] / c.rad.wrk_sol.fdn_n[-1]
            values = np.array([tau, k, f, bond, c.T_surf])
            if not np.all(np.isfinite(values)):
                raise ValueError('Nonfinite climate diagnostics')
            if not (0.25 - 1.e-8 <= f <= 2./3. + 1.e-8 and 0. <= bond <= 1.):
                raise ValueError('Climate diagnostics outside physical bounds')
            result.update(tau_LW=float(tau), k_term=float(k), f=float(f),
                          bond_albedo=float(bond), surface_albedo=float(sample[5]),
                          surface_temperature_K=float(c.T_surf))
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
    result['seconds'] = time.perf_counter() - started
    return result


def save_results(output, provenance, results, summary=None):
    """Atomically save completed rows, keeping failures rather than resampling."""
    rows = sorted(results, key=lambda r: r.get('sample_id', r['posterior_index']))
    report = dict(provenance, completed_samples=len(rows),
                  run_finished=summary is not None, summary=summary, results=rows)
    temporary = output.with_suffix('.tmp.h5')
    with h5py.File(temporary, 'w') as f:
        f.attrs['schema_version'] = 1
        f.attrs['fit_tag'] = FIT_TAG
        f.attrs['run_finished'] = report['run_finished']
        f.attrs['requested_samples'] = provenance['requested_samples']
        f.attrs['completed_samples'] = len(rows)
        f.attrs['seed'] = provenance['seed']
        f.attrs['posterior'] = provenance['posterior']
        f.attrs['posterior_sha256'] = provenance['posterior_sha256']
        f.attrs['assumptions'] = provenance['assumptions']
        string_type = h5py.string_dtype('utf-8')
        f.create_dataset('param_names', data=PARAMETERS, dtype=string_type)
        f.create_dataset('parameters', data=np.array([r['parameters'] for r in rows], dtype=float)
                         .reshape(-1, len(PARAMETERS)))
        f.create_dataset('posterior_index', data=[r['posterior_index'] for r in rows], dtype='i8')
        f.create_dataset('sample_id', data=[r.get('sample_id', i) for i, r in enumerate(rows)], dtype='i8')
        f.create_dataset('posterior_source', data=[r.get('posterior_source', 'equal_weight') for r in rows],
                         dtype=string_type)
        f.create_dataset('converged', data=[r['converged'] for r in rows], dtype='?')
        f.create_dataset('valid', data=[r['converged'] and 'error' not in r for r in rows], dtype='?')
        f.create_dataset('error', data=[r.get('error', '') for r in rows], dtype=string_type)
        for name in ('tau_LW', 'k_term', 'f', 'bond_albedo', 'surface_temperature_K',
                     'seconds', 'worker_initialization_seconds'):
            f.create_dataset(name, data=[r.get(name, np.nan) for r in rows])
        # Surface albedo is known even for a failed climate calculation.
        f.create_dataset('surface_albedo', data=[r['parameters'][5] for r in rows])
        f.create_dataset('sample_weight', data=np.ones(len(rows)))
        f.create_dataset('report_json', data=json.dumps(report), dtype=string_type)
        f['parameters'].attrs['description'] = 'Original equal-weight posterior samples; columns in param_names'
        f['f'].attrs['description'] = 'Koll redistribution parameter, not chi'
        f['bond_albedo'].attrs['description'] = 'TOA upward/downward bolometric stellar flux ratio'
        f['surface_temperature_K'].attrs['units'] = 'K'
    temporary.replace(output)
    json_output = output.with_suffix('.json')
    json_temporary = output.with_suffix('.tmp.json')
    json_temporary.write_text(json.dumps(report, indent=2) + '\n')
    json_temporary.replace(json_output)
    if summary is not None:
        # Keep failed draws as NaNs, never silently condition on convergence.
        csv_output = output.with_suffix('.csv')
        csv_temporary = output.with_suffix('.tmp.csv')
        values = np.array([[r.get('f', np.nan), r.get('bond_albedo', np.nan),
                            r['parameters'][5]] for r in rows], dtype=float).reshape(-1, 3)
        np.savetxt(csv_temporary, values, delimiter=',',
                   header='f_term,bond_albedo,surface_albedo', comments='')
        csv_temporary.replace(csv_output)


def prepare_extension(filename, posterior_file, metadata, samples, seed):
    """Validate saved work and plan reproducible additions from weighted draws."""
    with h5py.File(filename, 'r') as saved:
        report = json.loads(saved['report_json'][()])
        if saved.attrs.get('fit_tag') != FIT_TAG:
            raise ValueError('Saved climates are from a different system fit')
    posterior_hash = hashlib.sha256(posterior_file.read_bytes()).hexdigest()
    if report['posterior_sha256'] != posterior_hash:
        raise ValueError('Saved climates do not match the selected case posterior')
    results = report.pop('results')
    report.pop('completed_samples', None)
    report.pop('run_finished', None)
    previous_summary = report.pop('summary', None)
    for i, row in enumerate(results):
        row.setdefault('sample_id', i)
        row.setdefault('posterior_source', 'equal_weight')
    if len({r['sample_id'] for r in results}) != len(results):
        raise ValueError('Duplicate sample IDs in saved climates')
    if samples < len(results):
        raise ValueError(f'Already have {len(results)} draws; --samples is the desired total')
    weighted_file = posterior_file.with_name(
        posterior_file.name.removesuffix('post_equal_weights.dat')+'.txt')
    if not posterior_file.name.endswith('post_equal_weights.dat'):
        raise ValueError('--extend requires a native MultiNest post_equal_weights.dat posterior')
    weighted_hash = hashlib.sha256(weighted_file.read_bytes()).hexdigest()
    weighted = np.atleast_2d(np.loadtxt(weighted_file))
    if weighted.shape[1] != len(metadata['param_names'])+2 or not np.isfinite(weighted).all():
        raise ValueError('Invalid weighted MultiNest output')
    weights = weighted[:, 0]
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError('Invalid posterior weights')
    weights = weights/weights.sum()
    columns = [metadata['param_names'].index(p)+2 for p in PARAMETERS]
    parameters = weighted[:, columns]
    plan = report.get('extension_plan', [])
    if plan:
        if report['weighted_posterior_sha256'] != weighted_hash:
            raise ValueError('Weighted posterior changed since the extension was planned')
        if samples != report['requested_samples']:
            raise ValueError('Finish the pending extension before changing its target')
    else:
        if len(results) != report['requested_samples']:
            raise ValueError('Original run is incomplete; finish it before extending')
        indices = np.random.default_rng(seed).choice(len(weights), samples-len(results),
                                                     replace=True, p=weights)
        first_id = max((r['sample_id'] for r in results), default=-1)+1
        plan = [dict(sample_id=first_id+i, posterior_index=int(index))
                for i, index in enumerate(indices)]
        report['extension_plan'] = plan
        report['extension_seed'] = seed
        report['extension_base_samples'] = len(results)
        report['extension_base_summary'] = previous_summary
    report.update(requested_samples=samples, weighted_posterior=str(weighted_file.resolve()),
                  weighted_posterior_sha256=weighted_hash,
                  weighted_posterior_effective_samples=float(1./np.sum(weights**2)),
                  extension_sampling='Independent categorical draws from normalized MultiNest weights, with replacement',
                  extension_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    complete = {r['sample_id'] for r in results}
    pending = [dict(row, parameters=parameters[row['posterior_index']].tolist(),
                    posterior_source='weighted') for row in plan if row['sample_id'] not in complete]
    return report, results, pending


def plot_results(filename, output_prefix):
    """Plot equal-weight joint posteriors without any climate reevaluations."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.ndimage import gaussian_filter

    with h5py.File(filename, 'r') as saved:
        if not saved.attrs['run_finished']:
            raise ValueError('Climate run is not marked finished')
        valid = saved['valid'][:]
        case_label = ('F1500W alone' if Path(saved.attrs['posterior']).parent.name
                      == 'atm_F1500W_obs' else 'LRS + F1500W')
        if not np.all(valid):
            raise ValueError('Some samples failed; review failures before plotting the posterior')
        f = saved['f'][:]
        albedos = [saved[name][:] for name in ('surface_albedo', 'bond_albedo')]
    plt.rcdefaults()
    plt.rcParams.update({'font.size': 14})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    fig.subplots_adjust(left=.09, right=.98, bottom=.17, top=.96, wspace=.10)
    # Resolve the narrow concentration near f=2/3. Reflection at the physical
    # upper boundary prevents smoothing probability into an unphysical region.
    f_upper = 2./3.
    f_lower = max(.25, float(f.min()) - .01)
    yedges = np.linspace(f_lower, f_upper, 121)
    xedges = np.linspace(0., .4, 51)
    report = dict(source=str(filename), samples=len(f), smoothing_sigma_bins=3.0,
                  credible_probabilities=[.68, .95], panels={})
    for ax, x, name, label in zip(axes, albedos,
                                 ('surface_albedo', 'bond_albedo'),
                                 ('Surface albedo', 'Bond albedo')):
        if not (np.all((x >= xedges[0]) & (x <= xedges[-1]))
                and np.all((f >= yedges[0]) & (f <= yedges[-1]))):
            raise ValueError('Plot grid does not contain all posterior samples')
        counts, _, _ = np.histogram2d(x, f, bins=(xedges, yedges))
        density = gaussian_filter(counts.astype(float), sigma=3.0, mode='reflect')
        density /= density.sum()
        ordered = np.sort(density.ravel())[::-1]
        cumulative = np.cumsum(ordered)
        thresholds = [float(ordered[min(np.searchsorted(cumulative, p), len(ordered)-1)])
                      for p in (.95, .68)]
        xc = (xedges[:-1] + xedges[1:])/2
        yc = (yedges[:-1] + yedges[1:])/2
        # Extend cell centers to physical boundaries so contours are closed by
        # the boundary rather than suggesting support beyond f=2/3.
        xx = np.r_[xedges[0], xc, xedges[-1]]
        yy = np.r_[yedges[0], yc, yedges[-1]]
        zz = np.pad(density.T, 1, mode='edge')
        ax.contourf(xx, yy, zz, levels=[thresholds[0], thresholds[1], density.max()*1.001],
                    colors=['#d6e6f5', '#83b3d9'])
        ax.scatter(x, f, s=5, color='C0', alpha=.16, edgecolors='none', rasterized=True)
        ax.contour(xx, yy, zz, levels=thresholds, colors=['C0', 'C0'],
                   linestyles=['--', '-'], linewidths=[1.3, 1.7])
        # ax.axhline(f_upper, color='black', lw=1., ls=':')
        ax.set(xlabel=label, xlim=(0., .4), ylim=(1/4,2/3))
        ax.set_xticks([0., .1, .2, .3, .4])
        ax.text(.04, .05, case_label, transform=ax.transAxes, fontsize=12)
        report['panels'][name] = dict(albedo_quantiles=np.quantile(x, [.025,.16,.5,.84,.975]).tolist(),
                                     histogram_probability_thresholds=thresholds,
                                     correlation_with_f=float(np.corrcoef(x, f)[0,1]))
    axes[0].set_ylabel(r'Heat redistribution parameter $f$')
    handles = [Line2D([], [], color='C0', lw=1.7, label='68%'),
               Line2D([], [], color='C0', lw=1.3, ls='--', label='95%')]
    axes[1].legend(handles=handles, loc='lower right', frameon=False, fontsize=11)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.pdf', '.png'):
        path = output_prefix.with_suffix(suffix)
        fig.savefig(path, dpi=220)
        print(f'Saved {path}')
    report['f_quantiles'] = np.quantile(f, [.025,.16,.5,.84,.975]).tolist()
    output_prefix.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1445)
    parser.add_argument('--case', choices=('combined', 'f1500w'), default='combined',
                        help='Emission dataset (default: combined 16-bin LRS + F1500W)')
    parser.add_argument('--posterior', type=Path, default=None, help='Override the selected case posterior')
    parser.add_argument('--output', type=Path, default=None, help='Output HDF5 path (.h5 or .hdf5)')
    parser.add_argument('--extend', type=Path,
                        help='Extend this existing HDF5 in place to --samples total; also resumes an interrupted extension')
    parser.add_argument('--plot-only', action='store_true', help='Plot saved --output; no climate calculations')
    parser.add_argument('--figure-prefix', type=Path, default=None)
    args = parser.parse_args()
    args.posterior = args.posterior or (F1500W_POSTERIOR if args.case == 'f1500w' else DEFAULT_POSTERIOR)
    suffix = '_F1500W_only' if args.case == 'f1500w' else ''
    default_output = DERIVED_DIR / f'posterior_climate{suffix}_{args.samples}.h5'
    args.figure_prefix = args.figure_prefix or JOINT_FIGURE_DIR / f'figure_joint_albedos{suffix}'
    if args.samples < 1 or args.workers < 1:
        parser.error('Sample and worker counts must be positive')
    if args.extend and (args.plot_only or args.output):
        parser.error('--extend cannot be combined with --plot-only or --output')
    if args.plot_only:
        plot_results(args.output or default_output,
                     args.figure_prefix)
        return
    metadata = json.loads((args.posterior.parent / 'inputs.json').read_text())
    require_current_fit(metadata)
    posterior = np.atleast_2d(np.loadtxt(args.posterior))
    posterior = posterior[:, [metadata['param_names'].index(p) for p in PARAMETERS]]
    if not args.extend and args.samples > len(posterior):
        parser.error(f'Only {len(posterior)} posterior samples are available')
    output = args.extend or args.output or default_output
    if output.suffix not in ('.h5', '.hdf5'):
        parser.error('--output must end in .h5 or .hdf5')
    if not args.extend and (output.exists() or output.with_suffix('.json').exists()):
        parser.error(f'Output already exists: {output}; choose a different --output')
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.extend:
        provenance, results, pending = prepare_extension(output, args.posterior, metadata,
                                                        args.samples, args.seed)
    else:
        indices = np.random.default_rng(args.seed).choice(len(posterior), args.samples, replace=False)
        provenance = dict(posterior=str(args.posterior.resolve()), posterior_sha256=hashlib.sha256(
            args.posterior.read_bytes()).hexdigest(), seed=args.seed, param_names=PARAMETERS,
            requested_samples=args.samples, workers=args.workers, case=args.case, fit_tag=FIT_TAG,
            selected_posterior_indices=indices.tolist(), source_inputs=metadata,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            assumptions='Original grid mass/radius/stellar shape; no PICASO; one native thread per process.')
        results = []
        pending = [dict(sample_id=i, posterior_index=int(index), posterior_source='equal_weight',
                        parameters=posterior[index].tolist()) for i, index in enumerate(indices)]
    initial_count = len(results)
    print(f'Retaining {initial_count} completed draws; adding {len(pending)} for {args.samples} total.', flush=True)
    started = time.perf_counter()
    save_results(output, provenance, results)
    # Draws remain equally weighted, including repetitions. Evaluate each unique
    # parameter vector only once, reusing existing valid climate calculations.
    cached = {tuple(r['parameters']): r for r in results if r['converged'] and 'error' not in r}
    groups = {}
    for draw in pending:
        key = tuple(draw['parameters'])
        if key in cached:
            results.append(dict(cached[key], **draw, reused_climate=True))
        else:
            groups.setdefault(key, []).append(draw)
    print(f'{len(groups)} new climate calculations needed (repeated draws reuse results).', flush=True)
    if groups:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                                 initializer=initialize_worker) as pool:
            jobs = {pool.submit(compute_sample, draws[0]['posterior_index'], np.array(key)): draws
                    for key, draws in groups.items()}
            since_save = 0
            for job in as_completed(jobs):
                computed = job.result()
                draws = jobs[job]
                for i, draw in enumerate(draws):
                    results.append(dict(computed, **draw, reused_climate=i > 0))
                since_save += len(draws)
                print(f"{len(results)}/{args.samples}: row {computed['posterior_index']}, "
                      f"{computed['seconds']:.2f} s, converged={computed['converged']}, "
                      f"f={computed.get('f')}, Bond albedo={computed.get('bond_albedo')}"
                      + (f", ERROR={computed['error']}" if 'error' in computed else ''), flush=True)
                if since_save >= 25:
                    save_results(output, provenance, results)
                    since_save = 0
    elapsed = time.perf_counter() - started
    times = np.array([r['seconds'] for r in results])
    initialization = {r['worker_pid']: r['worker_initialization_seconds'] for r in results}
    success = sum(r['converged'] and 'error' not in r for r in results)
    summary = dict(samples=args.samples, workers=args.workers, successful=success,
                   retained_samples=initial_count, added_samples=len(pending),
                   new_climate_calculations=len(groups),
                   unique_parameter_vectors=len({tuple(r['parameters']) for r in results}),
                   wall_seconds=elapsed, mean_climate_seconds=float(times.mean()),
                   median_climate_seconds=float(np.median(times)),
                   min_climate_seconds=float(times.min()), max_climate_seconds=float(times.max()),
                   initialization_seconds_by_worker=initialization,
                   estimated_1000_compute_seconds=float(1000 * times.mean() / args.workers),
                   estimated_1000_wall_seconds=float(max(initialization.values())
                                                    + 1000 * times.mean() / args.workers),
                   note='Timing extrapolation assumes similar convergence costs and the selected worker count.')
    save_results(output, provenance, results, summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Saved {output}', flush=True)
    print(f'Saved {output.with_suffix(".csv")}', flush=True)
    if success != args.samples:
        raise SystemExit('Some samples failed; inspect the saved report before scaling up.')


if __name__ == '__main__':
    main()
