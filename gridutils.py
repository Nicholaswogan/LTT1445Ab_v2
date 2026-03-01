from threadpoolctl import threadpool_limits
_ = threadpool_limits(limits=1) # set threads to 1

from mpi4py import MPI
import dill as pickle
import os
import numpy as np
from tqdm import tqdm
from scipy import interpolate
import h5py
import traceback

def get_inputs(gridvals):
    tmp = np.meshgrid(*gridvals, indexing='ij')
    inputs = np.empty((tmp[0].size, len(tmp)))
    for i, t in enumerate(tmp):
        inputs[:, i] = t.flatten()
    return inputs

def initialize_hdf5(filename):
    """Initialize the HDF5 file."""
    with h5py.File(filename, 'w') as f:
        pass

def check_hdf5(filename, gridvals, gridnames, common):
    with h5py.File(filename, 'r') as f:
        gridshape = tuple(len(v) for v in gridvals)
        npoints = int(np.prod(gridshape))

        if 'gridvals' not in f:
            raise Exception('The following file lacks the group `gridvals`: '+filename)
        for i,gridval in enumerate(gridvals):
            key = '%i'%i
            if not np.allclose(f['gridvals'][key][:], gridval):
                raise Exception('Miss match between the input `gridvals`, and the `gridvals` in '+filename)
            
        if 'gridnames' not in f:
            raise Exception('The following file lacks `gridnames`: '+filename)

        gridnames_array = np.array(gridnames)
        if not np.all(f['gridnames'][:].astype(gridnames_array.dtype) == gridnames_array):
            raise Exception('Miss match between the input `gridnames`, and the `gridnames` in '+filename)
            
        if 'common' not in f:
            raise Exception('The following file lacks the group `common`: '+filename)
        for key, val in common.items():
            if not np.allclose(f['common'][key][:], val):
                raise Exception('Miss match between the input `common`, and the `common` in '+filename)

        # Validate restart-critical dataset shapes against the current grid definition.
        if 'inputs' in f:
            if f['inputs'].shape[0] != npoints:
                raise Exception(
                    'Miss match between current grid size and existing `inputs` first dimension in '
                    + filename
                )
            if f['inputs'].shape[1] != len(gridvals):
                raise Exception(
                    'Miss match between number of grid dimensions and existing `inputs` second dimension in '
                    + filename
                )

        if 'completed' in f:
            if f['completed'].shape != (npoints,):
                raise Exception(
                    'Miss match between current grid size and existing `completed` shape in ' + filename
                )

        if 'results' in f:
            for key in f['results'].keys():
                ds = f['results'][key]
                if ds.shape[:len(gridshape)] != gridshape:
                    raise Exception(
                        "Miss match between current grid shape and existing `results/%s` leading dimensions in %s"
                        % (key, filename)
                    )

def ensure_hdf5_layout(
    f,
    x,
    res,
    grid_shape,
    gridvals,
    gridnames,
    common,
    compression=None,
    compression_opts=None,
    shuffle=False,
):
    # Save the gridvals if that has not happened
    if 'gridvals' not in f:
        f.create_group('gridvals')
        for i,gridval in enumerate(gridvals):
            key = '%i'%i
            f['gridvals'].create_dataset(key, shape=(len(gridval),), dtype=gridval.dtype)
            f['gridvals'][key][:] = gridval

    if 'gridnames' not in f:
        gridnames_array = np.array(gridnames)
        f.create_dataset('gridnames', shape=(len(gridnames_array),), dtype=h5py.string_dtype())
        f['gridnames'][:] = gridnames_array

    if 'common' not in f:
        f.create_group('common')
        for key, val in common.items():
            f['common'].create_dataset(key, shape=val.shape, dtype=val.dtype)
            f['common'][key][:] = val

    # Save input parameters
    if 'inputs' not in f:
        f.create_dataset('inputs', shape=(np.prod(grid_shape),len(x),), dtype=x.dtype)
        f['inputs'][:] = np.nan
    elif f['inputs'].shape[1] != len(x):
        raise ValueError(
            "Result layout mismatch: input vector length changed. "
            f"Expected {f['inputs'].shape[1]}, got {len(x)}."
        )

    # Create 'results' group if it doesn't exist
    if 'results' not in f:
        f.create_group('results')

    existing_keys = set(f['results'].keys())
    current_keys = set(res.keys())

    # If results already exist, enforce exact key matches.
    if len(existing_keys) > 0 and current_keys != existing_keys:
        missing = sorted(existing_keys - current_keys)
        extra = sorted(current_keys - existing_keys)
        raise ValueError(
            "Result layout mismatch: output keys changed. "
            f"Missing keys: {missing if missing else '[]'}, "
            f"extra keys: {extra if extra else '[]'}."
        )

    # For each result key, create dataset on first successful result, then enforce shape/dtype.
    for key, val in res.items():
        val_arr = np.asarray(val)
        data_shape = grid_shape + val_arr.shape  # accommodate vector outputs
        if key not in f['results']:
            create_kwargs = {}
            if compression is not None:
                create_kwargs['compression'] = compression
                if compression_opts is not None:
                    create_kwargs['compression_opts'] = compression_opts
                create_kwargs['shuffle'] = shuffle
                # Compression requires chunked storage; let h5py choose a chunk shape.
                create_kwargs['chunks'] = True
            f['results'].create_dataset(key, shape=data_shape, dtype=val_arr.dtype, **create_kwargs)
        else:
            ds = f['results'][key]
            if ds.shape != data_shape:
                raise ValueError(
                    "Result layout mismatch: output shape changed for key "
                    f"'{key}'. Expected per-grid value shape {ds.shape[len(grid_shape):]}, "
                    f"got {val_arr.shape}."
                )
            if ds.dtype != val_arr.dtype:
                raise ValueError(
                    "Result layout mismatch: output dtype changed for key "
                    f"'{key}'. Expected {ds.dtype}, got {val_arr.dtype}."
                )

    if 'completed' not in f:
        f.create_dataset('completed', shape=(np.prod(grid_shape),), dtype='bool')
        f['completed'][:] = np.zeros(np.prod(grid_shape),dtype='bool')

def write_result_hdf5(f, index, x, res, grid_shape):
    unraveled_idx = np.unravel_index(index, grid_shape)
    f['inputs'][index] = x
    for key, val in res.items():
        f['results'][key][unraveled_idx] = np.asarray(val)
    f['completed'][index] = True

def save_result_hdf5(
    filename,
    index,
    x,
    res,
    grid_shape,
    gridvals,
    gridnames,
    common,
    compression=None,
    compression_opts=None,
    shuffle=False,
):
    """Save a single result to the HDF5 file."""
    with h5py.File(filename, 'a') as f:
        ensure_hdf5_layout(
            f,
            x,
            res,
            grid_shape,
            gridvals,
            gridnames,
            common,
            compression=compression,
            compression_opts=compression_opts,
            shuffle=shuffle,
        )
        write_result_hdf5(f, index, x, res, grid_shape)

def load_completed_mask(filename):
    if os.path.isfile(filename):
        with h5py.File(filename, 'r') as f:
            if 'completed' not in f:
                return np.array([], dtype=int)
            return np.where(f['completed'])[0]
    return np.array([], dtype=int)

def assign_job(comm, rank, serialized_model, job_iter, inputs):
    try:
        job_index = next(job_iter)
        comm.send((serialized_model, job_index, inputs[job_index]), dest=rank, tag=1)
        return True
    except StopIteration:
        comm.send(None, dest=rank, tag=0)
        return False

def master(
    model_func,
    gridvals,
    gridnames,
    filename,
    progress_filename,
    common,
    flush_every_n=1,
    compression=None,
    compression_opts=None,
    shuffle=False,
):

    if len(gridvals) != len(gridnames):
        raise ValueError('`gridvals` and `gridnames` have incompatable shapes.')
    if flush_every_n < 1:
        raise ValueError('`flush_every_n` must be >= 1.')

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    inputs = get_inputs(gridvals)
    gridshape = tuple(len(v) for v in gridvals)

    serialized_model = pickle.dumps(model_func)

    # Initialize HDF5 if needed
    if not os.path.exists(filename):
        print("Initializing HDF5 output...")
        initialize_hdf5(filename)
    else:
        check_hdf5(filename, gridvals, gridnames, common)

    completed_inds = load_completed_mask(filename)
    n_total = inputs.shape[0]
    n_completed = len(completed_inds)
    if n_completed > 0:
        print(f'Calculations completed/total: {n_completed}/{n_total}.')
        if n_completed == n_total:
            print('All calculations completed.')
        else:
            print('Restarting calculations...')

    # Get inputs that have not yet been computed
    completed_set = set(completed_inds.tolist())
    job_indices = [i for i in range(len(inputs)) if i not in completed_set]
    job_iter = iter(job_indices)

    # Handle single-rank runs without worker processes.
    if size == 1:
        with open(progress_filename, 'w') as log_file, h5py.File(filename, 'a') as h5f:
            pbar = tqdm(total=n_total, initial=n_completed, file=log_file, dynamic_ncols=True)
            pending_flush = 0
            for index in job_indices:
                x = inputs[index]
                try:
                    res = model_func(x)
                except Exception as e:
                    # Intentionally catch Exception (not BaseException) so Ctrl-C/KeyboardInterrupt
                    # can still stop the run immediately.
                    print(f"Rank 0 failed on job {index}: {repr(e)}", file=log_file, flush=True)
                    print(traceback.format_exc(), file=log_file, flush=True)
                    continue
                ensure_hdf5_layout(
                    h5f,
                    x,
                    res,
                    gridshape,
                    gridvals,
                    gridnames,
                    common,
                    compression=compression,
                    compression_opts=compression_opts,
                    shuffle=shuffle,
                )
                write_result_hdf5(h5f, index, x, res, gridshape)
                pending_flush += 1
                if pending_flush >= flush_every_n:
                    h5f.flush()
                    pending_flush = 0
                pbar.update(1)
                log_file.flush()
            if pending_flush > 0:
                h5f.flush()
            pbar.close()
        return
    
    # Open progress log file for writing
    with open(progress_filename, 'w') as log_file, h5py.File(filename, 'a') as h5f:
        pbar = tqdm(total=n_total, initial=n_completed, file=log_file, dynamic_ncols=True)
        status = MPI.Status()
        pending_flush = 0

        # Assign initial workers
        active_workers = 0
        for rank in range(1, size):
            if assign_job(comm, rank, serialized_model, job_iter, inputs):
                active_workers += 1

        # Continue until all workers are terminated
        while active_workers > 0:

            # Get result form worker
            msg = comm.recv(source=MPI.ANY_SOURCE, tag=2, status=status)
            worker_rank = status.Get_source()

            if msg['status'] == 'ok':
                index, x, res = msg['index'], msg['x'], msg['res']

                # Save the result
                ensure_hdf5_layout(
                    h5f,
                    x,
                    res,
                    gridshape,
                    gridvals,
                    gridnames,
                    common,
                    compression=compression,
                    compression_opts=compression_opts,
                    shuffle=shuffle,
                )
                write_result_hdf5(h5f, index, x, res, gridshape)
                pending_flush += 1
                if pending_flush >= flush_every_n:
                    h5f.flush()
                    pending_flush = 0
                
                pbar.update(1)
                log_file.flush()
            elif msg['status'] == 'error':
                index = msg['index']
                err = msg['error']
                tb = msg['traceback']
                print(f"Worker {worker_rank} failed on job {index}: {err}", file=log_file, flush=True)
                print(tb, file=log_file, flush=True)
            else:
                raise RuntimeError(f"Unknown worker message status: {msg['status']}")

            # Assign a new job to the worker.
            if not assign_job(comm, worker_rank, serialized_model, job_iter, inputs):
                active_workers -= 1

        if pending_flush > 0:
            h5f.flush()
        pbar.close()

def worker():
    comm = MPI.COMM_WORLD
    status = MPI.Status()

    while True:
        # Get inputs from master process
        data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag() == 0:
            break # Shutdown signal

        # Call the function on the inputs
        serialized_model, index, x = data
        model_func = pickle.loads(serialized_model)
        try:
            res = model_func(x)
            msg = {'status': 'ok', 'index': index, 'x': x, 'res': res}
        except Exception as e:
            # Intentionally catch Exception (not BaseException) so Ctrl-C/KeyboardInterrupt
            # can still stop the run immediately.
            msg = {
                'status': 'error',
                'index': index,
                'x': x,
                'error': repr(e),
                'traceback': traceback.format_exc(),
            }

        # Send the result or error to the master process
        comm.send(msg, dest=0, tag=2)

def make_grid(
    model_func,
    gridvals,
    gridnames,
    filename,
    progress_filename,
    common=None,
    flush_every_n=1,
    compression=None,
    compression_opts=None,
    shuffle=False,
):
    """
    Run a parallel grid computation using MPI, saving results to an HDF5 file.

    This function distributes computations across available MPI ranks. The master
    process assigns jobs to worker processes, collects results, and writes them to
    an HDF5 file. A separate progress log file tracks computation progress.

    Parameters
    ----------
    model_func : callable
        A function that takes a 1D numpy array of input parameters and returns
        a dictionary of results, where each key corresponds to a quantity (numpy array)
        to be saved.
    
    gridvals : tuple of 1D numpy arrays
        Defines the parameter grid. Each array in the tuple represents the discrete 
        values for one dimension of the parameter space.

    gridnames : 1D numpy array
        Names of the variables in the grid.

    filename : str
        Path to the HDF5 file where computed results will be stored. The file will contain
        groups for each grid point index, each with datasets for the input parameters 
        and the model output.

    progress_filename : str
        Path to the text file where progress updates (from the master process) will be logged.

    common: dict of numpy arrays
        Dictionary of arrays that will be saved in the HDF5 file. Sometimes saved results will
        share a common axis that might want to be saved alongside the data without repetition.

    flush_every_n: int, optional
        Flush HDF5 buffers to disk after this many successful writes. Use 1 for maximum
        restart durability, larger values for better write performance.

    compression : str or None, optional
        Compression filter for result datasets (e.g. 'gzip', 'lzf'). If None, no compression.

    compression_opts : int or None, optional
        Compression settings for the selected compression filter (e.g. gzip level).

    shuffle : bool, optional
        Enable HDF5 shuffle filter when compression is enabled.

    Notes
    -----
    - This function must be run with an MPI launcher (e.g., `mpiexec -n N python script.py`).
    - The results are saved incrementally, so the computation can be resumed if interrupted.
    - Only rank 0 (master) writes to the output files.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if common is None:
        common = {}

    if rank == 0:
        # Master process
        master(
            model_func,
            gridvals,
            gridnames,
            filename,
            progress_filename,
            common,
            flush_every_n=flush_every_n,
            compression=compression,
            compression_opts=compression_opts,
            shuffle=shuffle,
        )
    else:
        # Worker process
        worker()

def resave_with_new_grid(
    old_filename,
    new_gridvals,
    new_gridnames,
    new_filename,
    compression=None,
    compression_opts=None,
    shuffle=False,
):
    """
    Create a new restartable HDF5 grid file from a different grid definition.

    Grid dimensions must match exactly by name and order; only grid values may change.
    Completed points from `old_filename` are copied into `new_filename` where
    coordinates overlap exactly (within a tight floating tolerance).

    Parameters
    ----------
    compression : str or None, optional
        Compression filter for result datasets in the new file (e.g. 'gzip', 'lzf').
        If None, no compression is applied.
    compression_opts : int or None, optional
        Compression settings for the selected compression filter (e.g. gzip level).
    shuffle : bool, optional
        Enable HDF5 shuffle filter when compression is enabled.
    """
    if os.path.exists(new_filename):
        raise FileExistsError(f'Output file already exists: {new_filename}')

    new_gridvals = tuple(np.asarray(v) for v in new_gridvals)
    if len(new_gridvals) != len(new_gridnames):
        raise ValueError('`new_gridvals` and `new_gridnames` must have the same length.')

    def _normalize_names(values):
        out = []
        for v in values:
            if isinstance(v, bytes):
                out.append(v.decode('utf-8'))
            else:
                out.append(str(v))
        return out

    with h5py.File(old_filename, 'r') as oldf:
        if 'gridvals' not in oldf or 'gridnames' not in oldf or 'results' not in oldf:
            raise ValueError('Old file is missing one of required groups/datasets: gridvals, gridnames, results.')

        old_gridvals = []
        for i in range(len(oldf['gridvals'])):
            old_gridvals.append(oldf['gridvals'][f'{i}'][:])
        old_gridvals = tuple(old_gridvals)

        old_gridnames = _normalize_names(oldf['gridnames'][:])
        new_gridnames_norm = _normalize_names(new_gridnames)
        if old_gridnames != new_gridnames_norm:
            raise ValueError(
                'Grid-name mismatch. `new_gridnames` must match old grid names exactly (same order). '
                f'Old: {old_gridnames}, New: {new_gridnames_norm}.'
            )

        old_gridshape = tuple(len(v) for v in old_gridvals)
        new_gridshape = tuple(len(v) for v in new_gridvals)
        ndim = len(old_gridshape)
        if len(new_gridshape) != ndim:
            raise ValueError(
                f'Grid dimensionality mismatch. Old dims: {ndim}, new dims: {len(new_gridshape)}.'
            )

        # Map each old axis index into new axis index (or -1 if no overlap).
        axis_index_maps = []
        for dim in range(ndim):
            old_axis = np.asarray(old_gridvals[dim])
            new_axis = np.asarray(new_gridvals[dim])
            mapped = np.full(old_axis.shape, -1, dtype=int)
            for i, old_val in enumerate(old_axis):
                matches = np.where(np.isclose(new_axis, old_val, rtol=1e-12, atol=1e-14))[0]
                if matches.size > 1:
                    raise ValueError(
                        f'Axis {dim} maps old value {old_val} to multiple new indices: '
                        f'{matches.tolist()}. Axis values must be unique.'
                    )
                if matches.size == 1:
                    mapped[i] = int(matches[0])
            axis_index_maps.append(mapped)

        common_data = {}
        if 'common' in oldf:
            for key in oldf['common'].keys():
                common_data[key] = oldf['common'][key][:]

        result_specs = []
        for key in oldf['results'].keys():
            ds = oldf['results'][key]
            if ds.shape[:ndim] != old_gridshape:
                raise ValueError(
                    f'Old results dataset shape mismatch for key {key}: '
                    f'expected leading shape {old_gridshape}, got {ds.shape[:ndim]}.'
                )
            result_specs.append((key, ds.shape[ndim:], ds.dtype))

        if 'completed' in oldf:
            completed_inds = np.where(oldf['completed'][:])[0]
        else:
            completed_inds = np.array([], dtype=int)

        has_old_inputs = 'inputs' in oldf
        old_inputs_dtype = oldf['inputs'].dtype if has_old_inputs else np.float64
        n_new_points = int(np.prod(new_gridshape))

        with h5py.File(new_filename, 'w') as newf:
            newf.create_group('gridvals')
            for i, gv in enumerate(new_gridvals):
                newf['gridvals'].create_dataset(f'{i}', data=gv, shape=(len(gv),), dtype=gv.dtype)

            gridnames_arr = np.array(new_gridnames_norm, dtype=object)
            newf.create_dataset('gridnames', data=gridnames_arr, dtype=h5py.string_dtype(encoding='utf-8'))

            newf.create_group('common')
            for key, val in common_data.items():
                newf['common'].create_dataset(key, data=val, shape=val.shape, dtype=val.dtype)

            newf.create_dataset('inputs', shape=(n_new_points, ndim), dtype=old_inputs_dtype)
            if np.issubdtype(old_inputs_dtype, np.floating):
                newf['inputs'][:] = np.nan
            else:
                newf['inputs'][:] = 0

            newf.create_group('results')
            for key, tail_shape, dtype in result_specs:
                create_kwargs = {}
                if compression is not None:
                    create_kwargs['compression'] = compression
                    if compression_opts is not None:
                        create_kwargs['compression_opts'] = compression_opts
                    create_kwargs['shuffle'] = shuffle
                    create_kwargs['chunks'] = True
                newf['results'].create_dataset(
                    key,
                    shape=new_gridshape + tail_shape,
                    dtype=dtype,
                    **create_kwargs,
                )

            newf.create_dataset('completed', shape=(n_new_points,), dtype='bool')
            newf['completed'][:] = False

            copied_old_points = 0
            skipped_old_points = 0
            filled_new_points = 0
            collisions = 0
            for old_index in completed_inds:
                old_multi = np.unravel_index(int(old_index), old_gridshape)
                mapped_inds = [axis_index_maps[d][old_multi[d]] for d in range(ndim)]
                if any(ind < 0 for ind in mapped_inds):
                    skipped_old_points += 1
                    continue
                new_multi = tuple(mapped_inds)
                new_lin = np.ravel_multi_index(new_multi, new_gridshape)
                if newf['completed'][new_lin]:
                    collisions += 1
                    raise ValueError(
                        f'Collision while remapping old completed point {int(old_index)} to '
                        f'new index {new_lin}. Multiple old points map to the same new point.'
                    )

                if has_old_inputs:
                    x = oldf['inputs'][old_index]
                else:
                    x = np.array([old_gridvals[d][old_multi[d]] for d in range(ndim)])
                newf['inputs'][new_lin] = x

                for key, _, _ in result_specs:
                    newf['results'][key][new_multi] = oldf['results'][key][old_multi]
                newf['completed'][new_lin] = True
                copied_old_points += 1
                filled_new_points += 1

    return {
        'copied_old_points': int(copied_old_points),
        'skipped_old_points': int(skipped_old_points),
        'filled_new_points': int(filled_new_points),
        'collisions': int(collisions),
        'added_names': [],
        'dropped_names': [],
        'shared_names': old_gridnames,
    }

class GridInterpolator():
    """
    A class for interpolating data saved from an HDF5 grid of simulation outputs.

    This class reads an HDF5 file containing simulation outputs stored on a parameter grid.
    It provides a method to generate interpolators that can predict values or arrays of 
    results at arbitrary points within the grid using `scipy.interpolate.RegularGridInterpolator`.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file containing the simulation results.

    gridvals : tuple of np.ndarray
        The parameter grid values, used to define the interpolation space.

    Attributes
    ----------
    gridvals : tuple of np.ndarray
        The parameter values for each grid dimension.

    gridshape : tuple of int
        The shape of the parameter grid, inferred from the lengths of `gridvals`.

    data : dict of np.ndarray
        The results from the HDF5 file.
    """

    def __init__(self, filename):
        """
        Initialize the GridInterpolator by loading data from an HDF5 file.

        Parameters
        ----------
        filename : str
            Path to the HDF5 file containing the simulation results.

        gridvals : tuple of np.ndarray
            The parameter grid values, used to define the interpolation space.
        """

        with h5py.File(filename, 'r') as f:
            self.data = {}
            for key in f['results'].keys():
                self.data[key] = f['results'][key][...]
            gridvals = []
            for i in range(len(f['gridvals'])):
                key = '%i'%i
                gridvals.append(f['gridvals'][key][:])
            gridvals = tuple(gridvals)
            common = {}
            for key in f['common'].keys():
                common[key] = f['common'][key][:]
            
        self.common = common
        self.gridvals = gridvals
        self.gridshape = tuple(len(a) for a in gridvals)

        self.min_gridvals = np.array([np.min(a) for a in self.gridvals])
        self.max_gridvals = np.array([np.max(a) for a in self.gridvals])

    def make_interpolator(self, key, method='linear', linthresh=1.0, logspace=None, bounds_mode='clip'):
        """
        Create an interpolator for a grid parameter.

        Parameters
        ----------
        key : str
            The key in the `self.data` dictionary for which to create the interpolator.

        logspace : bool, optional
            If True, interpolation is performed in log10-space. This is useful for 
            quantities that span many orders of magnitude.

        bounds_mode : {'clip', 'error'}, optional
            Behavior for out-of-bounds interpolation coordinates.
            - 'clip': clip each coordinate to nearest grid bound (default).
            - 'error': raise a ValueError describing which dimensions are out of range.

        Returns
        -------
        interp : function
            Interpolator function, which is called with a tuple of arguments: `interp((2,3,4))`.
        """
    
        data = self.data[key]

        # for backwards compatibility
        if logspace == True:
            method = 'log'

        if method == 'linear':
            transform = linear_transform
            untransform = linear_inverse
        elif method == 'log':
            transform = log_transform
            untransform = log_inverse
        elif method == 'symlog':
            transform = symlog_transform_func(linthresh)
            untransform = symlog_inverse_func(linthresh)
        else:
            raise ValueError('`method` can not be: '+method)
        if bounds_mode not in ('clip', 'error'):
            raise ValueError("`bounds_mode` must be either 'clip' or 'error'.")

        # Apply transformation
        data = transform(data)

        # Create the interpolator.
        rgi = interpolate.RegularGridInterpolator(
            self.gridvals,
            data,
            bounds_error=True,
            fill_value=None,
        )
        min_vals = self.min_gridvals
        max_vals = self.max_gridvals

        def interp(vals):
            if bounds_mode == 'clip':
                vals = np.clip(vals, a_min=min_vals, a_max=max_vals)
            out = rgi(vals)
            out = untransform(out)
            return out[0]

        return interp
    
# Linear transform
def linear_transform(x):
    return x
def linear_inverse(z):
    return z

# Log transform
def log_transform(x):
    return np.log10(np.maximum(x, 2e-38))
def log_inverse(z):
    return 10.0**z

# Symmetric log
def symlog_transform_func(linthresh):
    """
    Symmetric log transform with a linear region around zero.
    linthresh: values with |y| <= linthresh are mapped linearly.
    """
    def func(y):
        y = np.array(y, dtype=float)
        sign = np.sign(y)
        mask = np.abs(y) > linthresh
        out = np.zeros_like(y)

        # Linear region
        out[~mask] = y[~mask] / linthresh

        # Logarithmic region
        out[mask] = sign[mask] * (np.log10(np.abs(y[mask]) / linthresh) + 1.0)

        return out
    return func
def symlog_inverse_func(linthresh):
    """
    Inverse of the symlog transform.
    """
    def func(z):
        z = np.array(z, dtype=float)
        sign = np.sign(z)
        mask = np.abs(z) > 1.0
        out = np.zeros_like(z)

        # Linear region
        out[~mask] = z[~mask] * linthresh

        # Logarithmic region
        out[mask] = sign[mask] * 10.0**(np.abs(z[mask]) - 1.0) * linthresh

        return out
    return func
