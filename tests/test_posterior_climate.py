"""Extension bookkeeping and CSV export; no climate solver is run."""
import hashlib
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from posterior_climate import PARAMETERS, FIT_TAG, prepare_extension, save_results


class ClimateExtensionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        folder = Path(self.temporary.name)
        self.posterior = folder/'casepost_equal_weights.dat'
        parameters = np.arange(27, dtype=float).reshape(3, 9)/100
        np.savetxt(self.posterior, np.column_stack((parameters, np.zeros(3))))
        np.savetxt(folder/'case.txt', np.column_stack(([.2, .3, .5], np.zeros(3), parameters)))
        self.output = folder/'climate_1000.h5'
        self.metadata = {'param_names': PARAMETERS}
        self.rows = [dict(posterior_index=i, parameters=parameters[i].tolist(), converged=True,
                          f=.6+i*.01, bond_albedo=.1, worker_pid=1,
                          seconds=1., worker_initialization_seconds=1.) for i in range(2)]
        self.provenance = dict(fit_tag=FIT_TAG, requested_samples=2, seed=1445,
                               posterior=str(self.posterior),
                               posterior_sha256=hashlib.sha256(self.posterior.read_bytes()).hexdigest(),
                               assumptions='Test')
        save_results(self.output, self.provenance, self.rows, {'successful': 2})

    def test_extension_preserves_rows_and_resumes_same_plan(self):
        provenance, retained, pending = prepare_extension(self.output, self.posterior,
                                                         self.metadata, 8, 42)
        self.assertEqual(len(retained), 2)
        self.assertEqual(len(pending), 6)
        self.assertEqual([r['parameters'] for r in retained], [r['parameters'] for r in self.rows])
        self.assertEqual(len({r['sample_id'] for r in retained+pending}), 8)
        added = dict(pending[0], converged=True, f=.65, bond_albedo=.2)
        save_results(self.output, provenance, retained+[added])
        _, saved, remaining = prepare_extension(self.output, self.posterior, self.metadata, 8, 999)
        self.assertEqual(len(saved), 3)
        self.assertEqual(remaining, pending[1:])

    def test_csv_matches_hdf5_row_order(self):
        data = np.loadtxt(self.output.with_suffix('.csv'), delimiter=',', skiprows=1)
        self.assertEqual(self.output.with_suffix('.csv').read_text().splitlines()[0],
                         'f_term,bond_albedo,surface_albedo')
        with h5py.File(self.output) as f:
            np.testing.assert_array_equal(data, np.column_stack(
                [f[name][:] for name in ('f', 'bond_albedo', 'surface_albedo')]))

    def test_changed_posterior_is_rejected(self):
        self.posterior.write_text('Changed source')
        with self.assertRaisesRegex(ValueError, 'do not match'):
            prepare_extension(self.output, self.posterior, self.metadata, 8, 42)


if __name__ == '__main__':
    unittest.main()
