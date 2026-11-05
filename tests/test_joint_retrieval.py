"""Validate the final-fit prior, asymmetric photometry, and retrieval predictions."""
from pathlib import Path
import unittest

import numpy as np

import retrieval_run as legacy
from joint_system_prior import JointSystemPrior, RESULTS_DIR, FIT_TAG, require_current_fit
from utils import observation_loglike, observation_yerr, make_joint_photometry
from retrieval_run_joint import JointRetrieval, CASE_NAMES, case_names


class JointPriorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prior = JointSystemPrior.from_chain()

    def test_chain_selection_and_measurement(self):
        meta = self.prior.metadata()
        self.assertEqual(meta['chain_draws'], 10000)
        self.assertEqual(meta['retained_draws'], 9791)
        self.assertAlmostEqual(meta['below_grid_fraction'], .0175)
        self.assertAlmostEqual(meta['above_grid_fraction'], .0034)
        self.assertIn('12758GP_RVGP', str(self.prior.source))
        self.assertAlmostEqual(self.prior.eclipse_median*1.e6, 157.4729237374)
        self.assertAlmostEqual(self.prior.eclipse_err_lower*1.e6, 17.313102253804)
        self.assertAlmostEqual(self.prior.eclipse_err_upper*1.e6, 17.816033503901)

    def test_asymmetric_measurement_likelihood(self):
        data = make_joint_photometry(self.prior)
        center = data['fpfs']
        self.assertEqual(observation_loglike(data, center), 0.)
        for side, error in [(-1, data['err_lower']), (1, data['err_upper'])]:
            self.assertAlmostEqual(observation_loglike(data, center+side*error), -.5)
            self.assertAlmostEqual(observation_loglike(data, center+side*2*error), -2.)
        self.assertNotAlmostEqual(observation_loglike(data, center+data['err_lower']), -.5)
        np.testing.assert_array_equal(observation_yerr(data), [data['err_lower'], data['err_upper']])
        symmetric = dict(fpfs=np.array([.1, .2]), err=np.array([.01, .03]))
        prediction = np.array([.12, .17])
        self.assertAlmostEqual(observation_loglike(symmetric, prediction), -2.5)

    def test_current_source_guard(self):
        metadata = dict(fit_tag=FIT_TAG, system_prior=self.prior.metadata())
        require_current_fit(metadata)
        with self.assertRaises(ValueError):
            require_current_fit(dict(metadata, fit_tag='different_fit'))

    def test_prior_reproduces_retained_distribution(self):
        report = self.prior.validate()
        self.assertTrue(report['passes_preliminary_checks'], report)
        # Also validate the physically derived Teff--Teq relation, which is
        # highly correlated and must not be replaced by independent priors.
        rng = np.random.default_rng(1729)
        generated = self.prior.transform(rng.random((100000, 3)))
        reference = self.prior.retained
        ref_teff = reference[:, 0]*np.sqrt(2*reference[:, 1])
        corr_ref = np.corrcoef(reference[:, 0], ref_teff)[0, 1]
        corr_new = np.corrcoef(generated[:, 0], generated[:, 1])[0, 1]
        self.assertLess(abs(corr_new-corr_ref), .003)
        self.assertGreater(corr_new, .98)
        self.assertTrue(np.all((generated[:, 0] >= 385.) & (generated[:, 0] <= 477.)))
        self.assertEqual(np.count_nonzero(np.isin(generated[:, 0], [385., 477.])), 0)

    def test_endpoint_safety_and_invalid_unit_coordinates(self):
        for unit in ([0., 0., 0.], [1., 1., 1.], [0., 1., 0.], [1., 0., 1.]):
            q, t, r = self.prior.transform(unit)
            self.assertTrue(np.all(np.isfinite([q, t, r])))
            self.assertTrue(385. <= q <= 477.)
            self.assertTrue(2000. <= t <= 4000.)
            self.assertTrue(0. < r < 1.)
        for unit in ([.5, .5], [0., -.1, .5], [1.1, 0., .5], [0., np.nan, .5]):
            with self.assertRaises(ValueError):
                self.prior.transform(unit)


class JointForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retrieval = JointRetrieval()

    def test_six_cases_and_physical_predictions(self):
        self.assertEqual(set(self.retrieval.cases), set(CASE_NAMES))
        before = {k: list(v) for k, v in legacy.SYSTEM_PRIORS.items()}
        old_data = legacy.RETRIEVAL_CASES['atm_F1500W_obs']['data_dict']['fpfs'].copy()
        checks = self.retrieval.check_forward_models()
        self.assertTrue(all(c['finite_at_unit_endpoints_and_center'] for c in checks.values()))
        self.assertEqual(before, legacy.SYSTEM_PRIORS)
        np.testing.assert_array_equal(old_data, legacy.RETRIEVAL_CASES['atm_F1500W_obs']['data_dict']['fpfs'])
        np.testing.assert_allclose(old_data, [148.e-6])
        for kind in ('rock', 'atm'):
            single = self.retrieval.cases[kind+'_F1500W_obs']
            combined = self.retrieval.cases[kind+'_16_F1500W_obs']
            parameters = single['prior'](np.full(len(single['param_names']), .5))
            mask = combined['data_dict']['photometric']
            np.testing.assert_allclose(combined['predict'](parameters)[mask], single['predict'](parameters))
            np.testing.assert_allclose(single['data_dict']['fpfs'], [self.retrieval.system.eclipse_median])
            np.testing.assert_allclose(single['data_dict']['err_lower'], [self.retrieval.system.eclipse_err_lower])
            np.testing.assert_allclose(single['data_dict']['err_upper'], [self.retrieval.system.eclipse_err_upper])
            mask = ~combined['data_dict']['photometric']
            np.testing.assert_array_equal(combined['data_dict']['err_lower'][mask],
                                          combined['data_dict']['err_upper'][mask])


class LegacyRegressionTests(unittest.TestCase):
    def test_saved_legacy_likelihoods(self):
        paths = sorted(Path('pymultinest').glob('*/*post_equal_weights.dat'))
        if not paths:
            self.skipTest('Historical MultiNest posterior files are not installed')
        legacy.initialize_retrieval()
        for path in paths:
            name = path.parent.name
            self.assertIn(name, legacy.RETRIEVAL_CASES)
            case = legacy.RETRIEVAL_CASES[name]
            data = np.atleast_2d(np.loadtxt(path))
            for i in np.linspace(0, len(data)-1, 5, dtype=int):
                with self.subTest(case=name, row=int(i)):
                    self.assertAlmostEqual(case['loglike'](data[i, :-1]), data[i, -1], places=9)
        self.assertEqual(legacy.SYSTEM_PRIORS, {
            'Teq': [431., 23., 385., 477.],
            'Teff': [3340., 150., 3040., 3640.],
            'R_planet_star': [.0454, .0012, .0430, .0478]})


class EightBinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eight = JointRetrieval(lrs_bins=8)
        cls.sixteen = JointRetrieval()

    def test_data_selection_and_shared_priors(self):
        self.assertEqual(set(self.eight.cases), set(case_names(8)))
        self.assertEqual(len(self.eight.cases['atm_8']['data_dict']['fpfs']), 8)
        self.assertEqual(len(self.eight.cases['atm_8_F1500W_obs']['data_dict']['fpfs']), 9)
        for kind in ('rock', 'atm'):
            c8 = self.eight.cases[kind+'_8']
            c16 = self.sixteen.cases[kind+'_16']
            unit = np.full(len(c8['param_names']), .5)
            np.testing.assert_array_equal(c8['prior'](unit), c16['prior'](unit))
            x = c8['prior'](unit)
            combined = self.eight.cases[kind+'_8_F1500W_obs']
            np.testing.assert_allclose(combined['predict'](x)[:8], c8['predict'](x))
            np.testing.assert_allclose(combined['predict'](x)[-1:],
                                      self.sixteen.cases[kind+'_F1500W_obs']['predict'](x))

    def test_eight_bin_forward_model_edges(self):
        checks = self.eight.check_forward_models()
        self.assertTrue(all(c['finite_at_unit_endpoints_and_center'] for c in checks.values()))

    def test_saved_final_likelihoods(self):
        root = RESULTS_DIR
        if not root.exists():
            self.skipTest('Joint retrieval outputs are not installed')
        for name, case in self.sixteen.cases.items():
            if not (root/name/(name+'post_equal_weights.dat')).exists():
                continue
            data = np.atleast_2d(np.loadtxt(root/name/(name+'post_equal_weights.dat')))
            for i in np.linspace(0, len(data)-1, 5, dtype=int):
                with self.subTest(case=name, row=int(i)):
                    self.assertAlmostEqual(case['loglike'](data[i, :-1]), data[i, -1], places=9)


if __name__ == '__main__':
    unittest.main()
