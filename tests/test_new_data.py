import unittest
import numpy as np
import utils
import retrieval_run as retrieval


class NewDataTests(unittest.TestCase):
    def test_observed_datasets(self):
        single = utils.make_F1500W_data_real()
        combined = utils.make_lrs_and_F1500W_data_real('data/LTT1445Ab_Sparta_16.txt')
        np.testing.assert_allclose(single['fpfs'], [148e-6])
        np.testing.assert_allclose(single['err'], [11e-6])
        self.assertEqual(len(combined['wv']), len(utils.make_lrs_data('data/LTT1445Ab_Sparta_16.txt')['wv'])+1)
        self.assertEqual(np.count_nonzero(combined['photometric']), 1)
        for model in ('rock', 'atm', 'atm_hot'):
            for dataset in ('16_F1500W_obs', 'F1500W_obs'):
                case = retrieval.RETRIEVAL_CASES[f'{model}_{dataset}']
                params = case['prior'](np.full(len(case['param_names']), .5))
                self.assertTrue(np.all(np.isfinite(params)))
        self.assertTrue(np.all(np.diff(combined['wv']) > 0))

    def test_photon_ratio_uses_stellar_weight_and_bin_width(self):
        edges = np.array([1., 2., 4.])
        fs = np.array([2., 5.])
        ratio = np.array([.01, .04])
        # Flat throughput: exact integral of lambda over each bin.
        weights = np.diff(edges**2)/2
        expected = np.sum(weights*fs*ratio)/np.sum(weights*fs)
        self.assertAlmostEqual(utils.photon_weighted_depth(edges, ratio, fs, [1.,4.], [1.,1.]), expected)
        self.assertNotAlmostEqual(expected, ratio.mean())
        # Throughput normalization cancels.
        self.assertAlmostEqual(utils.photon_weighted_depth(edges, ratio, fs, [1.,4.], [2.,2.]), expected)

    def test_filter_wings_and_invalid_response(self):
        for wave, response in [([.9, 3.], [0.,1.]), ([1.,4.1], [1.,0.]),
                               ([1.,4.], [0.,0.]), ([1.,2.,2.], [0.,1.,0.]),
                               ([1.,4.], [1.,-1.])]:
            with self.assertRaises(ValueError):
                utils.photon_weighted_depth([1.,2.,4.], [.1,.1], [1.,1.], wave, response)

    def test_revised_prior_must_fit_grid(self):
        priors = {k: list(v) for k,v in retrieval.SYSTEM_PRIORS.items()}
        priors['R_planet_star'][1] = 0.
        with self.assertRaises(ValueError):
            retrieval.configure_priors(priors)
        priors = {k: list(v) for k,v in retrieval.SYSTEM_PRIORS.items()}
        priors['Teq'][2] = 380.
        with self.assertRaises(ValueError):
            retrieval.configure_priors(priors)


class ForwardModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        retrieval.initialize_retrieval()
        retrieval.F1500W_RESPONSE = np.loadtxt('inputs/F1500W_response.txt').T

    @classmethod
    def tearDownClass(cls):
        retrieval.F1500W_RESPONSE = None

    def test_all_observed_cases_at_prior_edges(self):
        # Includes chi=10**log10(bound), which can fall a few ulps outside
        # the saved grid without an explicit numerical boundary correction.
        for model in ('rock', 'atm', 'atm_hot'):
            for data in ('16_F1500W_obs', 'F1500W_obs'):
                case = retrieval.RETRIEVAL_CASES[f'{model}_{data}']
                for q in (0., .5, 1.):
                    x = case['prior'](np.full(len(case['param_names']), q))
                    self.assertTrue(np.isfinite(case['loglike'](x)))

    def test_gray_surface_temperature_cancellation(self):
        dark = retrieval.model_rock_raw([0., 431., 3340., .0454], retrieval.WAVL)
        gray = retrieval.model_rock_raw([.2, 431., 3340., .0454], retrieval.WAVL)
        np.testing.assert_allclose(gray, .8*dark)

    def test_grid_rejects_unsupported_temperature(self):
        with self.assertRaises(ValueError):
            retrieval.SPECTRA(np.array([-7., -7., -5., -7., .2, .2, 380.]))


if __name__ == '__main__':
    unittest.main()
