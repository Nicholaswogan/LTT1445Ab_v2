"""Validate the new free profile, marginalized prior, and observation operator."""
import unittest

import numpy as np
from scipy.stats import ks_2samp, qmc

from joint_system_prior import JointSystemPrior
from retrieval_run_free import pt_profile, MarginalStellarPrior, FreeRetrieval, BOUNDS, PARAMETERS


class ProfileTests(unittest.TestCase):
    def test_reference_temperature_and_continuity(self):
        for p_break in (1.e-5, .01, 1.):
            self.assertAlmostEqual(pt_profile([.1], 600., p_break, -30., 100.)[0], 600.)
            nearby = pt_profile(p_break*np.array([1-1.e-8, 1+1.e-8]), 600., p_break, -30., 100.)
            self.assertLess(abs(np.diff(nearby)[0]), 1.e-5)
        p = np.geomspace(1.e-6, 100., 100)
        np.testing.assert_allclose(pt_profile(p, 700., .01, 50., 50.), 700+50*np.log10(p/.1))

    def test_marginal_matches_original_prior(self):
        original = JointSystemPrior.from_chain()
        marginal = MarginalStellarPrior(original)
        reference = original.transform(qmc.Sobol(3, scramble=True, seed=7).random_base2(17))[:, 1:]
        draws = np.array([marginal.transform(u) for u in
                          qmc.Sobol(2, scramble=True, seed=7).random_base2(12)])
        for i in (0, 1):
            self.assertLess(ks_2samp(reference[:, i], draws[:, i]).statistic, .01)
        self.assertLess(abs(np.corrcoef(reference.T)[0, 1]-np.corrcoef(draws.T)[0, 1]), .01)
        # Check quadrature convergence rather than just similarity to chains.
        refined = MarginalStellarPrior(original, quadrature_order=384)
        for u in np.random.default_rng(17).random((100, 2)):
            # <0.2 K is negligible beside the ~150 K stellar uncertainty.
            self.assertLess(abs(marginal.transform(u)[0]-refined.transform(u)[0]), .2)
        for u in ([0., 0.], [1., 1.], [0., 1.], [1., 0.]):
            self.assertTrue(np.isfinite(marginal.transform(u)).all())


class FreeForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retrieval = FreeRetrieval()

    def example(self):
        x = self.retrieval.prior(np.full(11, .5))
        x[:4] = -5.
        x[5:9] = [600., -3., 0., 0.]
        return x

    def test_priors_data_and_temperature_independence(self):
        self.assertEqual(len(PARAMETERS), 11)
        np.testing.assert_array_equal(BOUNDS[:5], [[-7, 2], [-7, 2], [-5, 2], [-7, 2], [0, .4]])
        self.assertEqual(len(self.retrieval.data['fpfs']), 17)
        photometric = self.retrieval.data['photometric']
        np.testing.assert_allclose(self.retrieval.data['fpfs'][photometric], [157.4729237374e-6])
        np.testing.assert_allclose(self.retrieval.data['err_lower'][photometric], [17.313102253804e-6])
        np.testing.assert_allclose(self.retrieval.data['err_upper'][photometric], [17.816033503901e-6])
        x = self.example()
        before = self.retrieval.atmosphere(x)
        x[0] = -2.
        after = self.retrieval.atmosphere(x)
        # Isothermal temperature stays fixed despite the altered composition.
        np.testing.assert_array_equal(before.temperature, after.temperature)
        self.assertGreater(after.pressure.iloc[-1], before.pressure.iloc[-1])

    def test_invalid_profiles_and_response_operator(self):
        from utils import photon_weighted_depth
        x = self.example()
        invalid = x.copy()
        invalid[7] = 400.
        self.assertIsNone(self.retrieval.atmosphere(invalid))
        self.assertLess(self.retrieval.loglike(invalid), -1.e10)
        different = invalid.copy()
        different[7] = 300.
        self.assertNotEqual(self.retrieval.loglike(invalid), self.retrieval.loglike(different))
        prediction = self.retrieval.predict(x)
        self.assertTrue(np.isfinite(prediction).all())
        # Compare photometry to a separate application of the shared integrator.
        _, fp, _ = self.retrieval.picaso.fpfs(self.retrieval.atmosphere(x), wavl=self.retrieval.wavelength)
        wave, fs = self.retrieval.sphinx(x[-2], self.retrieval.star.metal, self.retrieval.star.logg, rescale_to_Teff=False)
        fs = self.retrieval.stars.rebin(self.retrieval.stars.make_bins(wave), fs, self.retrieval.wavelength)
        expected = photon_weighted_depth(self.retrieval.wavelength, fp/fs*x[-1]**2, fs, *self.retrieval.response)
        np.testing.assert_allclose(prediction[self.retrieval.data['photometric']], [expected], rtol=1.e-10)


if __name__ == '__main__':
    unittest.main()
