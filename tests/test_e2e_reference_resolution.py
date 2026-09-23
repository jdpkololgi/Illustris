import unittest
import numpy as np
from workflows.sbi.e2e_reference_resolution import physical_prior, nested_indices


class ResolutionTests(unittest.TestCase):
    def test_same_physical_prior_on_nested_sites(self):
        small,_=physical_prior(4,cutoff=4)
        large,_=physical_prior(8,cutoff=4)
        ix=nested_indices(8,coarse=4)
        np.testing.assert_allclose(small,large[np.ix_(ix,ix)],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(np.diag(small),1.,atol=1e-12)
        self.assertGreater(np.linalg.eigvalsh(small).min(),0.)

    def test_reject_nonnested(self):
        with self.assertRaises(ValueError): nested_indices(12)


if __name__=='__main__': unittest.main()
