import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from workflows.sbi.e2e_field_error_budget import sample_grid
from workflows.sbi.e2e_field_regenerate_spectral import interpolate, SpectralParentDataset


class SpectralRegenerationTests(unittest.TestCase):
    def test_separable_cubic_matches_audited_corner_sum(self):
        field = np.random.default_rng(78).normal(size=(17,)*3)
        coords = [np.array([-.1,.4,5.2,16.6])]*3
        np.testing.assert_allclose(interpolate(field,coords),sample_grid(field,coords,"local_cubic_lagrange"),atol=2e-15)

    def test_quintic_polynomial_exactness(self):
        grid = np.indices((14,)*3,dtype=float)
        field = grid[0]**5+grid[1]**4+grid[2]**3
        coords = [np.array([3.2,5.7,8.8])]*3
        expected = coords[0][:,None,None]**5+coords[1][None,:,None]**4+coords[2][None,None,:]**3
        np.testing.assert_allclose(interpolate(field,coords,5),expected,rtol=3e-15)

    def test_constant_and_integer_periodic_sampling(self):
        field = np.random.default_rng(19).normal(size=(12,)*3)
        coords = [np.array([-1.,0.,13.])]*3
        np.testing.assert_allclose(interpolate(field,coords,5),field[np.ix_(*([np.array([11,0,1])]*3))],atol=1e-14)

    def test_guards_before_payload_access(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"index.json"
            path.write_text(json.dumps({"schema_version":"e2e-spectral-products-v2","training_ready":False}))
            with self.assertRaises(PermissionError):
                SpectralParentDataset(path,"train")
            with self.assertRaises(PermissionError):
                SpectralParentDataset(path,"internal_confirmation",allow_unreleased=True)


if __name__ == "__main__":
    unittest.main()
