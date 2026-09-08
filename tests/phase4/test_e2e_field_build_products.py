import copy
import json
import unittest

import numpy as np

from workflows.sbi.e2e_field_build_products import (
    DEFAULT_CONFIG, context_slice, native_indices, tensor_from_delta, eigs, validate_config,
)


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(DEFAULT_CONFIG.read_text())

    def test_role_change_fails(self):
        self.config["phase_roles"]["ph001"] = "train"
        with self.assertRaises(ValueError):
            validate_config(self.config)

    def test_cannot_authorize_training(self):
        self.config["science_training_authorized"] = True
        with self.assertRaises(ValueError):
            validate_config(self.config)

    def test_trace_and_constant_mean(self):
        rng = np.random.default_rng(34)
        delta = rng.normal(size=(12, 12, 12)) + 0.7
        tensor = tensor_from_delta(delta, 3.383)
        np.testing.assert_allclose(tensor[..., [0, 3, 5]].sum(axis=-1), delta, atol=2e-14)
        ordered = eigs(tensor)
        self.assertTrue(np.all(np.diff(ordered, axis=-1) >= 0))
        constant = tensor_from_delta(np.ones((8, 8, 8))*0.6, 3.383)
        np.testing.assert_allclose(constant[..., [0, 3, 5]], 0.2, atol=1e-14)
        np.testing.assert_allclose(constant[..., [1, 2, 4]], 0, atol=1e-14)

    def test_native_indices_periodic(self):
        grid = {"origin_mpc": [-2000.0, 0, 2000.0]}
        axes = native_indices(grid, [0, 8, 16], 32, self.config)
        self.assertTrue(all(np.all((x >= 0) & (x < 2048)) for x in axes))
        for a in range(3):
            expected = [int(np.floor(((grid["origin_mpc"][a] +
                       (i+0.5)*5) * 0.6766 - 1000) % 2000 / (2000/2048)))
                        for i in range([0,8,16][a], [0,8,16][a]+32)]
            np.testing.assert_array_equal(axes[a], expected)

    def test_matched_science_region(self):
        center = [64, 72, 80]
        for side in [64, 96]:
            parent = context_slice(center, side)
            core = context_slice([side//2]*3, 32)
            self.assertEqual([x.start+y.start for x,y in zip(parent,core)],
                             [x-16 for x in center])


if __name__ == "__main__":
    unittest.main()
