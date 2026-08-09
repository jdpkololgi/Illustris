import unittest

import numpy as np
import torch

from workflows.abacus_tweb.p8_validate_density_tensor_closure import (
    apply_box_taper,
    orientation_stability,
    radial_cosine_window,
    solve_tensors_at_positions,
)


class DensityTensorClosureTests(unittest.TestCase):
    def test_box_taper_is_symmetric_and_preserves_centre(self):
        field = torch.ones((9, 11, 13), dtype=torch.float32)
        tapered = apply_box_taper(field, 3).numpy()
        self.assertAlmostEqual(float(tapered[4, 5, 6]), 1.0)
        np.testing.assert_allclose(tapered, tapered[::-1, ::-1, ::-1], atol=1e-7)
        self.assertLess(float(tapered[0, 0, 0]), 0.01)

    def test_radial_window_has_zero_exterior_and_unit_interior(self):
        # Put x centres near the P8 radial range; keep y,z small.
        origin = np.array([400.0, -2.5, -2.5])
        window = radial_cosine_window((400, 1, 1), origin, 5.0, 100.0, "cpu")
        values = window[:, 0, 0].numpy()
        self.assertEqual(float(values[0]), 0.0)
        self.assertGreater(float(values.max()), 0.999)
        self.assertEqual(float(values[-1]), 0.0)

    def test_unsmoothed_projector_trace_matches_input_minus_k0(self):
        rng = np.random.default_rng(19)
        field = torch.from_numpy(rng.normal(size=(14, 16, 18)).astype(np.float32))
        origin = np.array([0.0, 0.0, 0.0])
        positions = {
            "probe": np.array([[22.5, 27.5, 32.5], [37.5, 42.5, 47.5]])
        }
        tensors, report = solve_tensors_at_positions(
            field, positions=positions, origin_mpc=origin, cell_mpc=5.0,
            padding_voxels=3,
        )
        self.assertEqual(tensors["probe"].shape, (2, 3, 3))
        self.assertLess(report["trace_identity"]["probe"]["rmse"], 2e-6)

    def test_orientation_stability_is_zero_for_identical_tensors(self):
        tensor = np.array([
            [[-1.0, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 1.5]],
            [[-0.4, 0.1, 0.0], [0.1, 0.3, 0.0], [0.0, 0.0, 1.1]],
            [[-0.7, 0.0, 0.2], [0.0, 0.4, 0.0], [0.2, 0.0, 1.3]],
            [[-0.2, 0.0, 0.0], [0.0, 0.6, 0.1], [0.0, 0.1, 1.4]],
        ])
        report = orientation_stability(tensor, tensor.copy())
        for row in report["eigengap_quantile_bins"].values():
            self.assertLess(max(row["median_angle_deg"]), 1e-5)


if __name__ == "__main__":
    unittest.main()
