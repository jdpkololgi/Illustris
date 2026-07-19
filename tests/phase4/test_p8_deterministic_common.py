import unittest

import numpy as np

from workflows.abacus_tweb.p8_deterministic_common import (
    evaluate_complete_fold,
    fit_affine_on_training,
    fit_target_scaler,
    increments_to_eigenvalues,
    linear_increments,
    scale_increments,
    shell_weights,
    unscale_increments,
)


class P8CommonTests(unittest.TestCase):
    def setUp(self):
        self.truth = np.array(
            [
                [-0.4, -0.2, 0.1], [-0.3, 0.0, 0.2], [-0.2, 0.1, 0.3],
                [-0.1, 0.2, 0.4], [0.0, 0.3, 0.5], [0.1, 0.4, 0.6],
                [0.2, 0.5, 0.7], [0.3, 0.6, 0.8],
                [0.4, 0.7, 0.9], [0.5, 0.8, 1.0], [0.6, 0.9, 1.1],
                [0.7, 1.0, 1.2],
            ],
            dtype=np.float64,
        )

    def test_increment_scaler_roundtrip(self):
        scaler = fit_target_scaler(self.truth)
        scaled = scale_increments(linear_increments(self.truth), scaler)
        got = increments_to_eigenvalues(unscale_increments(scaled, scaler))
        np.testing.assert_allclose(got, self.truth, atol=1e-6)

    def test_shell_weights_are_sqrt_balanced(self):
        shell = np.repeat(np.arange(4), [4, 9, 16, 25])
        weight, counts = shell_weights(shell)
        exposure = np.bincount(shell, weights=weight, minlength=4)
        np.testing.assert_allclose(exposure, np.sqrt([4, 9, 16, 25]))
        self.assertEqual(counts["0p45_0p55"], 25)

    def test_complete_fold_identity(self):
        assignment = {
            "parent_node_id": np.arange(12),
            "supervised_eligible": np.ones(12, bool),
            "fold": np.zeros(12, np.uint8),
            "shell": np.repeat(np.arange(4), 3),
            "superblock_id": np.arange(12) // 2,
            "distance_to_conservative_fold_boundary_mpc": np.arange(12) + 30.0,
        }
        report = evaluate_complete_fold(
            parent_node_id=np.arange(12),
            predicted_eigenvalues=self.truth,
            truth_by_parent=self.truth,
            assignment=assignment,
            validation_fold=0,
        )
        self.assertAlmostEqual(report["primary_macro_r2_lambda1"], 1.0)
        self.assertTrue(report["complete_core_coverage"])

    def test_missing_validation_row_is_rejected(self):
        assignment = {
            "parent_node_id": np.arange(12),
            "supervised_eligible": np.ones(12, bool),
            "fold": np.zeros(12, np.uint8),
            "shell": np.repeat(np.arange(4), 3),
            "superblock_id": np.arange(12) // 2,
            "distance_to_conservative_fold_boundary_mpc": np.arange(12) + 30.0,
        }
        with self.assertRaises(RuntimeError):
            evaluate_complete_fold(
                parent_node_id=np.arange(11),
                predicted_eigenvalues=self.truth[:11],
                truth_by_parent=self.truth,
                assignment=assignment,
                validation_fold=0,
            )

    def test_affine_is_training_only(self):
        raw = self.truth * 2.0 + 1.0
        calibrated, spec = fit_affine_on_training(
            raw, self.truth, np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], bool)
        )
        np.testing.assert_allclose(calibrated, self.truth, atol=1e-12)
        self.assertEqual(spec["fit_split"], "training cores")


if __name__ == "__main__":
    unittest.main()
