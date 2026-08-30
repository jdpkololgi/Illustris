import unittest

import numpy as np

from workflows.sbi.p12_calibration_diagnostics import (
    choose_indices,
    randomized_pit,
    rank_summary,
)


class P12CalibrationDiagnosticsTests(unittest.TestCase):
    def test_randomized_finite_ranks_are_uniform_for_exact_posterior(self):
        rng = np.random.default_rng(123)
        rows, draws, components = 12_000, 64, 3
        truth = rng.normal(size=(rows, components))
        samples = rng.normal(size=(rows, draws, components))
        ranks = randomized_pit(samples, truth, seed=8)
        summary = rank_summary(ranks, np.ones(rows), ("a", "b", "c"))
        for row in summary["components"].values():
            self.assertLess(row["weighted_ks_distance"], 0.02)
            self.assertLess(abs(row["weighted_mean_rank"] - 0.5), 0.02)

    def test_rank_shape_identifies_low_location_and_underdispersion(self):
        rng = np.random.default_rng(17)
        rows, draws = 8_000, 128
        truth = rng.normal(size=(rows, 3))
        low_samples = rng.normal(loc=-0.5, size=(rows, draws, 3))
        low_ranks = randomized_pit(low_samples, truth, seed=9)
        low = rank_summary(low_ranks, np.ones(rows), ("a", "b", "c"))
        for row in low["components"].values():
            self.assertIn(
                "posterior_location_low_relative_to_truth",
                row["interpretation_flags"],
            )

        narrow_samples = rng.normal(scale=0.25, size=(rows, draws, 3))
        narrow_ranks = randomized_pit(narrow_samples, truth, seed=10)
        narrow = rank_summary(narrow_ranks, np.ones(rows), ("a", "b", "c"))
        for row in narrow["components"].values():
            self.assertIn(
                "posterior_too_narrow_or_heavy_truth_tails",
                row["interpretation_flags"],
            )

    def test_calibration_and_selection_fold_rows_are_disjoint(self):
        fold = np.tile(np.arange(5, dtype=np.uint8), 100)
        calibration, evaluation = choose_indices(
            fold, [0, 1], [2, 3, 4], 80, 120, seed=42
        )
        self.assertEqual(len(calibration), 80)
        self.assertEqual(len(evaluation), 120)
        self.assertEqual(np.intersect1d(calibration, evaluation).size, 0)
        self.assertTrue(np.all(np.isin(fold[calibration], [0, 1])))
        self.assertTrue(np.all(np.isin(fold[evaluation], [2, 3, 4])))


if __name__ == "__main__":
    unittest.main()
