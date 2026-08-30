import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from workflows.sbi.p12_calibration_diagnostics import (
    choose_indices,
    randomized_pit,
    rank_summary,
    sample_posterior_resumable,
    spatial_block_bootstrap,
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

    def test_resumable_sampling_writes_progress_and_resumes(self):
        context = np.arange(30, dtype=np.float32).reshape(10, 3)

        def fake_sample(_posterior, batch, draws, _chunk, _device):
            values = batch[:, :1, None]
            return np.broadcast_to(values, (len(batch), draws, 3)).copy()

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample_path = root / "samples.npy"
            progress_path = root / "progress.json"
            with patch(
                "workflows.sbi.p12_calibration_diagnostics.sample_posterior",
                side_effect=fake_sample,
            ):
                first = sample_posterior_resumable(
                    object(), context, 4, 3, "cpu", sample_path, progress_path
                )
                self.assertEqual(first.shape, (10, 4, 3))
                self.assertEqual(
                    json.loads(progress_path.read_text())["completed_rows"], 10
                )
                progress = json.loads(progress_path.read_text())
                progress["completed_rows"] = 7
                progress_path.write_text(json.dumps(progress))
                second = sample_posterior_resumable(
                    object(), context, 4, 3, "cpu", sample_path, progress_path
                )
                np.testing.assert_allclose(first, second)

    def test_spatial_block_bootstrap_retains_uniform_decile_shape(self):
        blocks, rows_per_block = 20, 100
        groups = np.repeat(np.arange(blocks), rows_per_block)
        within = np.tile(
            np.repeat((np.arange(10) + 0.5) / 10.0, 10), blocks
        )
        ranks = np.column_stack((within, within, within))
        samples = np.zeros((len(ranks), 8, 3), dtype=np.float32)
        truth = np.zeros((len(ranks), 3), dtype=np.float32)
        result = spatial_block_bootstrap(
            ranks,
            samples,
            truth,
            np.ones(len(ranks)),
            groups,
            repeats=100,
            seed=11,
        )
        self.assertEqual(result["spatial_blocks"], blocks)
        for row in result["components"].values():
            self.assertEqual(row["uniform_deciles_outside_pointwise_95ci"], [])
if __name__ == "__main__":
    unittest.main()
