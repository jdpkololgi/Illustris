import unittest

import numpy as np

from workflows.abacus_tweb.p8_epoch_training import (
    EpochLossAccumulator,
    epoch_order,
    improved,
    patch_objective,
    should_stop,
    validate_resume_order,
)


class P8EpochTrainingTests(unittest.TestCase):
    def test_epoch_order_is_complete_deterministic_and_epoch_specific(self):
        cores = np.arange(101, dtype=np.int64)
        first = epoch_order(cores, seed=42, epoch=1)
        repeat = epoch_order(cores, seed=42, epoch=1)
        second = epoch_order(cores, seed=42, epoch=2)
        np.testing.assert_array_equal(first, repeat)
        np.testing.assert_array_equal(np.sort(first), cores)
        self.assertEqual(len(np.unique(first)), len(cores))
        self.assertFalse(np.array_equal(first, second))

    def test_epoch_order_rejects_duplicate_cores(self):
        with self.assertRaises(ValueError):
            epoch_order(np.array([1, 2, 2]), seed=42, epoch=1)

    def test_weighted_order_is_complete_and_prefers_large_weights(self):
        cores = np.arange(20, dtype=np.int64)
        weight = np.ones(20)
        weight[-1] = 1_000.0
        order = epoch_order(cores, seed=42, epoch=1, core_weight=weight)
        np.testing.assert_array_equal(np.sort(order), cores)
        self.assertEqual(int(order[0]), 19)
        with self.assertRaises(ValueError):
            epoch_order(cores, seed=42, epoch=1, core_weight=np.ones(19))

    def test_patch_objective_recovers_global_row_weighted_mean(self):
        # Three memory patches with deliberately unequal scientific weights.
        patch_losses = [
            np.array([1.0, 4.0]),
            np.array([9.0]),
            np.array([16.0, 25.0, 36.0]),
        ]
        patch_weights = [
            np.array([1.0, 2.0]),
            np.array([0.5]),
            np.array([3.0, 1.0, 2.0]),
        ]
        core_weight = np.array([w.sum() for w in patch_weights])
        objectives = [
            patch_objective(np.sum(loss * weight), mean_core_weight=core_weight.mean())
            for loss, weight in zip(patch_losses, patch_weights)
        ]
        expected = sum(
            np.sum(loss * weight)
            for loss, weight in zip(patch_losses, patch_weights)
        ) / core_weight.sum()
        self.assertAlmostEqual(float(np.mean(objectives)), float(expected))

    def test_accumulator_roundtrip_preserves_partial_epoch(self):
        accumulator = EpochLossAccumulator()
        accumulator.add(np.array([1.0, 3.0]), np.array([2.0, 1.0]))
        restored = EpochLossAccumulator.from_dict(accumulator.as_dict())
        self.assertEqual(restored.rows, 2)
        self.assertEqual(restored.patches, 1)
        self.assertAlmostEqual(restored.mean, 5.0 / 3.0)

    def test_resume_order_rejects_wrong_epoch_or_cursor(self):
        expected = np.arange(5)
        validate_resume_order(np.array([4, 3, 2, 1, 0]), expected, 3)
        with self.assertRaises(ValueError):
            validate_resume_order(np.array([4, 3, 2, 1, 9]), expected, 3)
        with self.assertRaises(ValueError):
            validate_resume_order(np.array([4, 3, 2, 1, 0]), expected, 6)

    def test_registered_early_stopping_contract(self):
        self.assertTrue(improved(0.505, 0.5, 0.005))
        self.assertFalse(improved(0.5049, 0.5, 0.005))
        self.assertFalse(should_stop(epoch=4, stale_epochs=4, min_epochs=5, patience=3))
        self.assertTrue(should_stop(epoch=5, stale_epochs=3, min_epochs=5, patience=3))


if __name__ == "__main__":
    unittest.main()
