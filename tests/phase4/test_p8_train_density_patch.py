import unittest
from pathlib import Path
import tempfile

import numpy as np
import torch

from workflows.abacus_tweb.p8_train_density_patch import (
    RegressionAccumulator,
    checkpoint_cuda_rng_state,
)
from workflows.abacus_tweb.p8_deterministic_common import acquire_run_lock
from workflows.abacus_tweb.p8_train_patch_recovery import atomic_torch_save


class DensityTrainerTests(unittest.TestCase):
    def test_run_lock_rejects_second_owner_and_releases_on_close(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.lock"
            first = acquire_run_lock(path, purpose="first")
            with self.assertRaises(RuntimeError):
                acquire_run_lock(path, purpose="second")
            first.close()
            second = acquire_run_lock(path, purpose="second")
            second.close()

    def test_atomic_torch_save_uses_replaceable_unique_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            atomic_torch_save({"value": torch.tensor([1, 2])}, path)
            atomic_torch_save({"value": torch.tensor([3, 4])}, path)
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            self.assertTrue(torch.equal(loaded["value"], torch.tensor([3, 4])))
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_regression_accumulator_matches_direct_metrics(self):
        truth = np.array([-2.0, -0.5, 1.0, 3.0, 4.5])
        prediction = np.array([-1.5, -0.2, 0.7, 2.8, 4.1])
        accumulator = RegressionAccumulator()
        accumulator.add(prediction[:2], truth[:2])
        accumulator.add(prediction[2:], truth[2:])
        report = accumulator.report()
        residual = prediction - truth
        expected_r2 = 1.0 - np.square(residual).sum() / np.square(truth - truth.mean()).sum()
        self.assertEqual(report["n"], len(truth))
        self.assertAlmostEqual(report["r2"], expected_r2)
        self.assertAlmostEqual(report["pearson"], np.corrcoef(prediction, truth)[0, 1])
        self.assertAlmostEqual(report["rmse"], np.sqrt(np.square(residual).mean()))
        self.assertAlmostEqual(report["mae"], np.abs(residual).mean())

    def test_regression_accumulator_rejects_nonfinite_truth(self):
        accumulator = RegressionAccumulator()
        with self.assertRaises(ValueError):
            accumulator.add(np.array([0.0, 1.0]), np.array([0.0, np.nan]))

    def test_regression_accumulator_perfect_prediction(self):
        truth = np.array([-1.0, 0.0, 2.0])
        accumulator = RegressionAccumulator()
        accumulator.add(truth, truth)
        report = accumulator.report()
        self.assertAlmostEqual(report["r2"], 1.0)
        self.assertAlmostEqual(report["pearson"], 1.0)
        self.assertAlmostEqual(report["rmse"], 0.0)

    def test_checkpoint_cuda_rng_state_prefers_v2_state(self):
        current = torch.tensor([1, 2, 3], dtype=torch.uint8)
        legacy = torch.tensor([4, 5, 6], dtype=torch.uint8)
        selected = checkpoint_cuda_rng_state({
            "cuda_rng_state": current,
            "cuda_rng_state_all": [legacy],
        })
        self.assertTrue(torch.equal(selected, current))

    def test_checkpoint_cuda_rng_state_accepts_legacy_multigpu_state(self):
        gpu0 = torch.tensor([1, 2, 3], dtype=torch.uint8)
        gpu1 = torch.tensor([4, 5, 6], dtype=torch.uint8)
        selected = checkpoint_cuda_rng_state({"cuda_rng_state_all": [gpu0, gpu1]})
        self.assertTrue(torch.equal(selected, gpu0))

    def test_checkpoint_cuda_rng_state_rejects_missing_state(self):
        with self.assertRaises(KeyError):
            checkpoint_cuda_rng_state({})
        with self.assertRaises(KeyError):
            checkpoint_cuda_rng_state({"cuda_rng_state_all": []})


if __name__ == "__main__":
    unittest.main()
