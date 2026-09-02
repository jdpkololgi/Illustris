from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from astropy.cosmology import Planck18
import numpy as np
import torch

from workflows.abacus_tweb.p8_deterministic_common import sha256
from workflows.sbi.p12f_common_evaluator import (
    efficient_crps_ensemble,
    sample_eigenvalues_at_galaxies,
    validate_archive_manifest,
)
from workflows.sbi.p12f_freeze_selection_panel import shell_from_radius
from workflows.sbi.p12f_gaussian_controls import (
    correlated_unit_residuals,
    finalize_residual_filter,
    residual_filter_accumulator,
    sample_correlated_gaussian,
    update_residual_filter_accumulator,
)
from workflows.sbi.p12f_train_matched_challenger import challenger_loss


class ConstantGaussian(torch.nn.Module):
    def forward(self, condition):
        shape = (condition.shape[0], 1, *condition.shape[2:])
        return torch.zeros(shape), torch.zeros(shape)


class ZeroDiffusion(torch.nn.Module):
    def forward(self, noisy, time_value, condition):
        del time_value, condition
        return torch.zeros_like(noisy)


class P12FProductionChallengerTest(unittest.TestCase):
    def test_crps_matches_brute_force_definition(self):
        rng = np.random.default_rng(3)
        samples = rng.normal(size=(7, 13))
        truth = rng.normal(size=13)
        brute = np.mean(
            np.mean(np.abs(samples - truth[None]), axis=0)
            - 0.5
            * np.mean(
                np.abs(samples[:, None, :] - samples[None, :, :]),
                axis=(0, 1),
            )
        )
        self.assertAlmostEqual(
            efficient_crps_ensemble(samples, truth), float(brute), places=12
        )

    def test_challenger_losses_exclude_m0_and_replay_global_rng(self):
        condition = torch.zeros((1, 3, 4, 4, 4))
        target = torch.full((1, 1, 4, 4, 4), 1000.0)
        target[..., 1, 1, 1] = 1.0
        support = torch.zeros_like(target, dtype=torch.bool)
        support[..., 1, 1, 1] = True
        core = (slice(0, 4), slice(0, 4), slice(0, 4))
        loss, _ = challenger_loss(
            "gaussian", ConstantGaussian(), condition, target, support, core
        )
        self.assertAlmostEqual(float(loss), 0.5)
        torch.manual_seed(8)
        first, _ = challenger_loss(
            "diffusion", ZeroDiffusion(), condition, target, support, core
        )
        torch.manual_seed(8)
        second, _ = challenger_loss(
            "diffusion", ZeroDiffusion(), condition, target, support, core
        )
        torch.testing.assert_close(first, second)
        with self.assertRaises(RuntimeError):
            challenger_loss(
                "gaussian",
                ConstantGaussian(),
                condition,
                target,
                torch.zeros_like(support),
                core,
            )

    def test_variable_shape_g1_filter_and_sampling(self):
        rng = np.random.default_rng(2)
        accumulator = residual_filter_accumulator(8)
        update_residual_filter_accumulator(
            accumulator, rng.normal(size=(6, 7, 8))
        )
        update_residual_filter_accumulator(
            accumulator, rng.normal(size=(8, 6, 7))
        )
        contract = finalize_residual_filter(accumulator)
        self.assertTrue(contract["supports_variable_shapes"])
        self.assertEqual(contract["fields"], 2)
        first = correlated_unit_residuals(
            contract, draws=4, seed=7, shape=(5, 6, 7)
        )
        second = correlated_unit_residuals(
            contract, draws=4, seed=7, shape=(5, 6, 7)
        )
        np.testing.assert_array_equal(first, second)
        sample = sample_correlated_gaussian(
            np.zeros((5, 6, 7)),
            np.ones((5, 6, 7)),
            contract,
            draws=4,
            seed=7,
        )
        self.assertEqual(sample.shape, (4, 5, 6, 7))

    def test_shell_contract_and_eigenvalue_sampling(self):
        bounds = [
            Planck18.comoving_distance(value).value
            for value in (0.15, 0.25, 0.35, 0.45, 0.55)
        ]
        for shell in range(4):
            self.assertEqual(
                shell_from_radius(0.5 * (bounds[shell] + bounds[shell + 1])),
                shell,
            )
        self.assertEqual(shell_from_radius(bounds[0] - 1.0), -1)
        self.assertEqual(shell_from_radius(bounds[-1] + 1.0), -1)
        draws = np.zeros((2, 4, 4, 4, 3), dtype=np.float32)
        truth = np.zeros((4, 4, 4, 3), dtype=np.float32)
        draws[..., 0] = 3.0
        draws[..., 1] = 2.0
        draws[..., 2] = 1.0
        truth[..., 0] = 3.0
        truth[..., 1] = 2.0
        truth[..., 2] = 1.0
        points = np.asarray([[1.2, 2.1, 0.7], [2.0, 1.0, 3.0]])
        sample, target = sample_eigenvalues_at_galaxies(draws, truth, points)
        self.assertEqual(sample.shape, (2, 2, 3))
        np.testing.assert_allclose(sample[..., 0], 3.0)
        np.testing.assert_allclose(target[:, 2], 1.0)

    def test_archive_validation_rejects_ph001_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core = root / "core.npz"
            np.savez(core, value=np.asarray([1]))
            panel_path = root / "panel.json"
            panel = {"selected_core_id": [7]}
            panel_path.write_text(json.dumps(panel))
            config = {"matched_contract": {"posterior_draws": 64}}
            archive_path = root / "archive.json"
            archive = {
                "schema_version": "p12f-sample-archive-v1",
                "phase": "ph006",
                "ph001_opened": False,
                "truth_files_read": ["ph006"],
                "draws": 64,
                "panel_sha256": sha256(panel_path),
                "entries": [
                    {"core_id": 7, "path": str(core), "sha256": sha256(core)}
                ],
            }
            archive_path.write_text(json.dumps(archive))
            rows = validate_archive_manifest(
                archive,
                archive_path=archive_path,
                panel=panel,
                panel_path=panel_path,
                config=config,
            )
            self.assertEqual(rows[0]["core_id"], 7)
            bad = root / "ph001_bad.npz"
            np.savez(bad, value=np.asarray([1]))
            archive["entries"][0].update(path=str(bad), sha256=sha256(bad))
            with self.assertRaises(PermissionError):
                validate_archive_manifest(
                    archive,
                    archive_path=archive_path,
                    panel=panel,
                    panel_path=panel_path,
                    config=config,
                )


if __name__ == "__main__":
    unittest.main()
