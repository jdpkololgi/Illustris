import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from workflows.abacus_tweb.p11_jepa_latent_diagnostics import (
    evaluate_series,
    linear_cka,
    load_latent_snapshot,
    save_latent_snapshot,
    spread_metrics,
)
from workflows.abacus_tweb.plot_p11_jepa_latents import (
    fit_reference_projection,
    load_projection_state,
    save_projection_state,
)


class P11JEPALatentDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(91)
        self.rows = 480
        self.dim = 12
        shared = rng.normal(size=(self.rows, self.dim))
        rotation, _ = np.linalg.qr(rng.normal(size=(self.dim, self.dim)))
        self.dense = shared + 0.04 * rng.normal(size=shared.shape)
        self.degraded = shared @ rotation + 0.10 * rng.normal(size=shared.shape)
        self.predicted = shared + 0.08 * rng.normal(size=shared.shape)
        self.response = np.linspace(0.05, 1.0, self.rows)
        self.split = np.zeros(self.rows, dtype=np.int8)
        self.split[self.rows // 2 :] = 1
        self.target = np.column_stack(
            (
                0.8 * shared[:, 0] - 0.3 * shared[:, 1],
                shared[:, 2] + 0.2 * shared[:, 3],
                shared[:, 4] - shared[:, 5],
            )
        )

    def _write(
        self,
        root: Path,
        epoch: int,
        step: int,
        *,
        phase="ph006",
        arm="jepa",
        include_response_only=True,
        include_target=True,
        degraded=None,
    ) -> Path:
        path = root / f"latent_epoch_{epoch:03d}.npz"
        response_only = np.column_stack(
            [np.sin((index + 1) * self.response) for index in range(self.dim)]
        )
        save_latent_snapshot(
            path,
            metadata={
                "run_id": "synthetic-jepa",
                "arm": arm,
                "predictor_trained": arm == "jepa",
                "epoch": epoch,
                "global_step": step,
                "phase": phase,
                "sealed_phase_opened": False,
                "source_paths": [f"/visible/{phase}/paired_views"],
            },
            sample_id=np.arange(self.rows, dtype=np.int64),
            dense_latent=self.dense,
            degraded_latent=self.degraded if degraded is None else degraded,
            predicted_dense_latent=self.predicted,
            response_only_latent=(response_only if include_response_only else None),
            response_strength=self.response,
            response_features=np.column_stack((self.response, self.response ** 2)),
            probe_split=self.split,
            target=self.target if include_target else None,
            core_id=np.where(self.split == 0, 100, 200),
            fold_id=np.where(self.split == 0, 0, 2),
            group_id=np.where(self.split == 0, 10, 20),
        )
        return path

    def test_shared_structure_beats_shuffled_and_does_not_collapse(self):
        rng = np.random.default_rng(2)
        paired = linear_cka(self.degraded, self.dense)
        shuffled = linear_cka(self.degraded, self.dense[rng.permutation(self.rows)])
        self.assertGreater(paired, 0.90)
        self.assertGreater(paired - shuffled, 0.50)
        self.assertGreater(spread_metrics(self.degraded)["effective_rank_fraction"], 0.75)
        self.assertEqual(spread_metrics(np.zeros_like(self.degraded))["effective_rank"], 0.0)

    def test_series_reports_retrieval_probe_response_and_calibration_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [self._write(root, 0, 0), self._write(root, 1, 100)]
            report = evaluate_series(paths, max_retrieval_rows=240, seed=7)
            self.assertEqual(report["status"], "advisory")
            self.assertEqual(report["registered_status_gate"]["missing_steps"], [250, 500])
            latest = report["checkpoints"][-1]
            self.assertTrue(latest["shared_predictable_subspace_gate"])
            self.assertGreater(latest["cross_view_retrieval"]["mrr_over_shuffle"], 0.5)
            self.assertGreater(
                latest["cross_view_retrieval"]["mrr_over_response_matched_shuffle"],
                0.5,
            )
            self.assertGreater(
                latest["downstream_linear_probe"]["degraded_student"]["macro_r2"], 0.95
            )
            self.assertEqual(len(latest["response_strata"]), 4)
            contract = report["interpretation_contract"]
            self.assertFalse(contract["latent_alignment_is_posterior_calibration"])
            self.assertFalse(contract["posterior_overconfidence_claim_licensed"])
            self.assertTrue(
                contract["posterior_retraining_and_full_p12_recalibration_required_if_promoted"]
            )

    def test_sealed_phase_and_series_identity_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sealed = self._write(root, 0, 0, phase="ph001")
            with self.assertRaisesRegex(ValueError, "sealed"):
                load_latent_snapshot(sealed)
            visible = self._write(root, 0, 0)
            second = root / "latent_epoch_002.npz"
            save_latent_snapshot(
                second,
                metadata={
                    "run_id": "synthetic-jepa",
                    "arm": "jepa",
                    "predictor_trained": True,
                    "epoch": 2,
                    "global_step": 200,
                    "phase": "ph006",
                    "sealed_phase_opened": False,
                },
                sample_id=np.arange(self.rows, dtype=np.int64)[::-1],
                dense_latent=self.dense,
                degraded_latent=self.degraded,
                response_strength=self.response,
                probe_split=self.split,
            )
            with self.assertRaisesRegex(ValueError, "sample_id order"):
                evaluate_series([visible, second], max_retrieval_rows=100)

    def test_registered_0_250_500_trajectory_can_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [
                self._write(root, 0, 0),
                self._write(root, 1, 250),
                self._write(root, 2, 500),
            ]
            report = evaluate_series(paths, max_retrieval_rows=240, seed=7)
            self.assertEqual(report["status"], "pass")
            self.assertTrue(report["pass"])
            self.assertEqual(report["registered_status_gate"]["missing_steps"], [])

    def test_unregistered_epoch_export_does_not_redefine_canary_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [
                self._write(root, 0, 0),
                self._write(root, 1, 250),
                self._write(root, 2, 500),
                self._write(root, 3, 750, degraded=np.zeros_like(self.degraded)),
            ]
            report = evaluate_series(paths, max_retrieval_rows=160, seed=7)
            self.assertEqual(report["status"], "pass")
            self.assertFalse(
                report["registered_temporal_risk_signals"][
                    "alignment_gain_with_rank_loss"
                ]
            )
            self.assertLess(
                report["registered_temporal_risk_signals"][
                    "student_effective_rank_fraction_loss"
                ],
                0.1,
            )
            self.assertGreater(
                report["temporal_risk_signals"][
                    "student_effective_rank_fraction_loss"
                ],
                0.5,
            )

    def test_non_jepa_predictor_is_ignored_and_control_is_advisory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [
                self._write(root, 0, 0, arm="supervised_masked"),
                self._write(root, 1, 250, arm="supervised_masked"),
                self._write(root, 2, 500, arm="supervised_masked"),
            ]
            report = evaluate_series(paths, max_retrieval_rows=160, seed=7)
            self.assertEqual(report["status"], "advisory")
            self.assertIsNone(report["checkpoints"][-1]["predictor"])
            self.assertFalse(report["checkpoints"][-1]["predictor_trained"])
            self.assertNotIn(
                "predicted_teacher_space",
                report["checkpoints"][-1]["downstream_linear_probe"],
            )

    def test_weaker_response_fallback_is_advisory_and_collapse_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fallback = [
                self._write(root, 0, 0, include_response_only=False),
                self._write(root, 1, 250, include_response_only=False),
                self._write(root, 2, 500, include_response_only=False),
            ]
            report = evaluate_series(fallback, max_retrieval_rows=160, seed=7)
            self.assertEqual(report["status"], "advisory")
            self.assertEqual(
                report["checkpoints"][-1]["downstream_linear_probe"][
                    "response_control_strength"
                ],
                "pointwise_response_covariates_only_advisory",
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collapsed = np.zeros_like(self.degraded)
            paths = [
                self._write(root, 0, 0),
                self._write(root, 1, 250),
                self._write(root, 2, 500, degraded=collapsed),
            ]
            report = evaluate_series(paths, max_retrieval_rows=160, seed=7)
            self.assertEqual(report["status"], "fail")
            self.assertFalse(report["pass"])

    def test_explicit_response_encoder_without_target_probe_is_advisory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [
                self._write(root, 0, 0, include_target=False),
                self._write(root, 1, 250, include_target=False),
                self._write(root, 2, 500, include_target=False),
            ]
            report = evaluate_series(paths, max_retrieval_rows=160, seed=7)
            self.assertEqual(report["status"], "advisory")
            self.assertFalse(
                report["registered_status_gate"]["response_only_control_evaluable"]
            )

    def test_projection_state_is_fixed_and_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = load_latent_snapshot(self._write(root, 0, 0))
            state = fit_reference_projection(snapshot, max_points=120, seed=8)
            path = root / "fixed_projection.npz"
            save_projection_state(path, state)
            loaded = load_projection_state(path)
            np.testing.assert_array_equal(loaded.sample_id, state.sample_id)
            np.testing.assert_allclose(loaded.mean, state.mean)
            np.testing.assert_allclose(loaded.components, state.components)
            first = loaded.transform(self.dense[:5])
            shifted = loaded.transform(self.dense[:5] + 0.1)
            self.assertFalse(np.allclose(first, shifted))
            # Transforming a later epoch never mutates/refits the frozen basis.
            np.testing.assert_allclose(loaded.components, state.components)


if __name__ == "__main__":
    unittest.main()
