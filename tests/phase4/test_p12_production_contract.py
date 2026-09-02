from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from workflows.sbi.p12_production_contract import (
    BLIND_SCHEMA,
    P12A_SCHEMA,
    QUALITY_BOUNDARY_LT_2R7,
    QUALITY_BOUNDARY_LT_R7,
    QUALITY_PRIOR_DOMINATED_WIDTH,
    QUALITY_RESPONSE_OOD,
    QUALITY_SPARSE_SHELL,
    assert_truth_free_payload,
    build_opened_marker,
    deterministic_audit_subset,
    fit_shell_cap_gaussian,
    freeze_blind_predictions,
    posterior_summaries,
    quality_bitmask,
)


class P12ProductionContractTest(unittest.TestCase):
    def test_posterior_summary_is_ordered_and_normalized(self):
        rng = np.random.default_rng(2)
        draws = np.sort(rng.normal(size=(9, 32, 3)), axis=-1).astype(np.float32)
        summary = posterior_summaries(draws)
        self.assertEqual(summary["eigenvalue_mean"].shape, (9, 3))
        np.testing.assert_allclose(summary["web_class_probability"].sum(axis=1), 1.0)
        self.assertTrue(np.all(summary["web_class_entropy_nats"] >= 0))
        bad = draws.copy()
        bad[0, 0] = [1.0, -1.0, 2.0]
        with self.assertRaises(ValueError):
            posterior_summaries(bad)

    def test_quality_bits_are_independent_and_nested_at_boundary(self):
        bits = quality_bitmask(
            redshift=np.array([0.46, 0.2, 0.2]),
            boundary_distance_mpc_h=np.array([6.0, 10.0, 20.0]),
            response_covariate=np.array([0.5, -3.0, 0.0]),
            posterior_width=np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [2.0, 0.1, 0.1]]),
            response_training_range=(-1.0, 1.0),
            prior_width_threshold=np.array([1.0, 1.0, 1.0]),
        )
        self.assertTrue(bits[0] & QUALITY_SPARSE_SHELL)
        self.assertTrue(bits[0] & QUALITY_BOUNDARY_LT_R7)
        self.assertTrue(bits[0] & QUALITY_BOUNDARY_LT_2R7)
        self.assertTrue(bits[1] & QUALITY_BOUNDARY_LT_2R7)
        self.assertTrue(bits[1] & QUALITY_RESPONSE_OOD)
        self.assertTrue(bits[2] & QUALITY_PRIOR_DOMINATED_WIDTH)

    def test_audit_subset_is_reproducible_and_stratified(self):
        parent = np.arange(1000, dtype=np.int64)
        shell = parent % 4
        cap = parent % 2
        boundary = np.linspace(0.0, 100.0, len(parent))
        first = deterministic_audit_subset(parent, shell, cap, boundary, maximum=120, seed=7)
        second = deterministic_audit_subset(parent, shell, cap, boundary, maximum=120, seed=7)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 120)
        self.assertEqual(set(np.unique(shell[first])), {0, 1, 2, 3})

    def test_train_only_gaussian_is_positive_definite(self):
        rng = np.random.default_rng(4)
        result = fit_shell_cap_gaussian(
            rng.normal(size=(80, 3)),
            np.repeat(np.arange(4), 20),
            np.tile(np.repeat([0, 1], 10), 4),
        )
        self.assertEqual(result["fit_scope"], "training phases only")
        for group in result["groups"].values():
            self.assertTrue(np.all(np.linalg.eigvalsh(group["covariance"]) > 0))

    def test_truth_free_guard_and_single_open_transition(self):
        assert_truth_free_payload({"truth_files_read": [], "open_count": 0})
        with self.assertRaises(PermissionError):
            assert_truth_free_payload({"truth_files_read": ["truth.h5"], "open_count": 0})
        with self.assertRaises(PermissionError):
            assert_truth_free_payload({"truth_files_read": [], "open_count": 1})
        with tempfile.TemporaryDirectory() as tmp:
            truth = Path(tmp) / "truth.bin"
            truth.write_bytes(b"frozen")
            frozen = {
                "schema_version": BLIND_SCHEMA,
                "truth_files_read": [],
                "open_count": 0,
                "pass": True,
            }
            with self.assertRaises(PermissionError):
                build_opened_marker(
                    frozen,
                    truth_artifacts=[truth],
                    explicit_authorization="not-authorized",
                )
            opened = build_opened_marker(
                frozen,
                truth_artifacts=[truth],
                explicit_authorization="OPEN_PH001_ONCE",
            )
            self.assertEqual(opened["open_count"], 1)
            self.assertFalse(opened["post_open_tuning_allowed"])

    def test_blind_freeze_rejects_stale_or_open_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate.json"
            selection = root / "selection.json"
            prediction = root / "prediction.json"
            deterministic = root / "p10.json"
            candidate.write_text(json.dumps({
                "schema_version": P12A_SCHEMA,
                "truth_files_read": [],
                "open_count": 0,
                "pass": True,
            }))
            selection.write_text(json.dumps({
                "schema_version": "p12f-no-field-finalist-v1",
                "truth_files_read": [],
                "open_count": 0,
                "pass": True,
            }))
            prediction.write_text(json.dumps({
                "schema_version": "p12a-blind-shard-v1",
                "truth_files_read": [],
                "open_count": 0,
                "pass": True,
            }))
            deterministic.write_text(json.dumps({
                "truth_files_read": [],
                "open_count": 0,
                "pass": True,
            }))
            marker = freeze_blind_predictions(
                candidate_marker=candidate,
                method_selection_marker=selection,
                prediction_manifests=[prediction],
                deterministic_contract=deterministic,
            )
            self.assertEqual(marker["schema_version"], BLIND_SCHEMA)
            prediction.write_text(json.dumps({
                "schema_version": "p12a-blind-shard-v1",
                "truth_files_read": ["truth.npy"],
                "open_count": 1,
                "pass": True,
            }))
            with self.assertRaises(PermissionError):
                freeze_blind_predictions(
                    candidate_marker=candidate,
                    method_selection_marker=selection,
                    prediction_manifests=[prediction],
                    deterministic_contract=deterministic,
                )


if __name__ == "__main__":
    unittest.main()
