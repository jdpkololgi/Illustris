from __future__ import annotations

import hashlib
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
    assert_ph001_sealed_payload,
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
            boundary_distance_mpc_h=np.array([6.0, 15.0, 25.0]),
            response_covariate=np.array([0.5, -3.0, 0.0]),
            posterior_width=np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [2.0, 0.1, 0.1]]),
            response_training_range=(-1.0, 1.0),
            prior_width_threshold=np.array([1.0, 1.0, 1.0]),
            boundary_r_mpc=10.35,
            boundary_2r_mpc=20.70,
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
        assert_ph001_sealed_payload(
            {"truth_files_read": ["ph006 density/T-web"], "open_count": 0}
        )
        with self.assertRaises(PermissionError):
            assert_ph001_sealed_payload(
                {"truth_files_read": ["/truth/ph001/tweb.h5"], "open_count": 0}
            )
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

    def test_blind_freeze_requires_complete_exact_row_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate.json"
            selection = root / "selection.json"
            deterministic = root / "p10.json"
            checkpoint = root / "fmpe.pt"
            checkpoint.write_bytes(b"checkpoint")
            candidate.write_text(json.dumps({
                "schema_version": P12A_SCHEMA,
                "posterior_draws": 512,
                "artifacts": {
                    "checkpoint": {"sha256": hashlib.sha256(b"checkpoint").hexdigest()}
                },
                "truth_files_read": [],
                "open_count": 0,
                "pass": True,
            }))
            selection.write_text(json.dumps({
                "schema_version": "p12f-no-field-finalist-v1",
                "truth_files_read": ["ph006 density/T-web"],
                "open_count": 0,
                "pass": True,
            }))
            deterministic.write_text(json.dumps({
                "schema_version": "p10-blind-evaluation-frozen-marker-v1",
                "phase": "ph001",
                "truth_files_read": [],
                "open_count": 0,
                "pass": True,
            }))

            parent = np.arange(6, dtype=np.int64)
            core = np.repeat([1, 2], 3).astype(np.int64)
            context_array = root / "context.npz"
            np.savez(
                context_array,
                parent_node_id=parent,
                core_id=core,
                support_random=np.ones(6, bool),
            )
            digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            context = root / "context.json"
            context.write_text(json.dumps({
                "schema_version": "p12a-blind-base-context-v1", "phase": "ph001",
                "rows": 6, "array": str(context_array), "array_sha256": digest(context_array),
                "truth_files_read": [], "open_count": 0, "pass": True,
            }))
            shard_summary = root / "summary.npz"
            np.savez(shard_summary, parent_node_id=parent, core_id=core)
            audit = root / "audit.npz"
            np.savez(audit, parent_node_id=parent[:2])
            shard = root / "shard.json"
            shard.write_text(json.dumps({
                "schema_version": "p12a-blind-posterior-shard-v1", "phase": "ph001",
                "draws": 512, "candidate_sha256": digest(candidate),
                "checkpoint_sha256": digest(checkpoint), "context_sha256": digest(context_array),
                "summary": str(shard_summary), "summary_sha256": digest(shard_summary),
                "audit_draws": str(audit), "audit_draws_sha256": digest(audit),
                "truth_files_read": [], "open_count": 0, "pass": True,
            }))
            complete = root / "complete.json"
            complete.write_text(json.dumps({
                "schema_version": "p12a-blind-export-complete-v1", "phase": "ph001",
                "rows": 6, "context": {"path": str(context_array), "sha256": digest(context_array)},
                "shards": [{"path": str(shard), "sha256": digest(shard)}],
                "truth_files_read": [], "open_count": 0, "pass": True,
            }))
            classical = []
            for estimator in ("cic", "dtfe"):
                array = root / f"{estimator}.npz"
                np.savez(array, parent_node_id=parent, core_id=core, support_random=np.ones(6, bool))
                manifest = root / f"{estimator}.json"
                manifest.write_text(json.dumps({
                    "schema_version": f"p12-blind-{estimator}-prediction-v1",
                    "phase": "ph001", "estimator": estimator, "prediction": str(array),
                    "prediction_sha256": digest(array), "truth_files_read": [],
                    "open_count": 0, "pass": True,
                }))
                classical.append(manifest)
            predictions = [context, complete, *classical]
            marker = freeze_blind_predictions(
                candidate_marker=candidate,
                method_selection_marker=selection,
                prediction_manifests=predictions,
                deterministic_contract=deterministic,
            )
            self.assertEqual(marker["schema_version"], BLIND_SCHEMA)
            array = root / "cic.npz"
            np.savez(array, parent_node_id=parent[::-1], core_id=core, support_random=np.ones(6, bool))
            with self.assertRaises(RuntimeError):
                freeze_blind_predictions(
                    candidate_marker=candidate,
                    method_selection_marker=selection,
                    prediction_manifests=predictions,
                    deterministic_contract=deterministic,
                )


if __name__ == "__main__":
    unittest.main()
