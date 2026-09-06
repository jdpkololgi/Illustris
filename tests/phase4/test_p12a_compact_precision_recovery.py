from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from workflows.sbi import p12a_authorized_truth as frozen
from workflows.sbi import p12a_compact_precision_recovery as recovery
from workflows.sbi.p12a_compact_precision_recovery import (
    PrecisionRecoveryError,
    precision_aware_join,
    validate_diagnostic_payload,
)


class CompactPrecisionRecoveryTests(unittest.TestCase):
    def arguments(self):
        eigen = np.asarray(
            [[-0.1, np.float32(0.2), 0.5], [-0.2, 0.1, 0.6]],
            dtype=np.float32,
        )
        cweb = np.sum(eigen.astype(np.float64) > 0.2, axis=1).astype(np.uint8)
        return {
            "context_parent": np.asarray([1, 0], dtype=np.int64),
            "canonical_parent": np.arange(2, dtype=np.int64),
            "canonical_targetid": np.asarray([1, 2], dtype=np.int64),
            "annotated_targetid": np.asarray([1, 2], dtype=np.int64),
            "annotated_eigenvalues": eigen,
            "annotated_cweb": cweb,
            "expected_boundary_ambiguities": 1,
        }

    def test_only_diagnosed_boundary_ambiguity_is_accepted(self):
        targetid, eigen, cweb, audit = precision_aware_join(**self.arguments())
        np.testing.assert_array_equal(targetid, [2, 1])
        np.testing.assert_array_equal(eigen, self.arguments()["annotated_eigenvalues"][[1, 0]])
        np.testing.assert_array_equal(cweb, self.arguments()["annotated_cweb"][[1, 0]])
        self.assertEqual(audit["boundary_ambiguity_rows"], 1)
        self.assertEqual(audit["nonboundary_class_mismatch_rows"], 0)
        self.assertEqual(audit["float64_crosscheck_mismatch_rows"], 0)

    def test_nonboundary_disagreement_is_rejected(self):
        arguments = self.arguments()
        arguments["annotated_cweb"] = np.asarray([2, 0], dtype=np.uint8)
        with self.assertRaises(PrecisionRecoveryError):
            precision_aware_join(**arguments)

    def test_unregistered_boundary_count_is_rejected(self):
        arguments = self.arguments()
        arguments["expected_boundary_ambiguities"] = 0
        with self.assertRaises(PrecisionRecoveryError):
            precision_aware_join(**arguments)

    def test_identity_and_eigenvalue_guards_remain_hard(self):
        for key, value in (
            ("context_parent", np.asarray([0, 0])),
            ("canonical_parent", np.asarray([1, 0])),
            ("annotated_targetid", np.asarray([2, 1])),
            ("annotated_eigenvalues", np.asarray([[0.5, 0.1, -0.1], [-0.2, 0.1, 0.6]], dtype=np.float32)),
        ):
            arguments = self.arguments()
            arguments[key] = value
            with self.assertRaises(PrecisionRecoveryError):
                precision_aware_join(**arguments)

    def test_diagnostic_contract_rejects_any_broader_exception(self):
        payload = {
            "schema_version": "p12a-compact-closure-diagnostic-v1",
            "open_count": 1,
            "identity_join_exact": True,
            "posterior_scores_computed": False,
            "predictions_modified": False,
            "truth_outputs_modified": False,
            "closure": {
                "rows": frozen.EXPECTED_CONTEXT_ROWS,
                "source_dtype": "float32",
                "finite": True,
                "ordered": True,
                "source_values_preserved_by_float32": True,
                "float32_comparison_class_mismatches": 1,
                "float64_comparison_class_mismatches": 0,
                "rows_at_rounded_threshold": 1,
                "all_float32_mismatches_explained_by_threshold_rounding": True,
            },
        }
        validate_diagnostic_payload(payload)
        for key in (
            "float32_comparison_class_mismatches",
            "float64_comparison_class_mismatches",
            "rows_at_rounded_threshold",
        ):
            changed = copy.deepcopy(payload)
            changed["closure"][key] += 1
            with self.assertRaises(PrecisionRecoveryError):
                validate_diagnostic_payload(changed)

    def test_preopen_builder_is_still_byte_identical(self):
        root = Path(__file__).resolve().parents[2]
        contract = json.loads(
            (root / "docs/evidence/p12/P12A_BLIND_EVALUATION_CONTRACT.json").read_text()
        )
        for key in ("authorized_truth_wrapper", "compact_truth_slurm"):
            record = contract["truth_construction_implementation"][key]
            observed = hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()
            self.assertEqual(observed, record["sha256"])

    def test_boundary_with_wrong_native_class_is_rejected(self):
        arguments = self.arguments()
        arguments["annotated_cweb"] = np.asarray([1, 1], dtype=np.uint8)
        with self.assertRaises(PrecisionRecoveryError):
            precision_aware_join(**arguments)

    def test_float64_source_is_not_silently_quantized(self):
        arguments = self.arguments()
        arguments["annotated_eigenvalues"] = arguments["annotated_eigenvalues"].astype(np.float64)
        with self.assertRaises(PrecisionRecoveryError):
            precision_aware_join(**arguments)

    def test_resume_rejects_other_failed_job_before_claim(self):
        with mock.patch.object(recovery, "validate_exception", return_value={}), \
             mock.patch.object(frozen, "write_json_exclusive") as write:
            with self.assertRaises(PrecisionRecoveryError):
                recovery.claim_resume(submission_id="test", failed_job="1", blocked_dispatcher="57928546")
            write.assert_not_called()

    def test_resume_record_rejects_broken_dependency(self):
        with mock.patch.object(recovery, "_validate_claim", return_value={}), \
             mock.patch.object(recovery, "_load", return_value={
                 "schema_version": recovery.JOB_SCHEMA, "slurm_job_id": "123"}), \
             mock.patch.object(frozen, "write_json_exclusive") as write:
            with self.assertRaises(PrecisionRecoveryError):
                recovery.record_resume_job(submission_id="test", job="postopen_dispatch", job_id="124", dependency_job_id="999")
            write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
