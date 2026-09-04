from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from workflows.sbi.p12_production_contract import BLIND_SCHEMA
from workflows.sbi.p12a_blind_evaluation_contract import (
    CONDITIONAL_STRATA,
    GATES,
    TRUTH_CONSTRUCTION_CONTRACT,
    TRUTH_CONSTRUCTION_IMPLEMENTATION_FILES,
    TRUTH_FREE_IDENTITY_INPUTS,
    shell_class_climatology,
)
from workflows.sbi.p12a_blind_proper_score import (
    clustered_mean_interval,
    gaussian_physical_log_prob,
)
from workflows.sbi.p12a_evaluate_blind import (
    evaluate_arrays,
    validate_open_state,
    validate_proper_score_report,
)
from workflows.sbi.p12a_open_blind import (
    AUTHORIZATION_FILENAME,
    OPEN_FILENAME,
    TRUTH_COMPLETE_FILENAME,
    build_authorization_marker,
    build_opened_marker,
    build_truth_complete_marker,
    validate_open_authorization,
    write_json_exclusive,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class P12ABlindOpeningTest(unittest.TestCase):
    @staticmethod
    def _record(path: Path) -> dict:
        return {"path": str(path.resolve()), "sha256": digest(path), "bytes": path.stat().st_size}

    def _frozen(self, root: Path) -> Path:
        sources = {}
        for key in ("p12a_candidate", "p12f_selection", "p10_deterministic_contract"):
            path = root / f"{key}.json"
            path.write_text(json.dumps({"pass": True}))
            sources[key] = self._record(path)
        manifests = []
        for schema in (
            "p12a-blind-base-context-v1",
            "p12a-blind-export-complete-v1",
            "p12-blind-cic-prediction-v1",
            "p12-blind-dtfe-prediction-v1",
        ):
            path = root / f"{schema}.json"
            payload = {
                "schema_version": schema, "phase": "ph001", "pass": True,
                "truth_files_read": [], "open_count": 0,
            }
            if schema == "p12a-blind-base-context-v1":
                array = root / "ph001_context.npz"
                np.savez(
                    array, parent_node_id=np.arange(2, dtype=np.int64),
                    core_id=np.zeros(2, dtype=np.int64), support_random=np.ones(2, dtype=bool),
                )
                payload.update({"array": str(array), "array_sha256": digest(array), "rows": 2})
            path.write_text(json.dumps(payload))
            manifests.append(self._record(path))
        frozen = root / "P12_BLIND_PREDICTIONS_FROZEN.json"
        frozen.write_text(json.dumps({
            "schema_version": BLIND_SCHEMA, "pass": True,
            "prediction_manifests": manifests,
            "truth_files_read": [], "open_count": 0, "sealed_phase_opened": False,
            **sources,
        }))
        return frozen

    def _contract(self, root: Path) -> Path:
        implementation = {}
        workflow = Path(__file__).resolve().parents[2] / "workflows/sbi"
        for name, filename in (
            ("blind_evaluator", "p12a_evaluate_blind.py"),
            ("proper_score_evaluator", "p12a_blind_proper_score.py"),
            ("one_open_guard", "p12a_open_blind.py"),
        ):
            implementation[name] = self._record(workflow / filename)
        sources = {}
        for key in ("candidate", "gaussian_baseline", "dataset_marker"):
            path = root / f"evaluation_{key}.json"
            path.write_text(json.dumps({"pass": True}))
            sources[key] = self._record(path)
        training = root / "training_sample.npz"
        np.savez(training, truth_eigenvalues=np.zeros((2, 3)))
        sources["training_sample"] = self._record(training)
        contract = root / "P12A_BLIND_EVALUATION_CONTRACT.json"
        contract.write_text(json.dumps({
            "schema_version": "p12a-blind-evaluation-contract-v1",
            "phase": "ph001", "open_count": 0, "sealed_phase_opened": False,
            "post_open_refit_allowed": False,
            "truth_files_read": [str(training.resolve())], "pass": True,
            "evaluation_implementation": implementation,
            "truth_construction_implementation": {
                name: self._record(path)
                for name, path in TRUTH_CONSTRUCTION_IMPLEMENTATION_FILES.items()
            },
            "truth_free_identity_inputs": {
                name: self._record(path)
                for name, path in TRUTH_FREE_IDENTITY_INPUTS.items()
            },
            "truth_construction_contract": TRUTH_CONSTRUCTION_CONTRACT,
            "gates": GATES, "conditional_strata": CONDITIONAL_STRATA,
            "class_threshold": 0.2, "bootstrap_unit": "authoritative core",
            "primary_proper_score": "physical joint log score on the frozen 50k audit rows",
            "shell_class_climatology": {
                str(index): {"probability_void_sheet_filament_knot": [0.25] * 4}
                for index in range(4)
            },
            **sources,
        }))
        return contract

    def _authorize(self, root: Path, frozen: Path, contract: Path) -> Path:
        authorization_path = root / AUTHORIZATION_FILENAME
        marker = build_authorization_marker(
            frozen_path=frozen, evaluation_contract_path=contract,
            explicit_authorization="OPEN_PH001_ONCE", deep=False,
        )
        write_json_exclusive(authorization_path, marker)
        return authorization_path

    def test_authorization_precedes_truth_and_is_exclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = self._frozen(root)
            contract = self._contract(root)
            with self.assertRaises(PermissionError):
                build_authorization_marker(
                    frozen_path=frozen, evaluation_contract_path=contract,
                    explicit_authorization="wrong", deep=False,
                )
            authorization = self._authorize(root, frozen, contract)
            payload = validate_open_authorization(
                authorization_path=authorization, frozen_path=frozen,
                evaluation_contract_path=contract,
            )
            self.assertEqual(payload["open_count"], 1)
            self.assertEqual(payload["truth_files_read"], [])
            self.assertFalse(payload["truth_materialization_complete"])
            with self.assertRaises(FileExistsError):
                write_json_exclusive(authorization, payload)

    def test_evaluator_refuses_sealed_or_unregistered_truth(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = self._frozen(root)
            contract = self._contract(root)
            authorization = self._authorize(root, frozen, contract)
            truth_array = root / "ph001_truth.npz"
            np.savez(truth_array, parent_node_id=np.arange(2), eigenvalues=np.zeros((2, 3)))
            truth_manifest = root / TRUTH_COMPLETE_FILENAME
            truth_payload = build_truth_complete_marker(
                authorization_path=authorization, truth_artifacts=[],
                truth_array=truth_array, rows=2,
            )
            write_json_exclusive(truth_manifest, truth_payload)
            opened = root / OPEN_FILENAME
            opened_payload = build_opened_marker(
                frozen_path=frozen, evaluation_contract_path=contract,
                authorization_path=authorization, truth_complete_path=truth_manifest,
            )
            write_json_exclusive(opened, opened_payload)
            validate_open_state(
                frozen_path=frozen, opened_path=opened, contract_path=contract,
                truth_manifest_path=truth_manifest,
            )
            proper = {
                "schema_version": "p12a-ph001-proper-score-v1", "phase": "ph001",
                "open_count": 1, "sealed_phase_opened": True,
                "post_open_refit_performed": False, "post_open_tuning_allowed": False,
                "frozen_predictions": self._record(frozen),
                "opened_marker": self._record(opened),
                "evaluation_contract": self._record(contract),
                "truth_manifest": self._record(truth_manifest),
            }
            validate_proper_score_report(
                proper, frozen_path=frozen, opened_path=opened,
                contract_path=contract, truth_manifest_path=truth_manifest,
            )
            proper["evaluation_contract"]["sha256"] = "0" * 64
            with self.assertRaises(RuntimeError):
                validate_proper_score_report(
                    proper, frozen_path=frozen, opened_path=opened,
                    contract_path=contract, truth_manifest_path=truth_manifest,
                )
            truth_array.write_bytes(b"tampered")
            with self.assertRaises(RuntimeError):
                validate_open_state(
                    frozen_path=frozen, opened_path=opened, contract_path=contract,
                    truth_manifest_path=truth_manifest,
                )

    def test_metric_gate_is_simultaneous(self):
        rows = 16
        parent = np.arange(rows, dtype=np.int64)
        anchor = np.linspace(-0.2, 0.19, rows)
        truth = np.column_stack((anchor, anchor + 1.0, anchor + 2.0))
        probability = np.zeros((rows, 4), dtype=np.float32)
        probability[:, 2] = 1.0
        summary = {
            "parent_node_id": parent,
            "shell": np.arange(rows) % 4,
            "ntilde_mpc3": np.linspace(0.1, 1.0, rows),
            "distance_to_support_boundary_mpc": np.linspace(1.0, 20.0, rows),
            "eigenvalue_q16": truth - 0.1,
            "eigenvalue_q84": truth + 0.1,
            "eigenvalue_q05": truth - 0.2,
            "eigenvalue_q95": truth + 0.2,
            "eigenvalue_mean": truth.copy(),
            "base_prediction_eigenvalues": truth + 0.05,
            "web_class_probability": probability,
        }
        audit = {
            "parent_node_id": parent[:4],
            "eigenvalue_draws": np.repeat(truth[:4, None, :], 8, axis=1),
        }
        contract = {
            "class_threshold": 0.2,
            "shell_class_climatology": {
                str(i): {"probability_void_sheet_filament_knot": [0.25] * 4}
                for i in range(4)
            },
            "gates": {
                "joint_eigenvalue_tarp_maximum": 0.05,
                "joint_eigengap_tarp_maximum": 0.05,
                "physical_rank_cdf_maximum": 0.05,
                "global_coverage_absolute_error_maximum": 0.4,
                "conditional_coverage_absolute_error_maximum": 0.4,
                "posterior_mean_lambda1_r2_delta_minimum": -0.02,
                "multiclass_brier_skill_minimum": 0.0,
                "fmpe_minus_gaussian_log_score_ci95_lower_minimum": 0.0,
            },
        }
        dependence = {
            "joint_eigenvalue_tarp": {"maximum_deviation": 0.01},
            "joint_eigengap_tarp": {"maximum_deviation": 0.01},
            "physical_rank_cdf_maximum_by_eigenvalue": [0.01] * 3,
            "physical_rank_cdf_maximum": 0.01,
        }
        with mock.patch(
            "workflows.sbi.p12a_evaluate_blind._tarp_and_ranks",
            return_value=dependence,
        ):
            result = evaluate_arrays(
                summary=summary, audit=audit, truth_parent=parent,
                truth_eigenvalues=truth, contract=contract,
                proper_score={"ci95": [0.1, 0.2]},
            )
        self.assertTrue(result["pass"])
        with self.assertRaises(RuntimeError):
            evaluate_arrays(
                summary=summary, audit=audit, truth_parent=parent[::-1],
                truth_eigenvalues=truth, contract=contract,
                proper_score={"ci95": [0.1, 0.2]},
            )

    def test_training_climatology_and_gaussian_score(self):
        truth = np.asarray([[0.0, 1.0, 2.0]] * 8)
        shell = np.repeat(np.arange(4), 2)
        climate = shell_class_climatology(truth, shell, np.ones(8))
        self.assertEqual(climate["0"]["probability_void_sheet_filament_knot"], [0.0, 0.0, 1.0, 0.0])
        gaussian = {
            "groups": {
                f"shell{s}_cap{c}": {"mean": [0.0] * 3, "covariance": np.eye(3).tolist()}
                for s in range(4) for c in (0, 1)
            }
        }
        score = gaussian_physical_log_prob(
            truth, truth, shell, np.arange(8) % 2, gaussian
        )
        np.testing.assert_allclose(score, -1.5 * np.log(2.0 * np.pi))
        interval = clustered_mean_interval(
            np.ones(8), shell, repeats=100, seed=2
        )
        np.testing.assert_allclose(interval["ci95"], [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
