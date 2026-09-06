from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from workflows.sbi.p12_production_contract import BLIND_SCHEMA
from workflows.sbi.p12a_blind_energy_score import (
    clustered_mean_interval,
    gaussian_samples,
    joint_energy_score,
)
from workflows.sbi.p12a_blind_evaluation_contract import (
    IMPLEMENTATION_FILES,
    SCHEMA as CONTRACT_SCHEMA,
    installed_distribution_versions,
    shell_class_climatology,
    validate_runtime_environment,
)
from workflows.sbi.p12a_evaluate_blind import (
    PROPER_SCORE_SCHEMA,
    RESULT_SCHEMA,
    evaluate_arrays,
    validate_evaluation_report,
    validate_open_state,
    validate_proper_score_report,
)
from workflows.sbi.p12a_open_blind import (
    AUTHORIZATION_FILENAME,
    OPEN_FILENAME,
    TRUTH_COMPLETE_FILENAME,
    _generator_record_matches,
    build_authorization_marker,
    build_opened_marker,
    build_truth_complete_marker,
    validate_open_authorization,
)
from workflows.sbi.p12a_immutable_io import write_json_exclusive
from workflows.sbi.p12a_plot_blind_evaluation import render


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class P12ABlindOpeningTest(unittest.TestCase):
    def test_runtime_inventory_ignores_distribution_without_metadata(self) -> None:
        malformed = mock.Mock(metadata=None, version="unknown")
        valid = mock.Mock(metadata={"Name": "Example-Package"}, version="1.2.3")
        with mock.patch(
            "workflows.sbi.p12a_blind_evaluation_contract.metadata.distributions",
            return_value=[malformed, valid],
        ):
            self.assertEqual(
                installed_distribution_versions(), {"example-package": "1.2.3"}
            )

    def test_historical_generator_source_is_content_addressed(self) -> None:
        source = IMPLEMENTATION_FILES["blind_inference"].resolve()
        blob = b"historical generator bytes\n"
        record = {
            "path": str(source),
            "sha256": hashlib.sha256(blob).hexdigest(),
            "bytes": len(blob),
        }
        completed = mock.Mock(returncode=0)
        with mock.patch(
            "workflows.sbi.p12a_open_blind.subprocess.run",
            return_value=completed,
        ), mock.patch(
            "workflows.sbi.p12a_open_blind.subprocess.check_output",
            return_value=blob,
        ):
            self.assertTrue(
                _generator_record_matches(record, source, "a" * 40)
            )
            bad = dict(record, sha256="0" * 64)
            self.assertFalse(
                _generator_record_matches(bad, source, "a" * 40)
            )
            self.assertFalse(
                _generator_record_matches(record, source, "not-a-revision")
            )

    @staticmethod
    def _record(path: Path) -> dict:
        return {
            "path": str(path.resolve()),
            "sha256": digest(path),
            "bytes": path.stat().st_size,
        }

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
                "schema_version": schema,
                "phase": "ph001",
                "pass": True,
                "truth_files_read": [],
                "open_count": 0,
            }
            if schema == "p12a-blind-base-context-v1":
                array = root / "ph001_context.npz"
                np.savez(
                    array,
                    parent_node_id=np.arange(2, dtype=np.int64),
                    core_id=np.zeros(2, dtype=np.int64),
                    support_random=np.ones(2, dtype=bool),
                )
                payload.update(
                    {"array": str(array), "array_sha256": digest(array), "rows": 2}
                )
            path.write_text(json.dumps(payload))
            manifests.append(self._record(path))
        frozen = root / "P12_BLIND_PREDICTIONS_FROZEN.json"
        frozen.write_text(
            json.dumps(
                {
                    "schema_version": BLIND_SCHEMA,
                    "pass": True,
                    "prediction_manifests": manifests,
                    "truth_files_read": [],
                    "open_count": 0,
                    "sealed_phase_opened": False,
                    **sources,
                }
            )
        )
        return frozen

    @staticmethod
    def _contract(root: Path) -> Path:
        contract = root / "P12A_BLIND_EVALUATION_CONTRACT.json"
        contract.write_text(
            json.dumps(
                {
                    "schema_version": CONTRACT_SCHEMA,
                    "phase": "ph001",
                    "open_count": 0,
                    "sealed_phase_opened": False,
                    "post_open_refit_allowed": False,
                    "pass": True,
                }
            )
        )
        return contract

    def _authorize(self, root: Path, frozen: Path, contract: Path) -> Path:
        authorization_path = root / AUTHORIZATION_FILENAME
        with mock.patch(
            "workflows.sbi.p12a_open_blind.validate_evaluation_contract",
            return_value={},
        ):
            marker = build_authorization_marker(
                frozen_path=frozen,
                evaluation_contract_path=contract,
                explicit_authorization="OPEN_PH001_ONCE",
                deep=False,
            )
        marker["deep_prediction_revalidation"] = {
            "performed": True,
            "prediction_manifest_count": 4,
            "summary_rows": 4_897_905,
            "audit_rows": 50_000,
            "posterior_draws": 512,
            "shards": 4,
            "pass": True,
        }
        write_json_exclusive(authorization_path, marker)
        return authorization_path

    def test_authorization_precedes_truth_and_is_exclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = self._frozen(root)
            contract = self._contract(root)
            with self.assertRaises(PermissionError):
                build_authorization_marker(
                    frozen_path=frozen,
                    evaluation_contract_path=contract,
                    explicit_authorization="wrong",
                    deep=False,
                )
            authorization = self._authorize(root, frozen, contract)
            with mock.patch(
                "workflows.sbi.p12a_open_blind.validate_evaluation_contract",
                return_value={},
            ):
                payload = validate_open_authorization(
                    authorization_path=authorization,
                    frozen_path=frozen,
                    evaluation_contract_path=contract,
                )
            self.assertEqual(payload["open_count"], 1)
            self.assertEqual(payload["truth_files_read"], [])
            self.assertFalse(payload["truth_materialization_complete"])
            with self.assertRaises(FileExistsError):
                write_json_exclusive(authorization, payload)

    def test_runtime_guard_rejects_mutable_python_overrides(self):
        validate_runtime_environment()
        with mock.patch.dict(os.environ, {"PYTHONPATH": "/tmp/poison"}, clear=False):
            with self.assertRaises(RuntimeError):
                validate_runtime_environment()

    def test_transitive_runtime_dependencies_are_frozen(self):
        required = {
            "workflows_package_init",
            "sbi_package_init",
            "shared_package_init",
            "shared_config_paths",
            "abacus_tweb_package_init",
            "training_contract_dependency",
            "graph_patch_dependency",
            "epoch_training_dependency",
            "calibration_diagnostics_dependency",
        }
        self.assertTrue(required.issubset(IMPLEMENTATION_FILES))
        for key in required:
            self.assertTrue(IMPLEMENTATION_FILES[key].is_file(), key)

    def test_open_transition_binds_truth_and_rejects_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = self._frozen(root)
            contract = self._contract(root)
            authorization = self._authorize(root, frozen, contract)
            truth_array = root / "ph001_truth.npz"
            np.savez(
                truth_array,
                parent_node_id=np.arange(2),
                eigenvalues=np.zeros((2, 3)),
            )
            truth_manifest = root / TRUTH_COMPLETE_FILENAME
            with mock.patch(
                "workflows.sbi.p12a_open_blind.validate_evaluation_contract",
                return_value={},
            ):
                truth_payload = build_truth_complete_marker(
                    authorization_path=authorization,
                    truth_artifacts=[],
                    truth_array=truth_array,
                    rows=2,
                )
                write_json_exclusive(truth_manifest, truth_payload)
                opened = root / OPEN_FILENAME
                opened_payload = build_opened_marker(
                    frozen_path=frozen,
                    evaluation_contract_path=contract,
                    authorization_path=authorization,
                    truth_complete_path=truth_manifest,
                )
                write_json_exclusive(opened, opened_payload)
                with mock.patch(
                    "workflows.sbi.p12a_evaluate_blind.validate_evaluation_implementation",
                    return_value=None,
                ), mock.patch(
                    "workflows.sbi.p12a_evaluate_blind.validate_evaluation_contract",
                    return_value={},
                ):
                    validate_open_state(
                        frozen_path=frozen,
                        opened_path=opened,
                        contract_path=contract,
                        truth_manifest_path=truth_manifest,
                    )
                with mock.patch(
                    "workflows.sbi.p12a_evaluate_blind.validate_evaluation_contract",
                    side_effect=RuntimeError("runtime fingerprint drift"),
                ), self.assertRaisesRegex(RuntimeError, "runtime fingerprint drift"):
                    validate_open_state(
                        frozen_path=frozen,
                        opened_path=opened,
                        contract_path=contract,
                        truth_manifest_path=truth_manifest,
                    )
            # A metadata-only proper-score report is deliberately insufficient:
            # raw sample and Gaussian replay inputs are mandatory.
            shallow = {
                "schema_version": PROPER_SCORE_SCHEMA,
                "phase": "ph001",
                "open_count": 1,
                "sealed_phase_opened": True,
                "post_open_refit_performed": False,
                "post_open_tuning_allowed": False,
                "unnormalized_fmpe_log_score_used": False,
                "lower_is_better": True,
                "comparison": "gaussian_minus_fmpe; positive favours FMPE",
                "frozen_predictions": self._record(frozen),
                "opened_marker": self._record(opened),
                "evaluation_contract": self._record(contract),
                "truth_manifest": self._record(truth_manifest),
            }
            with self.assertRaises(RuntimeError):
                validate_proper_score_report(
                    shallow,
                    frozen_path=frozen,
                    opened_path=opened,
                    contract_path=contract,
                    truth_manifest_path=truth_manifest,
                )
            truth_array.write_bytes(b"tampered")
            with mock.patch(
                "workflows.sbi.p12a_evaluate_blind.validate_evaluation_implementation",
                return_value=None,
            ), mock.patch(
                "workflows.sbi.p12a_evaluate_blind.validate_evaluation_contract",
                return_value={},
            ), mock.patch(
                "workflows.sbi.p12a_open_blind.validate_evaluation_contract",
                return_value={},
            ), self.assertRaises(RuntimeError):
                validate_open_state(
                    frozen_path=frozen,
                    opened_path=opened,
                    contract_path=contract,
                    truth_manifest_path=truth_manifest,
                )

    def test_existing_evaluation_requires_full_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {
                key: root / f"{key}.json"
                for key in (
                    "frozen_predictions",
                    "opened_marker",
                    "evaluation_contract",
                    "truth_manifest",
                    "proper_score_report",
                )
            }
            for path in paths.values():
                path.write_text("{}")
            report_path = root / "evaluation.json"
            report = {
                "schema_version": RESULT_SCHEMA,
                "created_utc": "frozen-time",
                "git_revision": "frozen-revision",
                "release_status": "green",
                **{key: self._record(path) for key, path in paths.items()},
            }
            report_path.write_text(json.dumps(report))
            with mock.patch(
                "workflows.sbi.p12a_evaluate_blind.recompute_evaluation_report",
                return_value=dict(report),
            ) as replay:
                self.assertEqual(
                    validate_evaluation_report(
                        report_path,
                        evaluation_contract_path=paths["evaluation_contract"],
                    ),
                    report,
                )
                replay.assert_called_once()
            tampered = {**report, "release_status": "blocked"}
            report_path.write_text(json.dumps(tampered))
            with mock.patch(
                "workflows.sbi.p12a_evaluate_blind.recompute_evaluation_report",
                return_value=dict(report),
            ), self.assertRaises(RuntimeError):
                validate_evaluation_report(
                    report_path,
                    evaluation_contract_path=paths["evaluation_contract"],
                )

    def test_metric_gate_and_green_amber_blocked_tree(self):
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
            "quality_bitmask": (np.arange(rows) % 4 == 3).astype(np.uint16),
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
                "nonsparse_conditional_coverage_absolute_error_maximum": 0.4,
                "sparse_shell_green_absolute_error_maximum": 0.4,
                "sparse_shell_release_absolute_error_maximum": 0.4,
                "posterior_mean_lambda1_r2_delta_minimum": -0.02,
                "multiclass_brier_skill_minimum": 0.0,
                "gaussian_minus_fmpe_energy_score_ci95_lower_minimum": 0.0,
            },
            "evaluation_protocol": {
                "tarp_seed": 3,
                "tarp_repetitions": 20,
                "eigengap_tarp_seed_offset": 1,
                "rank_seed_offset": 2,
                "rank_repetitions": 1,
            },
        }
        dependence = {
            "joint_eigenvalue_tarp": {
                "maximum_deviation": 0.01,
                "replicate_p90_maximum_deviation": 0.01,
            },
            "joint_eigengap_tarp": {
                "maximum_deviation": 0.01,
                "replicate_p90_maximum_deviation": 0.01,
            },
            "physical_rank_cdf_maximum_by_eigenvalue": [0.01] * 3,
            "physical_rank_cdf_maximum": 0.01,
        }
        with mock.patch(
            "workflows.sbi.p12a_evaluate_blind._tarp_and_ranks",
            return_value=dependence,
        ):
            result = evaluate_arrays(
                summary=summary,
                audit=audit,
                truth_parent=parent,
                truth_eigenvalues=truth,
                contract=contract,
                proper_score={"ci95": [0.1, 0.2], "pass": True},
            )
        self.assertTrue(result["pass"])
        self.assertEqual(result["release_status"], "green")

        # A visually benign primary TARP curve cannot mask instability across
        # the frozen reference-point seeds used by the registered decision.
        unstable_dependence = json.loads(json.dumps(dependence))
        unstable_dependence["joint_eigenvalue_tarp"][
            "replicate_p90_maximum_deviation"
        ] = 0.06
        with mock.patch(
            "workflows.sbi.p12a_evaluate_blind._tarp_and_ranks",
            return_value=unstable_dependence,
        ):
            unstable = evaluate_arrays(
                summary=summary,
                audit=audit,
                truth_parent=parent,
                truth_eigenvalues=truth,
                contract=contract,
                proper_score={"ci95": [0.1, 0.2], "pass": True},
            )
        self.assertFalse(unstable["gates"]["joint_eigenvalue_tarp"])
        self.assertFalse(unstable["pass"])
        self.assertEqual(unstable["release_status"], "blocked")

        with self.assertRaises(RuntimeError):
            evaluate_arrays(
                summary=summary,
                audit=audit,
                truth_parent=parent[::-1],
                truth_eigenvalues=truth,
                contract=contract,
                proper_score={"ci95": [0.1, 0.2], "pass": True},
            )

    def test_training_climatology_and_sample_energy_score(self):
        truth = np.asarray([[0.0, 1.0, 2.0]] * 8)
        shell = np.repeat(np.arange(4), 2)
        climate = shell_class_climatology(truth, shell, np.ones(8))
        self.assertEqual(
            climate["0"]["probability_void_sheet_filament_knot"],
            [0.0, 0.0, 1.0, 0.0],
        )
        gaussian = {
            "groups": {
                f"shell{s}_cap{c}": {
                    "mean": [0.0] * 3,
                    "covariance": np.eye(3).tolist(),
                }
                for s in range(4)
                for c in (0, 1)
            }
        }
        draws = gaussian_samples(
            base=truth,
            shell=shell,
            cap=np.arange(8) % 2,
            gaussian=gaussian,
            draws=16,
            seed=7,
        )
        replay = gaussian_samples(
            base=truth,
            shell=shell,
            cap=np.arange(8) % 2,
            gaussian=gaussian,
            draws=16,
            seed=7,
        )
        np.testing.assert_array_equal(draws, replay)
        score = joint_energy_score(draws, truth, pairing_offset=9)
        self.assertEqual(score.shape, (8,))
        self.assertTrue(np.all(np.isfinite(score)))
        interval = clustered_mean_interval(
            np.ones(8), shell, repeats=100, seed=2
        )
        np.testing.assert_allclose(interval["ci95"], [1.0, 1.0])

    def test_proper_score_report_is_recomputed_from_raw_draws(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = root / "frozen.json"
            opened = root / "opened.json"
            contract_path = root / "contract.json"
            truth_manifest = root / "truth.json"
            for path in (frozen, opened, contract_path, truth_manifest):
                path.write_text("{}")
            rows, draws = 4, 8
            parent = np.arange(rows, dtype=np.int64)
            core = np.asarray([0, 0, 1, 1], dtype=np.int64)
            shell = np.asarray([0, 0, 1, 1], dtype=np.int8)
            cap = np.asarray([0, 1, 0, 1], dtype=np.uint8)
            truth = np.asarray(
                [[-0.2, 0.1, 0.4], [-0.1, 0.2, 0.5], [0.0, 0.3, 0.6], [0.1, 0.4, 0.7]]
            )
            base = truth + 0.02
            fmpe_draws = np.repeat(truth[:, None, :], draws, axis=1)
            fmpe_draws += np.linspace(-0.01, 0.01, draws)[None, :, None]
            gaussian_contract = {
                "groups": {
                    f"shell{s}_cap{c}": {
                        "mean": [0.0, 0.0, 0.0],
                        "covariance": (np.eye(3) * 0.04).tolist(),
                    }
                    for s in range(4)
                    for c in (0, 1)
                }
            }
            gaussian_path = root / "gaussian.json"
            gaussian_path.write_text(json.dumps(gaussian_contract))
            protocol = {
                "audit_rows": rows,
                "posterior_draws": draws,
                "energy_pairing_offset": 5,
                "gaussian_draw_seed": 11,
                "gaussian_ordering_transform": "none-test",
                "bootstrap_repetitions": 100,
                "bootstrap_seed": 12,
                "bootstrap_unit": "authoritative core",
            }
            gaussian_draws = gaussian_samples(
                base=base,
                shell=shell,
                cap=cap,
                gaussian=gaussian_contract,
                draws=draws,
                seed=protocol["gaussian_draw_seed"],
            )
            fmpe_score = joint_energy_score(
                fmpe_draws, truth, pairing_offset=protocol["energy_pairing_offset"]
            )
            gaussian_score = joint_energy_score(
                gaussian_draws, truth, pairing_offset=protocol["energy_pairing_offset"]
            )
            difference = gaussian_score - fmpe_score
            score_path = root / "scores.npz"
            np.savez_compressed(
                score_path,
                parent_node_id=parent,
                core_id=core,
                fmpe_joint_energy_score=fmpe_score,
                gaussian_joint_energy_score=gaussian_score,
                gaussian_minus_fmpe=difference,
            )
            bootstrap = clustered_mean_interval(
                difference,
                core,
                repeats=protocol["bootstrap_repetitions"],
                seed=protocol["bootstrap_seed"],
            )
            contract = {
                "evaluation_protocol": protocol,
                "canonical_outputs": {
                    "energy_score_report": str(root / "report.json"),
                    "energy_score_array": str(score_path),
                },
                "gaussian_baseline": self._record(gaussian_path),
                "gates": {
                    "gaussian_minus_fmpe_energy_score_ci95_lower_minimum": -10.0
                },
            }
            report = {
                "schema_version": PROPER_SCORE_SCHEMA,
                "phase": "ph001",
                "open_count": 1,
                "sealed_phase_opened": True,
                "post_open_refit_performed": False,
                "post_open_tuning_allowed": False,
                "unnormalized_fmpe_log_score_used": False,
                "lower_is_better": True,
                "comparison": "gaussian_minus_fmpe; positive favours FMPE",
                "rows": rows,
                "posterior_draws": draws,
                "energy_pairing_offset": protocol["energy_pairing_offset"],
                "gaussian_draw_seed": protocol["gaussian_draw_seed"],
                "gaussian_ordering_transform": protocol["gaussian_ordering_transform"],
                "fmpe_mean_joint_energy_score": float(fmpe_score.mean()),
                "gaussian_mean_joint_energy_score": float(gaussian_score.mean()),
                **bootstrap,
                "score_array": self._record(score_path),
                "frozen_predictions": self._record(frozen),
                "opened_marker": self._record(opened),
                "evaluation_contract": self._record(contract_path),
                "truth_manifest": self._record(truth_manifest),
                "pass": True,
            }
            result = validate_proper_score_report(
                report,
                frozen_path=frozen,
                opened_path=opened,
                contract_path=contract_path,
                truth_manifest_path=truth_manifest,
                contract=contract,
                audit_parent=parent,
                audit_core=core,
                audit_draws=fmpe_draws,
                audit_truth=truth,
                gaussian_base=base,
                audit_shell=shell,
                audit_cap=cap,
            )
            self.assertTrue(result["pass"])
            with np.load(score_path) as archive:
                arrays = {name: archive[name] for name in archive.files}
            arrays["gaussian_minus_fmpe"] = arrays["gaussian_minus_fmpe"] + 1e-3
            np.savez_compressed(score_path, **arrays)
            report["score_array"] = self._record(score_path)
            with self.assertRaises(RuntimeError):
                validate_proper_score_report(
                    report,
                    frozen_path=frozen,
                    opened_path=opened,
                    contract_path=contract_path,
                    truth_manifest_path=truth_manifest,
                    contract=contract,
                    audit_parent=parent,
                    audit_core=core,
                    audit_draws=fmpe_draws,
                    audit_truth=truth,
                    gaussian_base=base,
                    audit_shell=shell,
                    audit_cap=cap,
                )

    def test_blind_summary_plot_is_deterministic(self):
        curve = {
            "alpha": [0.0, 0.5, 1.0],
            "expected_coverage_probability": [0.0, 0.5, 1.0],
        }
        point = {"lambda1_lambda2_lambda3": [{"r2": 0.4}] * 3}
        report = {
            "dependence": {
                "joint_eigenvalue_tarp": curve,
                "joint_eigengap_tarp": curve,
            },
            "coverage68": [0.68, 0.68, 0.68],
            "coverage90": [0.90, 0.90, 0.90],
            "conditional_coverage": {
                "strata": {
                    "redshift_shell": {
                        "0": {"maximum_absolute_error": 0.01},
                        "3": {"maximum_absolute_error": 0.04},
                    }
                }
            },
            "posterior_mean_r2": [0.5, 0.6, 0.7],
            "base_unet_r2": [0.49, 0.59, 0.69],
            "classical_deterministic": {
                "cic": {"train_affine_ordered": point},
                "dtfe": {"train_affine_ordered": point},
            },
            "release_status": "amber",
        }
        with tempfile.TemporaryDirectory() as temporary:
            left = Path(temporary) / "left.png"
            right = Path(temporary) / "right.png"
            render(report, left)
            render(report, right)
            self.assertEqual(digest(left), digest(right))


if __name__ == "__main__":
    unittest.main()
