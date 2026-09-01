import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from workflows.abacus_tweb.p11_jepa_supervisor_guard import (
    DEFAULT_CONTRACT,
    LATENT_DIAGNOSTIC_FILENAME,
    REPO_ROOT,
    validate_canary_marker,
    validate_latent_diagnostic_gate,
    validate_preallocation,
    validate_stored_complete,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


class P11JEPASupervisorGuardTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temporary.name) / "canary_v1/jepa/seed_42"
        self.run_dir.mkdir(parents=True)
        self.checkpoint = self.run_dir / "p11_jepa_checkpoint.pt"
        self.checkpoint.write_bytes(b"checkpoint-placeholder")
        self.digest = "a" * 64
        self.marker = {
            "schema_version": "p11-jepa-technical-canary-v2",
            "arm": "jepa",
            "pass": True,
            "global_step": 500,
            "teacher_gradient_free": True,
            "ph001_opened": False,
            "data_contract_aggregate_sha256": self.digest,
            "checkpoint": str(self.checkpoint),
            "gates": {
                "finite_loss": True,
                "finite_gradient_norm": True,
                "checkpoint_reload_valid": True,
            },
        }
        self.run_id = "canary_v1/jepa/seed_42"
        self.latent_exports = []
        self.latent_sources = []
        for step in (0, 250, 500):
            path = self.run_dir / "latent_exports" / f"step_{step:09d}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"latent-{step}".encode())
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.latent_exports.append(
                {
                    "path": str(path),
                    "sha256": digest,
                    "rows": 8,
                    "global_step": step,
                    "run_id": self.run_id,
                }
            )
            self.latent_sources.append(
                {
                    "path": str(path),
                    "sha256": digest,
                    "epoch": 0,
                    "global_step": step,
                }
            )
        self.marker["registered_latent_exports"] = self.latent_exports
        write_json(self.run_dir / "TECHNICAL_CANARY_COMPLETE.json", self.marker)
        data_artifact = Path(self.temporary.name) / "visible/ph006/contract.json"
        data_artifact.parent.mkdir(parents=True)
        data_artifact.write_text("{}")
        self.data_contract = {
            "schema_version": "p11-jepa-frozen-data-contract-v1",
            "aggregate_sha256": self.digest,
            "files": {
                "ph006_phase_contract": {
                    "path": str(data_artifact),
                    "bytes": 2,
                    "sha256": hashlib.sha256(b"{}").hexdigest(),
                }
            },
        }
        canonical = json.dumps(
            self.data_contract["files"], sort_keys=True, separators=(",", ":")
        ).encode()
        self.digest = hashlib.sha256(canonical).hexdigest()
        self.data_contract["aggregate_sha256"] = self.digest
        self.marker["data_contract_aggregate_sha256"] = self.digest
        write_json(self.run_dir / "TECHNICAL_CANARY_COMPLETE.json", self.marker)
        write_json(self.run_dir / "FROZEN_DATA_CONTRACT.json", self.data_contract)
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        source_name = str(DEFAULT_CONTRACT.relative_to(REPO_ROOT))
        source_hash = hashlib.sha256(DEFAULT_CONTRACT.read_bytes()).hexdigest()
        contract_hash = source_hash
        self.manifest = {
            "schema_version": "p11-paired-degrade-jepa-run-v1",
            "arm": "jepa",
            "seed": 42,
            "training_phases": ["ph002", "ph003", "ph004", "ph005"],
            "validation_and_selection_phase": "ph006",
            "sealed_blind_phase": "ph001",
            "blind_truth_accessed": False,
            "student_only_deployment": True,
            "git_revision_at_launch": revision,
            "contract_sha256": contract_hash,
            "source_contract": {source_name: source_hash},
            "data_contract": self.data_contract,
            "frozen_execution": {"arm": "jepa", "seed": 42},
        }
        write_json(self.run_dir / "run_manifest.json", self.manifest)
        contract = json.loads(DEFAULT_CONTRACT.read_text())
        thresholds = dict(contract["diagnostics"]["registered_gate"])
        gate_version = thresholds.pop("version")
        self.latent_report = {
            "schema_version": "p11-jepa-latent-diagnostics-v1",
            "status": "pass",
            "pass": True,
            "run_id": self.run_id,
            "selection_phase": "ph006",
            "sealed_phase": "ph001",
            "sealed_phase_opened": False,
            "snapshot_sources": self.latent_sources,
            "thresholds": thresholds,
            "checkpoints": [
                {"global_step": step} for step in (0, 250, 500)
            ],
            "registered_status_gate": {
                "version": gate_version,
                "status": "pass",
                "pass": True,
                "arm": "jepa",
                "required_steps": [0, 250, 500],
                "observed_steps": [0, 250, 500],
                "missing_steps": [],
                "response_only_encoder_available": True,
                "response_only_control_evaluable": True,
            },
        }
        write_json(
            self.run_dir / LATENT_DIAGNOSTIC_FILENAME, self.latent_report
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_marker_and_preallocation_guard_pass_for_exact_frozen_state(self):
        marker = validate_canary_marker(self.run_dir)
        self.assertTrue(marker["pass"])
        report = validate_preallocation(self.run_dir, DEFAULT_CONTRACT)
        self.assertTrue(report["pass"])
        self.assertEqual(report["sealed_phase"], "ph001")
        self.assertFalse(report["sealed_phase_opened"])
        self.assertEqual(report["latent_gate_status"], "pass")

    def test_latent_gate_requires_pass_and_exact_registered_trajectory(self):
        contract = json.loads(DEFAULT_CONTRACT.read_text())
        report = validate_latent_diagnostic_gate(
            self.run_dir, self.marker, contract
        )
        self.assertTrue(report["pass"])

        failed = copy.deepcopy(self.latent_report)
        failed["pass"] = False
        failed["status"] = "fail"
        write_json(self.run_dir / LATENT_DIAGNOSTIC_FILENAME, failed)
        with self.assertRaisesRegex(RuntimeError, "did not pass"):
            validate_preallocation(self.run_dir, DEFAULT_CONTRACT)

        wrong_steps = copy.deepcopy(self.latent_report)
        wrong_steps["registered_status_gate"]["observed_steps"] = [0, 250]
        wrong_steps["registered_status_gate"]["missing_steps"] = [500]
        write_json(self.run_dir / LATENT_DIAGNOSTIC_FILENAME, wrong_steps)
        with self.assertRaisesRegex(RuntimeError, "exact 0/250/500"):
            validate_preallocation(self.run_dir, DEFAULT_CONTRACT)

    def test_preallocation_requires_the_registered_latent_report(self):
        (self.run_dir / LATENT_DIAGNOSTIC_FILENAME).unlink()
        with self.assertRaisesRegex(RuntimeError, "required supervisor artifact"):
            validate_preallocation(self.run_dir, DEFAULT_CONTRACT)

    def test_latent_gate_binds_run_identity_and_snapshot_hashes(self):
        wrong_run = copy.deepcopy(self.latent_report)
        wrong_run["run_id"] = "another/jepa/seed_42"
        write_json(self.run_dir / LATENT_DIAGNOSTIC_FILENAME, wrong_run)
        with self.assertRaisesRegex(RuntimeError, "run identity"):
            validate_preallocation(self.run_dir, DEFAULT_CONTRACT)

        wrong_hash = copy.deepcopy(self.latent_report)
        wrong_hash["snapshot_sources"][1]["sha256"] = "f" * 64
        write_json(self.run_dir / LATENT_DIAGNOSTIC_FILENAME, wrong_hash)
        with self.assertRaisesRegex(RuntimeError, "snapshot hash"):
            validate_preallocation(self.run_dir, DEFAULT_CONTRACT)

    def test_marker_fails_closed_on_one_failed_gate(self):
        marker = copy.deepcopy(self.marker)
        marker["gates"]["finite_gradient_norm"] = False
        write_json(self.run_dir / "TECHNICAL_CANARY_COMPLETE.json", marker)
        with self.assertRaisesRegex(RuntimeError, "did not pass"):
            validate_canary_marker(self.run_dir)

    def test_preallocation_rejects_revision_and_blind_artifact(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["git_revision_at_launch"] = "0" * 40
        write_json(self.run_dir / "run_manifest.json", manifest)
        with self.assertRaisesRegex(RuntimeError, "revision"):
            validate_preallocation(self.run_dir, DEFAULT_CONTRACT)

        write_json(self.run_dir / "run_manifest.json", self.manifest)
        blind = copy.deepcopy(self.data_contract)
        blind["files"]["ph001_phase_contract"] = {
            "path": "/sealed/ph001/phase_contract.json",
            "bytes": 0,
            "sha256": "b" * 64,
        }
        write_json(self.run_dir / "FROZEN_DATA_CONTRACT.json", blind)
        manifest = copy.deepcopy(self.manifest)
        manifest["data_contract"] = blind
        write_json(self.run_dir / "run_manifest.json", manifest)
        with self.assertRaisesRegex(RuntimeError, "sealed ph001"):
            validate_preallocation(self.run_dir, DEFAULT_CONTRACT)

    def test_preallocation_rejects_tampered_stored_data_digest(self):
        tampered = copy.deepcopy(self.data_contract)
        tampered["files"]["ph006_phase_contract"]["bytes"] += 1
        write_json(self.run_dir / "FROZEN_DATA_CONTRACT.json", tampered)
        manifest = copy.deepcopy(self.manifest)
        manifest["data_contract"] = tampered
        write_json(self.run_dir / "run_manifest.json", manifest)
        with self.assertRaisesRegex(RuntimeError, "aggregate digest is invalid"):
            validate_preallocation(self.run_dir, DEFAULT_CONTRACT)

    def test_terminal_check_uses_the_signed_inventory_without_live_rehash(self):
        completion = {
            **self.manifest,
            "status": "TRAINING_COMPLETE",
            "global_steps": 2000,
            "epochs_completed": 10,
            "history": [{"epoch": 10}],
            "ph001_opened": False,
        }
        write_json(self.run_dir / "P11_MATCHED_ARM_COMPLETE.json", completion)
        report = validate_stored_complete(self.run_dir, DEFAULT_CONTRACT)
        self.assertTrue(report["pass"])
        self.assertEqual(report["mode"], "terminal")
        self.assertFalse(report["live_data_rehashed"])


if __name__ == "__main__":
    unittest.main()
