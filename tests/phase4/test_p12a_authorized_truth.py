from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from workflows.sbi import p12a_authorized_truth as truth


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class P12AAuthorizedTruthTest(unittest.TestCase):
    def test_join_preserves_frozen_nonmonotonic_order(self):
        context_parent = np.asarray([3, 0, 2], dtype=np.int64)
        canonical_parent = np.arange(4, dtype=np.int64)
        canonical_targetid = np.asarray([2, 4, 1, 3], dtype=np.int64)
        annotated_targetid = np.arange(1, 5, dtype=np.int64)
        eigenvalues = np.asarray(
            [
                [-0.4, -0.2, 0.0],
                [-0.1, 0.3, 0.5],
                [0.0, 0.2, 0.7],
                [-0.2, 0.1, 0.4],
            ],
            dtype=np.float32,
        )
        cweb = np.sum(eigenvalues > 0.2, axis=1).astype(np.uint8)
        targetid, joined, joined_class = truth.join_by_parent(
            context_parent=context_parent,
            canonical_parent=canonical_parent,
            canonical_targetid=canonical_targetid,
            annotated_targetid=annotated_targetid,
            annotated_eigenvalues=eigenvalues,
            annotated_cweb=cweb,
        )
        np.testing.assert_array_equal(targetid, [3, 2, 1])
        np.testing.assert_array_equal(joined, eigenvalues[[2, 1, 0]])
        np.testing.assert_array_equal(joined_class, cweb[[2, 1, 0]])

    def test_join_fails_closed_on_identity_or_physics_mismatch(self):
        arguments = {
            "context_parent": np.asarray([1, 0]),
            "canonical_parent": np.arange(2),
            "canonical_targetid": np.asarray([1, 2]),
            "annotated_targetid": np.asarray([1, 2]),
            "annotated_eigenvalues": np.asarray(
                [[-0.2, 0.0, 0.3], [-0.1, 0.1, 0.4]], dtype=np.float32
            ),
            "annotated_cweb": np.asarray([1, 1], dtype=np.uint8),
        }
        truth.join_by_parent(**arguments)
        for key, replacement in (
            ("context_parent", np.asarray([0, 0])),
            ("canonical_parent", np.asarray([1, 0])),
            ("annotated_targetid", np.asarray([2, 1])),
            ("annotated_cweb", np.asarray([0, 1], dtype=np.uint8)),
        ):
            changed = dict(arguments)
            changed[key] = replacement
            with self.assertRaises(truth.AuthorizedTruthError):
                truth.join_by_parent(**changed)

    def test_stage_marker_is_exclusive_and_detects_artifact_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "ph001" / "p12a_v1"
            root.mkdir(parents=True)
            authorization = root / "P12_BLIND_OPEN_AUTHORIZED.json"
            frozen = root / "P12_BLIND_PREDICTIONS_FROZEN.json"
            contract = root / "P12A_BLIND_EVALUATION_CONTRACT.json"
            for path in (authorization, frozen, contract):
                path.write_text("{}")
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"immutable")
            fake_authorization = {
                "frozen_predictions_reference": truth.record(frozen),
                "evaluation_contract_reference": truth.record(contract),
            }
            patches = (
                mock.patch.object(truth, "TRUTH_ROOT", root),
                mock.patch.object(truth, "AUTHORIZATION", authorization),
                mock.patch.object(truth, "FROZEN_PREDICTIONS", frozen),
                mock.patch.object(truth, "EVALUATION_CONTRACT", contract),
                mock.patch.object(
                    truth, "authorization_context", return_value=fake_authorization
                ),
                mock.patch.object(truth, "git_revision", return_value="test-revision"),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
                inventory = {
                    "verified": True,
                    "field_asdf_count": 34,
                    "halo_asdf_count": 34,
                    "payload_bytes": 9,
                }
                with mock.patch.object(truth, "verify_b_tree", return_value=inventory):
                    marker = truth.write_stage_marker(
                        stage="particle_b",
                        artifacts={"p10_b_stage_marker": truth.record(artifact)},
                        audit={
                            "particle_root": str(root / "particle_b"),
                            "field_asdf_count": 34,
                            "halo_asdf_count": 34,
                            "payload_bytes": 9,
                        },
                        truth_root=root,
                        authorization_path=authorization,
                        frozen_path=frozen,
                        evaluation_contract_path=contract,
                    )
                    self.assertEqual(marker["open_count"], 1)
                    # Re-entry is allowed only for the same content-addressed payload.
                    same = truth.write_stage_marker(
                        stage="particle_b",
                        artifacts={"p10_b_stage_marker": truth.record(artifact)},
                        audit={"ignored_on_idempotent_reentry": True},
                        truth_root=root,
                        authorization_path=authorization,
                        frozen_path=frozen,
                        evaluation_contract_path=contract,
                    )
                    self.assertEqual(
                        same["artifacts"]["p10_b_stage_marker"]["sha256"],
                        digest(artifact),
                    )
                    artifact.write_bytes(b"changed")
                    with self.assertRaises(truth.AuthorizedTruthError):
                        truth.validate_stage_marker(
                            stage="particle_b",
                            truth_root=root,
                            authorization_path=authorization,
                            frozen_path=frozen,
                            evaluation_contract_path=contract,
                        )

    def test_truth_root_is_canonical_and_disjoint(self):
        self.assertEqual(truth.validate_truth_root(truth.TRUTH_ROOT), truth.TRUTH_ROOT)
        with self.assertRaises(PermissionError):
            truth.validate_truth_root(truth.P10_ROOT / "ph001/truth")

    def test_guard_validates_immediate_upstream_deeply(self):
        with mock.patch.object(
            truth, "authorization_context", return_value={"open_count": 1}
        ) as authorization, mock.patch.object(
            truth, "validate_stage_marker", return_value={"pass": True}
        ) as validate:
            report = truth.guard_stage(stage="tweb", truth_root=truth.TRUTH_ROOT)
        authorization.assert_called_once()
        validate.assert_called_once_with(
            stage="density", truth_root=truth.TRUTH_ROOT, deep_artifacts=True
        )
        self.assertEqual(report["upstream_stage"], "density")

    def test_frozen_physics_constants_match_builder(self):
        from workflows.sbi.p12a_blind_evaluation_contract import (
            TRUTH_CONSTRUCTION_CONTRACT,
        )

        self.assertEqual(TRUTH_CONSTRUCTION_CONTRACT["truth_root"], str(truth.TRUTH_ROOT))
        self.assertEqual(
            TRUTH_CONSTRUCTION_CONTRACT["expected_supported_context_rows"],
            truth.EXPECTED_CONTEXT_ROWS,
        )
        self.assertEqual(
            TRUTH_CONSTRUCTION_CONTRACT["density_grid_size"],
            truth.TARGET["grid_size"],
        )
        self.assertEqual(
            TRUTH_CONSTRUCTION_CONTRACT["tweb_mpi_ranks"],
            truth.TARGET["mpi_ranks"],
        )

    def test_slurm_chain_is_fail_closed_and_resource_exact(self):
        root = Path(__file__).resolve().parents[2] / "workflows/sbi"
        names = {
            "particle": "submit_p12a_ph001_particle_b.slurm",
            "density": "submit_p12a_ph001_density.slurm",
            "tweb": "submit_p12a_ph001_tweb.slurm",
            "annotation": "submit_p12a_ph001_annotation.slurm",
            "compact": "submit_p12a_ph001_compact_truth.slurm",
        }
        payload = {key: (root / name).read_text() for key, name in names.items()}
        for text in payload.values():
            self.assertIn("p12a_authorized_truth guard", text)
            self.assertIn("--licenses=", text)
            self.assertIn("scratch", text)
            self.assertIn("unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD", text)
        self.assertIn("--qos=xfer", payload["particle"])
        self.assertIn("--constraint=cron", payload["particle"])
        self.assertIn("--licenses=hpss,scratch", payload["particle"])
        self.assertNotIn("#SBATCH --nodes", payload["particle"])
        self.assertIn("#SBATCH --nodes=4", payload["tweb"])
        self.assertIn("#SBATCH --ntasks=16", payload["tweb"])
        self.assertIn("#SBATCH --ntasks-per-node=4", payload["tweb"])
        self.assertIn("--nodes=4 --ntasks=16 --ntasks-per-node=4", payload["tweb"])
        chain = (root / "submit_p12a_ph001_truth_chain.sh").read_text()
        self.assertIn("p12a_authorized_truth guard", chain)
        self.assertEqual(chain.count("--dependency=\"afterok:"), 4)
        self.assertNotIn("p10_phase_registry_v1.json", chain)


if __name__ == "__main__":
    unittest.main()
