import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from workflows.abacus_tweb.p10_build_blind_observed_geometry import output_dtype
from workflows.abacus_tweb.p10_build_phase_graph import concatenate_npy
from workflows.abacus_tweb.p10_build_phase_index import write_compatibility_manifest
from workflows.abacus_tweb.p10_validate_phase_products import atomic_json, phase_paths
from workflows.abacus_tweb.p10_multiphase_status import PHASES, readiness, record
from workflows.abacus_tweb.p10_import_ph000_reference import atomic_json as reference_atomic_json


class P10PhaseProductTests(unittest.TestCase):
    def test_normalized_status_keeps_scientific_roles_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = record(root, "ph000")
            training = record(root, "ph003")
            selection = record(root, "ph006")
            self.assertEqual(reference["role"], "development_reference")
            self.assertFalse(reference["eligible_for_training"])
            self.assertEqual(reference["status"]["p2_ngc"], "legacy_global_graph")
            self.assertTrue(training["eligible_for_training"])
            self.assertEqual(selection["role"], "validation_and_selection")
            self.assertFalse(selection["eligible_for_training"])

    def test_training_readiness_separates_products_from_loader_canary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = {phase: record(root, phase) for phase in PHASES}
            for phase in ("ph001", "ph002", "ph003", "ph004", "ph005", "ph006"):
                records[phase]["status"]["phase_complete"] = True
            records["ph000"]["status"]["phase_complete"] = True
            first = readiness(root, records)
            self.assertTrue(first["p1_p4_products_ready"])
            self.assertFalse(first["ready_to_launch_deterministic_training"])
            marker = root / "training_contract/TRAINING_LOADER_READY.json"
            marker.parent.mkdir(parents=True)
            marker.write_text("{}\n")
            second = readiness(root, records)
            self.assertTrue(second["ready_to_launch_deterministic_training"])

    def test_phase_paths_include_frozen_truth_products(self):
        registry = {"path_templates": {"phase_output": "/tmp/p10/{phase}"}}
        paths = phase_paths(registry, "ph004")
        self.assertEqual(
            paths["density"].name,
            "AbacusSummit_base_c000_ph004_z0.200_ngrid2048_ab10_tsc_counts.manifest.json",
        )
        self.assertEqual(paths["tweb"].name, "TWEB_COMPLETE.json")
        self.assertEqual(
            paths["catalogue_field_target_closure"].name,
            "catalogue_field_target_closure.json",
        )

    def test_atomic_json_serializes_numpy_gate_scalars(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "marker.json"
            atomic_json(output, {"pass": np.bool_(True), "count": np.int64(7)})
            self.assertEqual(json.loads(output.read_text()), {"pass": True, "count": 7})

    def test_reference_import_json_serializes_numpy_gate_scalars(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "marker.json"
            reference_atomic_json(output, {"pass": np.bool_(True), "count": np.int64(7)})
            self.assertEqual(json.loads(output.read_text()), {"pass": True, "count": 7})

    def test_blind_observed_dtype_has_linkage_but_no_truth(self):
        names = set(output_dtype().names)
        self.assertTrue({"TARGETID", "RA", "DEC", "Z", "FILE_NUM", "HALO_INDEX", "BOX_INDEX"} <= names)
        self.assertFalse({"CWEB", "LAMBDA1", "LAMBDA2", "LAMBDA3"} & names)

    def test_cap_concatenation_preserves_global_indices(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = root / "ngc.npy"
            right = root / "sgc.npy"
            out = root / "all.npy"
            np.save(left, np.asarray([[5, 9], [9, 12]], dtype=np.int32))
            np.save(right, np.asarray([[1, 3]], dtype=np.int32))
            concatenate_npy([left, right], out, (2,), np.dtype("int32"))
            self.assertEqual(np.load(out).tolist(), [[5, 9], [9, 12], [1, 3]])

    def test_compatibility_manifest_preserves_blind_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = root / "canonical_index.npz"
            points = root / "points.npy"
            marker = root / "CATALOGUE_COMPLETE.json"
            manifest = root / "manifest.json"
            np.save(points, np.asarray([[1.0, 2.0, 3.0, 1.0], [2.0, 3.0, 4.0, 0.0]]))
            np.savez(index, parent_node_id=np.arange(2, dtype=np.int64))
            import hashlib
            def digest(path):
                return hashlib.sha256(path.read_bytes()).hexdigest()
            marker.write_text(json.dumps({
                "phase": "ph001", "role": "sealed_blind",
                "catalogue_id": "ph001_bgs_bright_full_ngc_sgc_v1",
                "counts": {"total": 2},
                "canonical_parent": {"path": "/sealed/input.fits", "sha256": "abc"},
                "artifacts": {"canonical_index_sha256": digest(index),
                              "points_sha256": digest(points)},
                "target_truth_present": False,
                "blind_contract": {"sealed": True},
                "target_contract": {"tidal_smoothing_mpc_h": 7.0},
            }))
            payload = write_compatibility_manifest(
                marker=marker, manifest=manifest, index_path=index, points_path=points,
            )
            self.assertFalse(payload["target_truth_present"])
            self.assertTrue(payload["blind_contract"]["sealed"])
            self.assertEqual(payload["catalogue_id"], "ph001_bgs_bright_full_ngc_sgc_v1")


if __name__ == "__main__":
    unittest.main()
