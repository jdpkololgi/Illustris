import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from workflows.abacus_tweb.p10_build_blind_observed_geometry import output_dtype
from workflows.abacus_tweb.p10_build_phase_graph import concatenate_npy
from workflows.abacus_tweb.p10_build_phase_index import write_compatibility_manifest


class P10PhaseProductTests(unittest.TestCase):
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
