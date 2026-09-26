"""Synthetic checks; no scientific catalogue or native halo access."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from workflows.abacus_tweb import p12a_coordinate_sample_audit as audit


class CoordinateSampleTests(unittest.TestCase):
    def test_stored_member_matches_numpy_with_zip64_and_fortran(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "slab.npz"
            for a in (np.arange(120, dtype=np.float64).reshape(3, 4, 5, 2),
                      np.asfortranarray(np.arange(120).reshape(3, 4, 5, 2))):
                np.savez(path, unrelated=np.arange(7), eig_vals=a)
                mapped = audit.mapped_member(path, "eig_vals")
                np.testing.assert_array_equal(mapped[:, [0, 3], [4, 1], [1, 0]],
                                              a[:, [0, 3], [4, 1], [1, 0]])
                self.assertFalse(mapped.flags.writeable)
                del mapped

    def test_compressed_member_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compressed.npz"
            np.savez_compressed(path, eig_vals=np.arange(10))
            with self.assertRaisesRegex(ValueError, "uncompressed"):
                audit.mapped_member(path, "eig_vals")

    def test_object_member_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "object.npz"
            np.savez(path, eig_vals=np.array([object()], dtype=object))
            with self.assertRaisesRegex(ValueError, "dtype"):
                audit.mapped_member(path, "eig_vals")

    def test_periodic_native_voxels_and_boundary(self):
        cell = 2000. / 2048
        actual = audit.grid_indices([[0, 2000, -cell], [-1000, cell, 2000+cell]])
        np.testing.assert_array_equal(actual, [[0, 0, 2047], [1024, 1, 1]])

    def test_parent_join_detects_identity_and_label_errors(self):
        dtype = [(name, "f8") for name in ("TARGETID", "FILE_NUM", "HALO_INDEX", "BOX_INDEX",
                 "RA", "DEC", "LAMBDA1", "LAMBDA2", "LAMBDA3", "CWEB")]
        observed = np.zeros(2, dtype=dtype)
        observed["TARGETID"] = [1, 2]
        audit.assert_parent_join(observed, observed.copy())
        with self.assertRaisesRegex(ValueError, "TARGETID"):
            audit.assert_parent_join(observed, observed[::-1])
        corrupt = observed.copy()
        corrupt["LAMBDA2"][0] = .1
        with self.assertRaisesRegex(ValueError, "LAMBDA2"):
            audit.assert_parent_join(observed, corrupt)

    def test_phase_guard_excludes_selection_blind_and_e2e(self):
        for phase in ("ph000", "ph001", "ph006", "ph007", "ph014", "../ph002"):
            with self.assertRaises(ValueError):
                audit.phase_guard(phase)
        audit.phase_guard("ph002")

    def test_slurm_variable_does_not_license_login_node_compute(self):
        with patch.dict(audit.os.environ, {"SLURM_JOB_ID": "123"}), \
                patch.object(audit.socket, "gethostname", return_value="login01"):
            with self.assertRaises(RuntimeError):
                audit.require_compute()

    def test_sampling_covers_small_southern_population_without_labels(self):
        data = np.zeros(808, dtype=[("Z", "f8"), ("BOX_INDEX", "i4"),
                                    ("FILE_NUM", "i4"), ("HALO_INDEX", "i4")])
        caps = np.ones(len(data), dtype=int)
        data["Z"][:800] = np.tile([.2, .3, .4, .5], 200)
        data["FILE_NUM"][:800] = 26
        data["Z"][800:] = np.tile([.2, .3, .4, .5], 2)
        data["FILE_NUM"][800:] = [0, 0, 1, 1, 0, 0, 1, 1]
        caps[800:] = 0
        selected, strata, slabs = audit.choose_native_rows(data, caps)
        self.assertTrue(all(n > 0 for n in strata.values()))
        self.assertLessEqual(len(slabs), 6)
        self.assertEqual(len(selected), len(set(selected)))
        self.assertTrue(all(n <= 16 for n in strata.values()))

    def test_missing_stratum_is_not_silently_qualified(self):
        data = np.zeros(2, dtype=[("Z", "f8"), ("BOX_INDEX", "i4"),
                                  ("FILE_NUM", "i4"), ("HALO_INDEX", "i4")])
        data["Z"] = .2
        _, strata, _ = audit.choose_native_rows(data, np.ones(2, dtype=int))
        self.assertFalse(all(n > 0 for n in strata.values()))


if __name__ == "__main__":
    unittest.main()
