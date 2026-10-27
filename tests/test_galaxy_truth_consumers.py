"""Regression checks for the label-validity and reuse boundaries."""
import unittest
from unittest.mock import patch
import tempfile
import json
from pathlib import Path

import numpy as np
import fitsio

from workflows.abacus_tweb.galaxy_truth_contract import label_validity, verify_observed_truth
from workflows.abacus_tweb.audit_galaxy_truth_replacement import host_fold_audit, compare_p1
from workflows.abacus_tweb.p10_build_phase_index import shell_and_masks
from workflows.abacus_tweb.galaxy_position_mapping import MAPPING_VERSION
from workflows.abacus_tweb.galaxy_position_validation_status import LABEL_POLICY_VERSION
from workflows.abacus_tweb.product_lifecycle import digest


class ConsumerTests(unittest.TestCase):
    def test_closure_compares_source_identity_not_distinct_host_members(self):
        from workflows.abacus_tweb.p3a_catalogue_field_closure import host_consistency
        rows = np.zeros(2, dtype=[(k, "i8") for k in
            ("FILE_NUM", "BOX_INDEX", "HALO_INDEX", "SOURCE_ROW", "CWEB")]
            + [(f"LAMBDA{i}", "f4") for i in (1, 2, 3)])
        rows["SOURCE_ROW"] = [10, 11]
        rows["LAMBDA1"] = [-1, -.5]
        rows["LAMBDA2"] = 0
        rows["LAMBDA3"] = 1
        rows["CWEB"] = 1
        result = host_consistency(rows, np.ones(2, bool))
        self.assertEqual(result["identity_key"], "SOURCE_ROW")
        self.assertEqual(result["repeated_host_adjacent_pairs"], 0)
        rows["SOURCE_ROW"][:] = 10
        result = host_consistency(rows, np.ones(2, bool))
        self.assertEqual(result["repeated_host_adjacent_pairs"], 1)
        self.assertGreater(result["max_abs_dlambda_within_repeated_host"], 0)

    def test_negative_box_and_unchecked_position_are_valid_labels(self):
        rows = np.zeros(3, dtype=[(k, "i4") for k in ("BOX_INDEX", "HAS_LABEL", "TWEB_STATUS", "INDEP_POS")]
                        + [(k, "f4") for k in ("LAMBDA1", "LAMBDA2", "LAMBDA3")])
        rows["BOX_INDEX"] = -1
        rows["HAS_LABEL"] = rows["TWEB_STATUS"] = 1
        rows["LAMBDA1"], rows["LAMBDA2"], rows["LAMBDA3"] = -1, 0, 1
        valid = label_validity(rows)
        self.assertTrue(valid.all())
        _, active, context = shell_and_masks(np.array([.08, .12, .20]), valid)
        self.assertEqual(active.tolist(), [False, False, True])
        self.assertEqual(context.tolist(), [False, True, True])
        rows["LAMBDA2"][1] = np.nan
        rows["TWEB_STATUS"][2] = 0
        self.assertEqual(label_validity(rows).tolist(), [True, False, False])

    def test_distinct_cubic_galaxies_in_same_host_must_not_cross_folds(self):
        table = np.array([(0, 20, 10), (0, 21, 10)], dtype=[
            ("FILE_NUM", "i4"), ("BOX_INDEX", "i4"), ("HALO_INDEX", "i4")])
        assignment = dict(parent_node_id=np.array([0, 1]), fold=np.array([0, 1]),
                          supervised_eligible=np.ones(2, dtype=bool))
        self.assertFalse(host_fold_audit(table, assignment)["pass"])
        assignment["fold"][:] = 0
        self.assertTrue(host_fold_audit(table, assignment)["pass"])
        table["BOX_INDEX"][1] = -1
        self.assertFalse(host_fold_audit(table, assignment)["pass"])

    def test_equal_points_do_not_certify_changed_active_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/"a", Path(tmp)/"b"
            a.mkdir(); b.mkdir()
            index = dict(parent_node_id=np.arange(2), targetid=np.arange(2),
                         cap=np.array([0, 1]), shell=np.array([0, 0]),
                         active=np.array([True, False]), context=np.ones(2, bool),
                         valid_target=np.array([True, False]))
            np.savez(a/"canonical_index.npz", **index)
            index["active"][:] = True
            np.savez(b/"canonical_index.npz", **index)
            for root in (a, b):
                np.save(root/"points.npy", np.zeros((2, 4)))
            result = compare_p1(a, b)
            self.assertTrue(result["points"])
            self.assertFalse(result["active"])

    def test_receipt_requires_correct_phase_policy_and_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); observed = root/"observed.fits"; parent = root/"parent.fits"
            fitsio.write(observed, np.zeros(1, dtype=[("SOURCE_ROW", "i8")]),
                         header=dict(TWMAP=MAPPING_VERSION, TWPOLICY=LABEL_POLICY_VERSION, SIMPHASE="ph002"))
            annotation = Path(str(parent)+".manifest.json")
            annotation.write_text(json.dumps(dict(completed=True, phase="ph002",
                label_policy_version=LABEL_POLICY_VERSION, input_rows_fully_labelled=True,
                source_rows_verified=True)))
            rec = dict(schema_version="p10-observed-truth-complete-v3", phase="ph002",
                       join=dict(label_complete=True),
                       output=dict(path=str(observed), rows=1, sha256=digest(observed)),
                       annotated_parent=dict(path=str(parent)),
                       annotation_receipt_sha256=digest(annotation))
            receipt = Path(str(observed)+".complete.json")
            receipt.write_text(json.dumps(rec))
            self.assertEqual(verify_observed_truth(observed, "ph002")["label_policy_version"], LABEL_POLICY_VERSION)
            with self.assertRaises(ValueError):
                verify_observed_truth(observed, "ph003")
            with observed.open("ab") as f:
                f.write(b"changed")
            with self.assertRaises(ValueError):
                verify_observed_truth(observed, "ph002")


if __name__ == "__main__":
    unittest.main()
