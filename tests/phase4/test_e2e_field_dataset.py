import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from workflows.sbi.e2e_field_dataset import Moments, ParentDataset, centred_crop, child_layout, transformed


class DatasetTests(unittest.TestCase):
    def test_transforms_reject_negative_counts(self):
        with self.assertRaises(ValueError):
            transformed("counts",np.array([-1.]))
        self.assertTrue(np.isfinite(transformed("ntilde_mpc3",np.zeros(3))).all())

    def test_training_moments(self):
        m = Moments()
        m.add(np.array([1,2,3]))
        m.add(np.array([4,5,6]))
        result = m.report()
        self.assertEqual(result["mean"],3.5)
        self.assertAlmostEqual(result["std"],np.std([1,2,3,4,5,6]))

    def test_nested_voxel_identity_and_ownership(self):
        x = np.arange(96**3,dtype=np.float32).reshape(96,96,96)
        a = centred_crop(x,64)
        np.testing.assert_array_equal(a,x[16:80,16:80,16:80])
        children = child_layout([64,72,80],96)
        owner = np.zeros((96,)*3,dtype=np.uint8)
        start = np.array([64,72,80])-48
        for child in children:
            lo, hi = np.array(child["core_start"])-start,np.array(child["core_stop"])-start
            owner[tuple(slice(a,b) for a,b in zip(lo,hi))] += 1
        self.assertTrue(np.all(owner==1))

    def test_unreleased_and_confirmation_guards(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root)/"index.json"
            p.write_text(json.dumps({"training_ready":False}))
            with self.assertRaises(PermissionError):
                ParentDataset(p,"train")
            with self.assertRaises(PermissionError):
                ParentDataset(p,"internal_confirmation",allow_unreleased=True)


if __name__ == "__main__":
    unittest.main()
