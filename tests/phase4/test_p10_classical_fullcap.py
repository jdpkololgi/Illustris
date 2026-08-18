import unittest
from pathlib import Path
import tempfile

import h5py
import numpy as np
import torch

from workflows.abacus_tweb.p10_classical_fullcap import (
    _dtfe_delta_to_gpu,
    apply_affine,
)


class P10ClassicalFullcapTests(unittest.TestCase):
    def test_apply_affine_is_columnwise_and_float32(self):
        raw = np.asarray([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
        affine = {
            "coefficients": [
                {"slope": 2.0, "intercept": 0.5},
                {"slope": -1.0, "intercept": 4.0},
                {"slope": 0.5, "intercept": -2.0},
            ]
        }
        expected = np.asarray([[2.5, 2.0, -0.5], [-1.5, 4.0, -1.5]], dtype=np.float32)
        actual = apply_affine(raw, affine)
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, expected)

    def test_dtfe_expected_count_conversion_matches_response_convention(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            density_path = root / "density.npy"
            field_path = root / "field.h5"
            np.save(
                density_path,
                np.asarray([[[1.0, 2.0], [4.0, np.nan]]], dtype=np.float32),
            )
            with h5py.File(field_path, "w") as handle:
                handle.create_dataset(
                    "expected_counts",
                    data=np.asarray([[[8.0, 8.0], [8.0, 8.0]]], dtype=np.float32),
                )
                handle.create_dataset(
                    "exposure_apodized",
                    data=np.asarray([[[1.0, 0.5], [0.0, 1.0]]], dtype=np.float32),
                )
            # cell=2 Mpc makes nbar=expected/cell^3=1 galaxy/Mpc^3.
            delta, report = _dtfe_delta_to_gpu(
                density_path,
                field_path,
                cell_mpc=2.0,
                device="cpu",
                slab=1,
            )
            expected = np.asarray([[[0.0, 0.5], [0.0, 0.0]]], dtype=np.float32)
            np.testing.assert_allclose(delta.cpu().numpy(), expected)
            self.assertEqual(report["used_voxels"], 2)
            self.assertEqual(delta.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
