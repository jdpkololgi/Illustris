import unittest
from pathlib import Path
import tempfile

import h5py
import numpy as np

from workflows.abacus_tweb.p11_build_factorial_view_counts import (
    apply_random_support,
    classify,
    grid_equal,
)
from workflows.abacus_tweb.p8_deterministic_common import sha256
from workflows.abacus_tweb.p10_multitracer_source_audit import (
    BRIGHT_BITS,
    FAINT_BITS,
)


class FactorialViewCountsTest(unittest.TestCase):
    def test_tracer_classification_is_explicit(self):
        result = classify(np.asarray([BRIGHT_BITS, FAINT_BITS, 0], dtype=np.int64))
        np.testing.assert_array_equal(result, np.asarray([0, 1, 255], dtype=np.uint8))

    def test_ambiguous_tracer_refused(self):
        with self.assertRaises(RuntimeError):
            classify(np.asarray([BRIGHT_BITS | FAINT_BITS], dtype=np.int64))

    def test_grid_identity_is_exact(self):
        grid = {"shape": [4, 5, 6], "origin_mpc": [1.0, 2.0, 3.0], "cell_mpc": 5.0}
        self.assertTrue(grid_equal(grid, dict(grid)))
        shifted = {**grid, "origin_mpc": [1.0, 2.0, 3.000001]}
        self.assertFalse(grid_equal(grid, shifted))

    def test_random_support_masks_both_tracers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            grid = {
                "shape": [2, 2, 2],
                "origin_mpc": [0.0, 0.0, 0.0],
                "cell_mpc": 5.0,
            }
            product_components = {}
            response_components = {}
            support = np.ones((2, 2, 2), dtype=np.uint8)
            support[0, 0, 0] = 0
            for cap in ("NGC", "SGC"):
                count_path = root / f"{cap}_counts.h5"
                response_path = root / f"{cap}_response.h5"
                with h5py.File(count_path, "w") as handle:
                    handle.create_dataset(
                        "bright_counts", data=np.ones((2, 2, 2), dtype=np.float32),
                        chunks=(2, 2, 2),
                    )
                    handle.create_dataset(
                        "faint_counts", data=2 * np.ones((2, 2, 2), dtype=np.float32),
                        chunks=(2, 2, 2),
                    )
                with h5py.File(response_path, "w") as handle:
                    handle.create_dataset(
                        "support_random", data=support, chunks=(2, 2, 2)
                    )
                product_components[cap] = {
                    "file": str(count_path),
                    "file_sha256": sha256(count_path),
                    "grid": grid,
                }
                response_components[cap] = {
                    "file": str(response_path),
                    "file_sha256": sha256(response_path),
                    "grid": grid,
                }
            result = apply_random_support(
                {"components": product_components, "pass": True},
                {"components": response_components},
            )
            self.assertTrue(result["common_random_support_applied"])
            for cap in ("NGC", "SGC"):
                with h5py.File(product_components[cap]["file"], "r") as handle:
                    self.assertEqual(float(handle["bright_counts"][0, 0, 0]), 0.0)
                    self.assertEqual(float(handle["faint_counts"][0, 0, 0]), 0.0)
                    self.assertEqual(float(handle["bright_counts"][1, 1, 1]), 1.0)


if __name__ == "__main__":
    unittest.main()
