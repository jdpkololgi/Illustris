import unittest

import numpy as np

from workflows.abacus_tweb.p11_build_factorial_view_counts import (
    classify,
    grid_equal,
)
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


if __name__ == "__main__":
    unittest.main()
