import unittest

import numpy as np

from workflows.abacus_tweb.p10_build_multitracer_views import (
    assigned_faint_ids,
    cap_from_radec,
    join_assigned_faint,
    unique_forfa_faint,
)
from workflows.abacus_tweb.p10_multitracer_source_audit import BRIGHT_BITS, FAINT_BITS


FORFA_DTYPE = np.dtype([
    ("TARGETID", "i8"), ("BGS_TARGET", "i8"), ("RA", "f8"), ("DEC", "f8"),
    ("RSDZ", "f8"), ("TRUEZ", "f8"),
])


class P10MultitracerViewTests(unittest.TestCase):
    def test_assigned_join_is_unique_and_context_limited(self):
        forfa = np.asarray([
            (8, FAINT_BITS, 10.0, 10.0, 0.61, 0.61),
            (2, FAINT_BITS, 20.0, 20.0, 0.30, 0.30),
            (3, BRIGHT_BITS, 30.0, 30.0, 0.30, 0.30),
        ], dtype=FORFA_DTYPE)
        assigned = np.asarray(
            [(2, FAINT_BITS), (2, 0), (8, FAINT_BITS)],
            dtype=[("TARGETID", "i8"), ("BGS_TARGET", "i8")],
        )
        truth = unique_forfa_faint(forfa)
        result = join_assigned_faint(truth, assigned_faint_ids(assigned))
        np.testing.assert_array_equal(result["TARGETID"], [2])

    def test_duplicate_forfa_truth_is_rejected(self):
        rows = np.asarray([
            (2, FAINT_BITS, 20.0, 20.0, 0.30, 0.30),
            (2, FAINT_BITS, 20.0, 20.0, 0.30, 0.30),
        ], dtype=FORFA_DTYPE)
        with self.assertRaises(RuntimeError):
            unique_forfa_faint(rows)

    def test_cap_transform_is_deterministic(self):
        cap = cap_from_radec(np.asarray([0.0, 180.0]), np.asarray([90.0, -90.0]))
        np.testing.assert_array_equal(cap, [1, 0])


if __name__ == "__main__":
    unittest.main()
