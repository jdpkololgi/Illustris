import unittest

import numpy as np

from workflows.abacus_tweb.p10_multitracer_source_audit import (
    BRIGHT_BITS,
    FAINT_BITS,
    target_counts,
)


class P10MultitracerSourceAuditTests(unittest.TestCase):
    def test_target_counts_or_bits_across_duplicate_assignments(self):
        targetid = np.asarray([4, 2, 4, 7, 2], dtype=np.int64)
        bits = np.asarray([0, BRIGHT_BITS, FAINT_BITS, FAINT_BITS, 0], dtype=np.int64)
        row = target_counts(targetid, bits)
        np.testing.assert_array_equal(row["targetid"], [2, 4, 7])
        np.testing.assert_array_equal(row["bright"], [True, False, False])
        np.testing.assert_array_equal(row["faint"], [False, True, True])
        self.assertEqual(row["duplicate_rows"], 2)


if __name__ == "__main__":
    unittest.main()
