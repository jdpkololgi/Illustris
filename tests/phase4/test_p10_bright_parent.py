import unittest

import numpy as np

from workflows.abacus_tweb.p10_build_bright_parent import (
    compact_parent_block,
    make_keys,
    match_bright_chunk,
)


class P10BrightParentTests(unittest.TestCase):
    def test_exact_match_preserves_cutsky_order(self):
        table = np.array(
            [
                (10.0, 1.0, 0.1, 19.0, 1, 11, 21),
                (20.0, 2.0, 0.2, 19.6, 2, 12, 22),
                (30.0, 3.0, 0.3, 18.0, 3, 13, 23),
                (40.0, 4.0, 0.4, 17.0, 4, 14, 24),
            ],
            dtype=[
                ("RA", "f8"), ("DEC", "f8"), ("Z", "f4"), ("R_MAG_APP", "f4"),
                ("FILE_NUM", "i4"), ("HALO_INDEX", "i4"), ("BOX_INDEX", "i4"),
            ],
        )
        keys = make_keys(
            np.array([10.0, 40.0]), np.array([1.0, 4.0]), np.array([0.1, 0.4], dtype=np.float32)
        )
        order = np.argsort(keys, order=("RA", "DEC", "Z"))
        selected, targetids = match_bright_chunk(
            table,
            r_limit=19.5,
            sorted_keys=keys[order],
            sorted_targetids=np.array([1, 2], dtype=np.int64)[order],
        )
        self.assertEqual(selected["RA"].tolist(), [10.0, 40.0])
        self.assertEqual(targetids.tolist(), [1, 2])

    def test_compact_block_assigns_sequential_targetids_and_linkage(self):
        selected = np.array(
            [(1.0, 2.0, 0.3, 0.29, 18.5, 7, 8, 9)],
            dtype=[
                ("RA", "f8"), ("DEC", "f8"), ("Z", "f4"),
                ("Z_COSMO", "f4"), ("R_MAG_APP", "f4"),
                ("FILE_NUM", "i4"), ("HALO_INDEX", "i4"), ("BOX_INDEX", "i4"),
            ],
        )
        block = compact_parent_block(selected, np.array([42]))
        self.assertEqual(int(block["TARGETID"][0]), 42)
        self.assertEqual(int(block["BGS_TARGET"][0]), 2)
        self.assertEqual(
            (int(block["FILE_NUM"][0]), int(block["HALO_INDEX"][0]), int(block["BOX_INDEX"][0])),
            (7, 8, 9),
        )


if __name__ == "__main__":
    unittest.main()
