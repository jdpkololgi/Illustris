import unittest

import numpy as np

from workflows.abacus_tweb.p8_build_field_output_tiling import (
    axis_owner_ranges,
    core_owner_table,
    coverage_from_rows,
    supporting_rows,
)


class FieldOutputTilingTests(unittest.TestCase):
    def test_axis_owner_ranges_are_exact_partition(self):
        self.assertEqual(
            axis_owner_ranges(5, 1.0, 2.0),
            {0: (0, 2), 1: (2, 4), 2: (4, 5)},
        )

    def test_owner_table_partitions_every_voxel_once(self):
        shape = (5, 4, 3)
        keys, starts, stops = core_owner_table(shape, 1.0, 2.0)
        self.assertEqual(len(keys), 12)
        multiplicity = np.zeros(shape, dtype=np.uint8)
        for start, stop in zip(starts, stops, strict=True):
            multiplicity[
                start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]
            ] += 1
        np.testing.assert_array_equal(multiplicity, np.ones(shape, dtype=np.uint8))

    def test_inference_extension_covers_supported_absent_owner(self):
        shape = (5, 4, 3)
        keys, starts, stops = core_owner_table(shape, 1.0, 2.0)
        support = np.zeros(shape, dtype=bool)
        support[0, 0, 0] = True
        support[4, 3, 2] = True
        needed = supporting_rows(support, starts, stops)
        nominal = np.all(keys == np.asarray([0, 0, 0]), axis=1)
        output = nominal | needed
        coverage = coverage_from_rows(shape, starts[output], stops[output])
        self.assertEqual(int(np.count_nonzero(needed)), 2)
        self.assertTrue(np.all(coverage[support]))
        self.assertEqual(int(np.count_nonzero(output & ~nominal)), 1)


if __name__ == "__main__":
    unittest.main()
