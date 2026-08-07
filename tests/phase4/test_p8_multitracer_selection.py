from __future__ import annotations

import unittest

import numpy as np

from workflows.abacus_tweb.p8_refit_multitracer_selection import point_folds


class MultitracerSelectionTests(unittest.TestCase):
    def test_point_folds_respect_lookup_and_bounds(self) -> None:
        lookup = np.arange(8, dtype=np.int8).reshape(2, 2, 2) % 5
        xyz = np.array(
            [[0.1, 0.1, 0.1], [1.1, 0.1, 1.1], [-0.1, 0.1, 0.1], [2.1, 0.1, 0.1]]
        )
        folds = point_folds(
            xyz, base_mpc=np.zeros(3), core_mpc=1.0, fold_lookup=lookup
        )
        self.assertEqual(folds[0], lookup[0, 0, 0])
        self.assertEqual(folds[1], lookup[1, 0, 1])
        self.assertEqual(folds[2], -1)
        self.assertEqual(folds[3], -1)


if __name__ == "__main__":
    unittest.main()
