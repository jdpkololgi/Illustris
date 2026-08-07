from __future__ import annotations

import unittest

import numpy as np

from workflows.abacus_tweb.p3a_build_canonical_fields import GridSpec
from workflows.abacus_tweb.p8_build_multitracer_control_fields import (
    angular_null,
    complete_tsc_support,
    density_matched_indices,
    tsc_deposit,
)


class MultitracerControlFieldTests(unittest.TestCase):
    def test_tsc_conserves_and_has_expected_centred_weights(self) -> None:
        spec = GridSpec((0.0, 0.0, 0.0), (5, 5, 5), 1.0, 0.0)
        point = np.array([[2.5, 2.5, 2.5]])
        field, stats = tsc_deposit(point, spec)
        self.assertAlmostEqual(float(field.sum()), 1.0, places=6)
        self.assertAlmostEqual(float(field[2, 2, 2]), 0.75**3, places=6)
        self.assertAlmostEqual(stats["lost_weight"], 0.0, places=12)

    def test_tsc_support_rejects_grid_edge(self) -> None:
        spec = GridSpec((0.0, 0.0, 0.0), (5, 5, 5), 1.0, 0.0)
        supported = complete_tsc_support(
            np.array([[2.5, 2.5, 2.5], [0.5, 2.5, 2.5]]), spec
        )
        np.testing.assert_array_equal(supported, [True, False])

    def test_density_matching_reproduces_bright_count_in_each_stratum(self) -> None:
        tracer = np.array([0, 0, 1, 1, 0, 1], dtype=np.uint8)
        context = np.ones(6, dtype=bool)
        cap = np.zeros(6, dtype=np.uint8)
        stratum = np.array([1, 1, 1, 1, 2, 2], dtype=np.int8)
        selected, _ = density_matched_indices(
            tracer=tracer, context=context, cap=cap, stratum=stratum, seed=42
        )
        self.assertEqual(np.count_nonzero(stratum[selected] == 1), 2)
        self.assertEqual(np.count_nonzero(stratum[selected] == 2), 1)

    def test_angular_null_preserves_radii_and_direction_multiset(self) -> None:
        xyz = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
        null, donor, _ = angular_null(
            xyz, cap=np.zeros(3, dtype=np.uint8),
            stratum=np.ones(3, dtype=np.int8), seed=7,
        )
        np.testing.assert_allclose(np.linalg.norm(null, axis=1), [1.0, 2.0, 3.0])
        original_direction = xyz / np.linalg.norm(xyz, axis=1)[:, None]
        null_direction = null / np.linalg.norm(null, axis=1)[:, None]
        np.testing.assert_allclose(null_direction, original_direction[donor])


if __name__ == "__main__":
    unittest.main()
