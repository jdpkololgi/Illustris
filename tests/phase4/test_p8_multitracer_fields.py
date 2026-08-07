from __future__ import annotations

import unittest

import healpy as hp
import numpy as np

from workflows.abacus_tweb.p3a_build_canonical_fields import GridSpec
from workflows.abacus_tweb.p8_build_multitracer_fields import (
    context_redshift,
    estimate_angular_response,
    complete_cic_support,
)


class MultitracerFieldTests(unittest.TestCase):
    def test_response_is_regularized_and_bounded(self) -> None:
        target = np.array([3, 3, 3, 7, 7])
        selected = np.array([3, 7])
        response, target_count, selected_count, summary = estimate_angular_response(
            target, selected, nside=1, prior_targets=2.0
        )
        self.assertEqual(target_count.sum(), 5)
        self.assertEqual(selected_count.sum(), 2)
        self.assertEqual(summary["global_selected_over_target"], 0.4)
        self.assertTrue(np.all((response >= 0) & (response <= 1)))
        self.assertEqual(response.shape, (hp.nside2npix(1),))

    def test_selected_cannot_exceed_target_support(self) -> None:
        with self.assertRaises(RuntimeError):
            estimate_angular_response(
                np.array([1]), np.array([1, 1]), nside=1, prior_targets=2.0
            )

    def test_complete_cic_support_identifies_grid_edge_points(self) -> None:
        spec = GridSpec(
            origin=(0.0, 0.0, 0.0),
            shape=(4, 4, 4),
            cell_mpc=1.0,
            padding_mpc=0.0,
        )
        points = np.array(
            [
                [1.0, 1.0, 1.0],
                [3.4, 3.4, 3.4],
                [3.5, 3.5, 3.5],
                [-0.1, 1.0, 1.0],
            ]
        )
        np.testing.assert_array_equal(
            complete_cic_support(points, spec), [True, True, False, False]
        )

    def test_context_contract_excludes_sentinel(self) -> None:
        redshift = np.array([0.099, 0.10, 0.30, 0.586, 0.595, 0.599, 0.60])
        np.testing.assert_array_equal(
            context_redshift(redshift),
            np.array([False, True, True, False, True, True, False]),
        )


if __name__ == "__main__":
    unittest.main()
