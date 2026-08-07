from __future__ import annotations

import unittest

import healpy as hp
import numpy as np

from workflows.abacus_tweb.p8_build_multitracer_fields import (
    context_redshift,
    estimate_angular_response,
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

    def test_context_contract_excludes_sentinel(self) -> None:
        redshift = np.array([0.099, 0.10, 0.30, 0.586, 0.595, 0.599, 0.60])
        np.testing.assert_array_equal(
            context_redshift(redshift),
            np.array([False, True, True, False, True, True, False]),
        )


if __name__ == "__main__":
    unittest.main()
