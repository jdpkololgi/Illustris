from __future__ import annotations

import unittest

import numpy as np
import torch

from workflows.abacus_tweb.p8_multitracer_classical import (
    bias_aware_contrast,
    combined_count_contrast,
    fit_relative_bias,
    fold_block,
)


class MultitracerClassicalTests(unittest.TestCase):
    def test_combined_count_contrast_uses_sum_of_responses(self) -> None:
        counts_b = torch.tensor([2.0, 4.0])
        expected_b = torch.tensor([1.0, 2.0])
        counts_f = torch.tensor([3.0, 2.0])
        expected_f = torch.tensor([2.0, 1.0])
        delta, valid, _ = combined_count_contrast(
            counts_b, expected_b, counts_f, expected_f
        )
        self.assertTrue(bool(valid.all()))
        torch.testing.assert_close(delta, torch.tensor([2.0 / 3.0, 1.0]))

    def test_relative_bias_recovers_known_scale(self) -> None:
        bright = torch.linspace(-1.0, 1.0, 200)
        faint = 1.7 * bright
        fit = fit_relative_bias(bright, faint, torch.ones(200, dtype=torch.bool))
        self.assertAlmostEqual(fit["relative_bias_faint_over_bright"], 1.7, places=6)
        self.assertAlmostEqual(fit["correlation"], 1.0, places=6)

    def test_bias_aware_combination_normalizes_faint_response(self) -> None:
        delta_b = torch.tensor([0.5, 0.5, 0.0])
        delta_f = torch.tensor([1.0, 1.0, 1.0])
        expected_b = torch.ones(3)
        expected_f = torch.ones(3)
        valid_b = torch.tensor([True, True, False])
        valid_f = torch.tensor([True, False, True])
        result, valid = bias_aware_contrast(
            delta_b, expected_b, valid_b, delta_f, expected_f, valid_f, 2.0
        )
        self.assertTrue(bool(valid.all()))
        torch.testing.assert_close(result, torch.full((3,), 0.5))

    def test_fold_block_respects_half_open_core_bounds(self) -> None:
        lookup = np.arange(8, dtype=np.int8).reshape(2, 2, 2)
        result = fold_block(
            left=0, right=2, shape=(2, 2, 2), origin=np.zeros(3),
            cell_mpc=1.0, base_mpc=np.zeros(3), core_mpc=1.0, lookup=lookup,
        )
        np.testing.assert_array_equal(result, lookup)


if __name__ == "__main__":
    unittest.main()
