import unittest
import numpy as np
from scipy import fft
from workflows.sbi.e2e_field_build_products import tensor_from_delta
from workflows.sbi.e2e_field_regenerate_spectral import interpolate
from workflows.sbi.e2e_field_wide_coarse import (
    attribution, block_sum, compact_lowpass, load_wide_parent, local_coords, padded_extract,
)


class WideCoarseTests(unittest.TestCase):
    def test_compact_fourier_normalization_and_rejection(self):
        n, m, box = 32, 16, 100.
        x = np.arange(n)
        field = np.broadcast_to((.7*np.cos(2*np.pi*2*x/n)+.3*np.sin(2*np.pi*7*x/n))[:,None,None], (n,)*3)
        got = fft.irfftn(compact_lowpass(fft.rfftn(field), box, m, .2), s=(m,)*3)
        want = np.broadcast_to((.7*np.cos(2*np.pi*2*np.arange(m)/m))[:,None,None], (m,)*3)
        np.testing.assert_allclose(got, want, atol=2e-15)

    def test_compact_mixed_mode_and_tensor_sign(self):
        n, m, box = 32, 16, 100.
        xyz = np.meshgrid(*([np.arange(n)]*3), indexing="ij")
        field = np.cos(2*np.pi*(xyz[0]-xyz[1]+xyz[2])/n)
        small = fft.irfftn(compact_lowpass(fft.rfftn(field), box, m, .2), s=(m,)*3)
        np.testing.assert_allclose(small, field[::2,::2,::2], atol=3e-15)
        np.testing.assert_allclose(tensor_from_delta(small, box/m)[...,1], -small/3, atol=2e-15)

    def test_cutoff_guard(self):
        with self.assertRaises(ValueError):
            compact_lowpass(np.zeros((8,8,5), complex), 100., 4, 1.)

    def test_conservation_partial_cells(self):
        a = np.arange(5*7*3).reshape(5,7,3)
        for factor in (2,4):
            self.assertEqual(block_sum(a, factor).sum(), a.sum())
            self.assertEqual(block_sum(np.ones_like(a), factor).sum(), a.size)

    def test_padding_is_not_periodic(self):
        a = np.arange(27).reshape((3,)*3)+1
        got = padded_extract(a, [-1,1,1], 3)
        np.testing.assert_array_equal(got[0], 0)
        np.testing.assert_array_equal(got[1:,:2,:2], a[:2,1:,1:])
        np.testing.assert_array_equal(got[:,2], 0)
        np.testing.assert_array_equal(padded_extract(a, [9,9,9], 2), 0)

    def test_coordinate_nesting(self):
        v = {"span_fine_cells": 192, "factor": 4}
        np.testing.assert_allclose((local_coords(v)[0]+.5)*4-96, np.arange(96)-48+.5)

    def test_trace_and_attribution(self):
        v = {"span_fine_cells": 192, "factor": 4}
        coarse = np.broadcast_to(np.sin(np.arange(48)[:,None,None]/8), (48,)*3).copy()
        up = interpolate(coarse, local_coords(v,8), degree=5)
        delta = up+.2
        low = tensor_from_delta(coarse, 4.)
        low = np.stack([interpolate(low[...,j], local_coords(v,8), degree=5) for j in range(6)], -1)
        combined = low+tensor_from_delta(delta-up, 1.)
        np.testing.assert_allclose(combined[...,[0,3,5]].sum(-1), delta, atol=1e-14)
        result = attribution([combined*.3, combined*.2, combined*.5], combined, np.ones((8,)*3, bool))
        self.assertLess(result["closure_max_abs"], 1e-14)
        self.assertGreater(result["frobenius_gram_including_cross_terms"][0][1], 0)

    def test_reader_rejects_holdout_before_io(self):
        with self.assertRaises(PermissionError):
            load_wide_parent("/not/opened", {"role": "internal_selection", "phase": "ph004"}, {})


if __name__ == "__main__":
    unittest.main()
