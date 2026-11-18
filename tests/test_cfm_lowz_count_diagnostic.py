import unittest
from unittest.mock import patch

import numpy as np

from workflows.sbi import diagnose_cfm_lowz_counts as diagnostic
from workflows.abacus_tweb.p3a_build_canonical_fields import GridSpec, cic_deposit


class CountDiagnosticTests(unittest.TestCase):
    def test_crop_preserves_full_lattice_inner_counts(self):
        native = GridSpec((-10., -10., -10.), (20, 20, 20), 1., 0.)
        xyz = np.random.default_rng(13).uniform(-9., 9., (3000, 3))
        for factor in (1, 2, 4):
            full = GridSpec(native.origin, tuple(n*factor for n in native.shape), 1./factor, 0.)
            values, _ = cic_deposit(xyz, full)
            local, actual, audit = diagnostic.cropped_counts(native, xyz, 2., factor)
            start = np.rint((np.asarray(local.origin)-np.asarray(full.origin))/full.cell_mpc).astype(int)
            expected = values[tuple(slice(i, i+n) for i, n in zip(start, local.shape))]
            np.testing.assert_array_equal(actual, expected)

    def test_cap_box_need_not_cover_whole_sphere(self):
        native = GridSpec((-1., -10., -10.), (11, 20, 20), 1., 0.)
        xyz = np.random.default_rng(14).uniform([-.9, -9., -9.], [9., 9., 9.], (3000, 3))
        for factor in (1, 2, 4):
            full = GridSpec(native.origin, tuple(n*factor for n in native.shape), 1./factor, 0.)
            values, _ = cic_deposit(xyz, full)
            local, actual, _ = diagnostic.cropped_counts(native, xyz, 2., factor)
            start = np.rint((np.asarray(local.origin)-np.asarray(full.origin))/full.cell_mpc).astype(int)
            expected = values[tuple(slice(i, i+n) for i, n in zip(start, local.shape))]
            np.testing.assert_array_equal(actual, expected)

    def test_phase_guard_and_refinement_limits(self):
        with patch.object(diagnostic, 'require_compute'), patch.object(diagnostic.lowz, 'load_plan') as load:
            with self.assertRaises(ValueError):
                diagnostic.run('ph018')
            load.assert_not_called()
        with self.assertRaises(ValueError):
            diagnostic.cropped_counts(GridSpec((-2.,)*3, (4,)*3, 1., 0.), np.zeros((1, 3)), 1., 8)


if __name__ == '__main__':
    unittest.main()
