"""Small analytic checks only; no production catalogues or grid payloads."""
import unittest
from unittest.mock import patch

import numpy as np

from workflows.sbi import diagnose_cfm_lowz_volume as diagnostic


class VolumeDiagnosticTests(unittest.TestCase):
    def test_full_sphere_reference_and_response_weight(self):
        angular = dict(support=np.ones(12, bool), domain=np.zeros(12, int),
                       angular_response=np.full(12, .5))
        actual = diagnostic.analytic_volume(angular, 'SGC', [1., 2., 3.])
        np.testing.assert_allclose(actual, 2*np.pi*np.diff(np.array([1., 2., 3.])**3)/3)
        np.testing.assert_array_equal(diagnostic.analytic_volume(angular, 'NGC', [1., 2.]), [0.])

    def test_cap_support_and_invalid_radii(self):
        angular = dict(support=np.array([True]*6+[False]*6), domain=np.zeros(12, int),
                       angular_response=np.ones(12))
        np.testing.assert_allclose(diagnostic.analytic_volume(angular, 'SGC', [1., 2.]), [2*np.pi*7/3])
        for radii in ([0., 1.], [2., 1.], [1., np.nan]):
            with self.assertRaises(ValueError):
                diagnostic.analytic_volume(angular, 'SGC', radii)

    def test_ratios_and_grouping(self):
        np.testing.assert_allclose(diagnostic.relative_change([99., 102.], [100., 100.]), [-.01, .02])
        with self.assertRaises(ValueError):
            diagnostic.relative_change([1.], [0.])
        edges = np.linspace(.01, .05, 9)
        np.testing.assert_array_equal(diagnostic.group_volume(np.ones(8), edges), [4., 4.])
        with self.assertRaises(ValueError):
            diagnostic.group_volume(np.ones(8), edges, [.01, .031, .05])

    def test_phase_guard_precedes_source_access(self):
        with patch.object(diagnostic, 'require_compute'), patch.object(diagnostic.lowz, 'load_plan') as load:
            for phase in ('ph018', 'ph019', 'ph000'):
                with self.assertRaises(ValueError):
                    diagnostic.run(phase)
            load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
