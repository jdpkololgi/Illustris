import unittest
from unittest.mock import patch

from workflows.sbi import prepare_cfm_lowz_target_canary as canary


class TargetCanaryTests(unittest.TestCase):
    def test_selection_is_deterministic_geometry_only(self):
        rows = [dict(phase='ph001', cap=cap, redshift=z, center=[i, 0, 0])
                for cap in ('NGC', 'SGC') for i, z in enumerate([.08, .04, .06])]
        chosen = canary.select_canaries(rows, 'ph001')
        self.assertEqual([r['redshift'] for r in chosen], [.04, .04])
        self.assertEqual([r['cap'] for r in chosen], ['SGC', 'NGC'])
        self.assertNotIn('pair_id', rows[0])
        with self.assertRaises(ValueError):
            canary.select_canaries(rows, 'ph006')

    def test_sealed_guard_precedes_sources(self):
        with patch.object(canary, 'require_compute'), patch.object(canary.lowz, 'load_plan') as load:
            with self.assertRaises(ValueError):
                canary.run('ph018')
            load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
