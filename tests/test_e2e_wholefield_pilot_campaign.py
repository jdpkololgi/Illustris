import json
import unittest
from unittest.mock import patch

from workflows.sbi.e2e_wholefield_pilot_campaign import tasks


class CampaignTests(unittest.TestCase):
    def test_fixed_panel_and_early_refinement(self):
        rows = [dict(cap=cap, pair_id=f'{cap}_{i}') for cap in ('NGC', 'SGC') for i in range(8)]
        with patch('pathlib.Path.read_text', return_value=json.dumps(dict(pairs=rows))):
            selected = tasks()
        self.assertEqual(len(selected), 386)
        self.assertEqual({r['phase'] for r in selected}, {f'ph{i:03d}' for i in range(12, 18)})
        self.assertEqual({r['seed'] for r in selected}, {17, 29})
        self.assertEqual({r['coarse'] for r in selected}, {'oracle', 'sampled'})
        refined = [r for r in selected if r['nfe'] == 256]
        self.assertEqual(len(refined), 2)
        self.assertTrue(all(r['phase'] == 'ph012' and r['seed'] == 17 for r in refined))
        self.assertEqual(selected[0]['cap'], 'NGC')
        self.assertEqual(next(r for r in selected if r['pair'] != selected[0]['pair'])['cap'], 'SGC')

    def test_missing_cap_fails(self):
        with patch('pathlib.Path.read_text', return_value=json.dumps(dict(pairs=[]))):
            with self.assertRaises(ValueError):
                tasks()


if __name__ == '__main__':
    unittest.main()
