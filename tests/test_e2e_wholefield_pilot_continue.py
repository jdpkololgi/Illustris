import unittest
from unittest.mock import patch

from workflows.sbi import e2e_wholefield_pilot_continue as m


class ContinueTests(unittest.TestCase):
    def test_failed_or_unverified_segment_never_retries(self):
        self.assertTrue(m.successor_allowed(True, 'COMPLETED', '0:0', 100, 386))
        self.assertFalse(m.successor_allowed(True, 'COMPLETED', '0:0', 386, 386))
        for status in ('TIMEOUT', 'FAILED', 'CANCELLED', 'RUNNING'):
            self.assertFalse(m.successor_allowed(True, status, '0:0', 100, 386))
        self.assertFalse(m.successor_allowed(None, 'COMPLETED', '0:0', 100, 386))
        self.assertFalse(m.successor_allowed(False, 'COMPLETED', '0:0', 100, 386))
        self.assertFalse(m.successor_allowed(True, 'COMPLETED', '1:0', 100, 386))

    def test_owner_requires_end_and_both_success_codes(self):
        prefix = 'salloc: Pending job allocation 123\nsalloc: Granted job allocation 123\n'
        self.assertEqual(m.terminal_owner(prefix), ('123', None))
        self.assertEqual(m.terminal_owner(prefix+'allocation_exit_code=0 source_check_exit_code=0\nowner_end=now'), ('123', True))
        self.assertEqual(m.terminal_owner(prefix+'allocation_exit_code=0 source_check_exit_code=1\nowner_end=now'), ('123', False))

    def test_panel_is_fixed_and_excludes_sealed_phases(self):
        rows = [dict(cap=cap, pair_id=f'{cap}_{i}') for cap in ('NGC', 'SGC') for i in range(8)]
        with patch.object(m, 'read', return_value={'pairs': rows}):
            paths = m.expected('unet')
        self.assertEqual(len(paths), 386)
        self.assertEqual(len(set(paths)), 386)
        self.assertEqual(sum('nfe256' in str(p) for p in paths), 2)
        self.assertTrue(all('ph018' not in str(p) and 'ph019' not in str(p) for p in paths))


if __name__ == '__main__':
    unittest.main()
