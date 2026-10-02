import unittest
from unittest.mock import patch
from pathlib import Path
from workflows.sbi import e2e_exterior_legacy as legacy
from workflows.sbi import e2e_coupled_contract as original


class LegacyAuthorityTests(unittest.TestCase):
    def test_reject_before_any_file_access(self):
        with patch.object(Path, 'read_text', side_effect=AssertionError('Opened a file')):
            for phase in ('ph001', 'ph000', 'ph018', 'ph999'):
                with self.assertRaises(PermissionError):
                    legacy.guard(phase)

    def test_cross_phase_paths_rejected(self):
        with self.assertRaises(PermissionError):
            legacy.checked_path(legacy.ROOT/'ph001'/'native.npy', 'ph004')
        with self.assertRaises(PermissionError):
            legacy.checked_path('/tmp/legacy.npy', 'ph004', output=True)

    def test_new_authority_keeps_old_guards(self):
        for phase in legacy.PHASES:
            self.assertFalse(legacy.guard(phase)['model_training_allowed'])
            with self.assertRaises(PermissionError):
                original.phase_guard(phase)


if __name__ == '__main__':
    unittest.main()
