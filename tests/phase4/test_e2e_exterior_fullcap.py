import unittest
from workflows.sbi.e2e_exterior_fullcap import input_guard, inference_guard


class ExteriorFullCapTests(unittest.TestCase):
    def test_reserve_unconditionally_rejected(self):
        for phase in ('ph001', 'ph000', 'ph999'):
            with self.assertRaises(PermissionError):
                input_guard(phase)

    def test_explicit_legacy_replication_allowed(self):
        for phase in ('ph004', 'ph005', 'ph006'):
            input_guard(phase)

    def test_fresh_prediction_needs_freeze(self):
        for phase in ('ph018', 'ph019'):
            input_guard(phase)  # targetless preparation only
            with self.assertRaises(PermissionError):
                inference_guard(phase, None)

    def test_development_and_loa_prediction_allowed(self):
        for phase in ('Loa', 'ph012', 'ph013'):
            inference_guard(phase, None)


if __name__ == '__main__':
    unittest.main()
