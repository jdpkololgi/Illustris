from datetime import datetime, timezone
import json
from pathlib import Path
import unittest

from workflows.sbi.e2e_wholefield_fullcap_contract import guard, address, admit
from workflows.sbi.e2e_cfm_mixture_contract import sampler_seed_guard


class FullCapContractTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((Path(__file__).parents[1]/'configs/e2e_wholefield_fullcap_v1.json').read_text())

    def test_full_panel(self):
        for p in self.cfg['phases']:
            for seed in (17, 29):
                sampler_seed_guard(seed, p, False, None, self.cfg)
                for cap in ('NGC', 'SGC'):
                    for first in range(0, 32, 4):
                        self.assertIn(p, address(p, cap, seed, first))

    def test_sealed_and_foreign(self):
        for phase in ('ph001', 'ph004', 'ph018', 'ph019', 'Loa'):
            with self.assertRaises(PermissionError):
                guard(self.cfg, phase, 17)
            with self.assertRaises(PermissionError):
                address(phase, 'NGC', 29, 0)

    def test_parameters(self):
        for key, value in [('draws', 64), ('nfe', 64), ('phases', ['ph012']),
                           ('gpu_hour_limit', 251), ('padding', 16), ('fine_step', 13312)]:
            with self.assertRaises(PermissionError):
                guard(dict(self.cfg, **{key: value}), 'ph012', 17)
        for seed, benchmark, freeze in [(42, False, None), (17, True, None), (29, False, 'freeze')]:
            with self.assertRaises(PermissionError):
                guard(self.cfg, 'ph014', seed, benchmark, freeze)

    def test_old_scope_not_relaxed(self):
        for p in ('ph014', 'ph015', 'ph016', 'ph017', 'ph018'):
            with self.assertRaises(PermissionError):
                sampler_seed_guard(29, p, False, None)

    def test_bad_addresses(self):
        for args in [('ph012', 'NGC', 17, True), ('ph012', 'NGC', 17, 32),
                     ('ph012', 'ALL', 17, 0), ('ph012', 'SGC', 42, 0)]:
            with self.assertRaises(PermissionError):
                address(*args)

    def test_resource_admission(self):
        now = datetime(2026, 10, 10, tzinfo=timezone.utc)
        admit(self.cfg, 234, 4, 14400, now)
        for used, gpus, seconds in [(235, 4, 14400), (0, 4, 14401), (0, 8, 3600), (-1, 1, 60)]:
            with self.assertRaises(PermissionError):
                admit(self.cfg, used, gpus, seconds, now)
        with self.assertRaises(PermissionError):
            admit(self.cfg, 0, 1, 3600, datetime(2026, 10, 23, 19, tzinfo=timezone.utc))


if __name__ == '__main__':
    unittest.main()
