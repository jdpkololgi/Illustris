import unittest
from workflows.sbi.e2e_cfm_mixture_contract import (
    component_seed, pilot_guard, sampler_seed_guard,
)


class MixtureContractTests(unittest.TestCase):
    def test_addressed_component_choice(self):
        for phase in ('ph012', 'ph013'):
            choices = [component_seed(phase, d) for d in range(32)]
            self.assertEqual(choices, [component_seed(phase, d) for d in range(32)])
            self.assertEqual(set(choices), {17, 29})

    def test_phase_draw_guard(self):
        for phase, draw in [('ph001', 0), ('ph018', 0), ('Loa', 0),
                            ('ph012', -1), ('ph013', 32), ('ph012', True)]:
            with self.assertRaises(PermissionError):
                component_seed(phase, draw)

    def test_pilot_scope(self):
        for phase in ('ph012', 'ph013'):
            for seed in (17, 29):
                pilot_guard(phase, seed)
        for phase, seed in [('ph018', 17), ('Loa', 29), ('ph012', 42)]:
            with self.assertRaises(PermissionError):
                pilot_guard(phase, seed)

    def test_unchanged_seed17_interface(self):
        # Existing phase/freeze checks still govern these calls in the sampler.
        sampler_seed_guard(17, 'ph012', False, None)
        sampler_seed_guard(17, 'Loa', False, 'existing-freeze')

    def test_seed29_requires_bounded_pilot(self):
        sampler_seed_guard(29, 'ph012', True, None)
        sampler_seed_guard(29, 'ph013', True, None)
        for args in [(29, 'ph012', False, None), (29, 'Loa', True, None),
                     (29, 'ph018', True, None), (29, 'ph013', True, 'freeze'),
                     (42, 'ph012', True, None)]:
            with self.assertRaises(PermissionError):
                sampler_seed_guard(*args)

    def test_registered_full_comparison(self):
        record = dict(schema='cfm-model-mixture-generation-v1', phases=['ph012', 'ph013'],
                      seeds=[17, 29], nfe=128, draws=32)
        for phase in record['phases']:
            sampler_seed_guard(29, phase, False, None, record)
        for key, value in [('schema', 'other'), ('phases', ['ph018']), ('draws', 64),
                           ('nfe', 2), ('seeds', [17, 29, 42])]:
            with self.assertRaises(PermissionError):
                sampler_seed_guard(29, 'ph012', False, None, dict(record, **{key: value}))
        for phase, benchmark, freeze in [('ph018', False, None), ('Loa', False, None),
                                         ('ph012', True, None), ('ph013', False, 'freeze')]:
            with self.assertRaises(PermissionError):
                sampler_seed_guard(29, phase, benchmark, freeze, record)


if __name__ == '__main__':
    unittest.main()
