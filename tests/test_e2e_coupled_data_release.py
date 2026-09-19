from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from workflows.sbi import e2e_coupled_data_release as release


class CoupledDataReleaseTests(unittest.TestCase):
    def panel(self):
        c = release.c
        phases = {p: dict(role=role, pairs=128 if p in c.TRAIN else 16, **{'pass': True})
                  for p, role in c.ROLES.items()}
        normalizer = dict(fit_phases=list(c.TRAIN), phase_weight=1/13, **{'pass': True})
        interface = dict(phase_roles=c.ROLES, pairs=1792, offset_cases=11776,
                         science_scores_evaluated=False, **{'pass': True})
        return phases, normalizer, interface

    def test_exact_panel_is_required(self):
        release.require_panel(*self.panel())
        for missing in release.c.ROLES:
            args = self.panel(); args[0].pop(missing)
            with self.assertRaises(ValueError): release.require_panel(*args)
        args = self.panel(); args[0]['ph001'] = dict(role='train', pairs=128, **{'pass': True})
        with self.assertRaises(ValueError): release.require_panel(*args)

    def test_wrong_roles_quotas_and_training_statistics_are_rejected(self):
        args = self.panel(); args[0]['ph019']['role'] = 'train'
        with self.assertRaises(ValueError): release.require_panel(*args)
        args = self.panel(); args[0]['ph007']['pairs'] = 127
        with self.assertRaises(ValueError): release.require_panel(*args)
        args = self.panel(); args[1]['fit_phases'][-1] = 'ph019'
        with self.assertRaises(ValueError): release.require_panel(*args)
        args = self.panel(); args[1]['phase_weight'] = 1/12
        with self.assertRaises(ValueError): release.require_panel(*args)

    def test_partial_or_predictively_opened_interface_cannot_qualify(self):
        for key, value in [('pairs', 1791), ('offset_cases', 11775), ('pass', False),
                           ('science_scores_evaluated', True)]:
            args = self.panel(); args[2][key] = value
            with self.assertRaises(ValueError): release.require_panel(*args)

    def test_missing_prerequisites_fail_without_triggering_builds(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(release.coord, 'ROOT', Path(tmp)), \
                patch.object(release.c, 'require_compute'), patch.object(release.coord, 'require_host_checks'), \
                patch.object(release.norm, 'fit', side_effect=AssertionError('premature normalization')):
            with self.assertRaises(FileNotFoundError): release.run()
            self.assertFalse((Path(tmp) / 'data_release').exists())

    def test_physical_panel_requires_every_diagnostic_offset(self):
        cfg = {'phases': ['ph007', 'ph008']}
        cases = [dict(phase=phase, cap=cap, shell=shell, support_stratum=kind, offset_raw=offset)
                 for phase in cfg['phases'] for cap in ('NGC', 'SGC') for shell in range(4)
                 for kind in ('interior', 'boundary')
                 for offset in release.audit.op.layout()['context_offsets_raw']]
        record = dict(cases=cases, no_posterior_or_predictive_scoring=True)
        release.require_physical_panel(record, cfg)
        record['cases'] = cases[:-1] + [cases[-2]]
        with self.assertRaises(ValueError): release.require_physical_panel(record, cfg)

    def test_sealed_phase_is_rejected_before_audit_lookup(self):
        with patch.object(release.queue, 'qualified', side_effect=AssertionError('sealed phase lookup')):
            for phase in ('ph001', 'ph006'):
                with self.assertRaises(PermissionError): release.phase_evidence(phase)


if __name__ == '__main__':
    unittest.main()
