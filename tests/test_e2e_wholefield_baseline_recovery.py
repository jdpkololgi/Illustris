from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

from workflows.sbi import e2e_wholefield_baseline_campaign_v3 as v3


class RecoveryTests(unittest.TestCase):
    def authority(self):
        return dict(allocated_gpu_hour_limit=None, supersedes_allocated_gpu_hour_limit=250,
                    maximum_simultaneous_allocations=2, maximum_segment_seconds=10800,
                    deadline_utc='2026-10-23T19:17:04+00:00', interactive_only=True)

    def test_resource_waiver_retains_limits(self):
        now = datetime(2026, 10, 12, tzinfo=timezone.utc).timestamp()
        v3.resource_admit(self.authority(), 4, 10800, now)
        for gpus, seconds in ((8, 10800), (1, 0), (1, 14400)):
            with self.assertRaises(PermissionError): v3.resource_admit(self.authority(), gpus, seconds, now)
        for field, value in (('allocated_gpu_hour_limit', 250), ('interactive_only', False),
                             ('maximum_simultaneous_allocations', 3)):
            authority = {**self.authority(), field: value}
            with self.assertRaises(PermissionError): v3.resource_admit(authority, 1, 3600, now)
        deadline = datetime.fromisoformat(self.authority()['deadline_utc']).timestamp()
        with self.assertRaises(PermissionError): v3.resource_admit(self.authority(), 1, 3600, deadline-4000)

    def identity(self):
        task = dict(phase='ph013', cap='NGC', seed=29, first=0)
        record = dict(passed=True, phase='ph013', cap='NGC', first=0, draws=4)
        binding = dict(record, model_seed=29, nfe=128, checkpoints={'coarse':'c','fine':'f'},
                       normalizer='n', prepared_sha256='p', comparison_sha256='manifest',
                       benchmark=False, freeze_sha256=None)
        manifest = dict(checkpoints={'29':binding['checkpoints']}, normalizer='n', prepared={'ph013/NGC':'p'})
        return task, record, binding, manifest

    def test_density_provenance(self):
        args = self.identity()
        with patch.object(v3.c, 'digest', return_value='manifest'):
            v3.validate_density_identity(*args)
            for field, value in (('phase','ph018'), ('model_seed',17), ('nfe',64),
                                 ('benchmark',True), ('freeze_sha256','sealed'), ('normalizer','changed')):
                changed = {**args[2], field:value}
                with self.assertRaises(ValueError):
                    v3.validate_density_identity(args[0], args[1], changed, args[3])

    def test_recovery_before_generation_and_no_duplicate_extraction(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(v3.c, 'ROOT', Path(tmp)):
            manifest = {'recoveries':[{'key':'ph013/NGC/seed29/d000'}]}
            with patch.object(v3.c, 'read', return_value=manifest):
                self.assertEqual(v3.choose_work(False, [], [], 79, 105, 105),
                                 ('smoke', ['ph013/NGC/seed29/d000'], 0))
                self.assertIsNone(v3.choose_work(False, ['smoke'], [], 79, 105, 105))
                self.assertEqual(v3.choose_work(True, [], [], 79, 106, 106)[0], 'recover')
                self.assertEqual(v3.choose_work(True, ['recover'], ['x']*8, 79, 113, 105)[0], 'generate')

    def test_explicit_reservation_above_old_ceiling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); campaign=root/'campaign'; campaign.mkdir()
            v3.c.write(campaign/'MANIFEST.json', {'bound': True})
            v3.c.write(campaign/'RESOURCE_AUTHORITY.json', self.authority())
            v3.c.write(root/'allocations/s001/RESERVED.json', {'reserved_gpu_hours':300})
            v3.c.write(root/'allocations/s001/ACCOUNTED.json', {'allocated_gpu_hours':300})
            resource={'path':str(campaign/'RESOURCE_AUTHORITY.json'), 'sha256':'authority'}
            gen=Mock(); gen.record.return_value={'path':'original-scientific-manifest','sha256':'frozen'}
            with patch.object(v3.c,'ROOT',root), patch.object(v3.c,'CAMPAIGN',campaign), \
                 patch.object(v3.c,'verify_sources',return_value={'resources':resource}), \
                 patch.object(v3.c,'frozen',return_value=gen), \
                 patch.object(v3.c,'interactive_jobs',return_value=[]), \
                 patch.object(v3.c.subprocess,'Popen') as launch, \
                 patch.object(v3.time,'time',return_value=datetime(2026,10,12,tzinfo=timezone.utc).timestamp()):
                segment, _, mode = v3.reserve_launch('recover',['key'],0)
            reservation=v3.c.read(root/'allocations'/segment/'RESERVED.json')
            self.assertEqual(reservation['previously_booked_gpu_hours'],300)
            self.assertEqual(reservation['resource_amendment'],resource)
            self.assertEqual(reservation['manifest']['sha256'],'frozen')
            self.assertEqual(mode,'recover'); launch.assert_called_once()
            gen.reserve.assert_not_called()


if __name__ == '__main__': unittest.main()
