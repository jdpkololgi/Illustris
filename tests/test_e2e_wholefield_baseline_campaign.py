import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from workflows.sbi import e2e_wholefield_baseline_campaign as c
from workflows.sbi import e2e_wholefield_baseline_campaign_v2 as v2


class CampaignTests(unittest.TestCase):
    def test_nersc_expanded_qos_names(self):
        output = '1|gpu_interactive|RUNNING\n2|gpu_shared_interactive|PENDING\n3|cpu_interactive|RUNNING\n4|interactive|RUNNING\n5|shared_interactive|RUNNING\n6|regular|RUNNING\n'
        self.assertEqual(len(v2.parse_interactive_jobs(output)), 5)
        self.assertEqual(v2.parse_interactive_jobs(''), [])
        with self.assertRaises(ValueError): v2.parse_interactive_jobs('unexpected')

    def test_adoption_requires_successful_accounting(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(v2.c, 'ROOT', Path(tmp)):
            binding = {'adopted_segments': [{'segment': 's020', 'job': '2'}]}
            self.assertFalse(v2.adopted_complete(binding))
            c.write(Path(tmp)/'allocations/s020/ACCOUNTED.json', {'state': 'COMPLETED', 'exit_code': '0:0', 'job': '2'})
            self.assertFalse(v2.adopted_complete(binding))
            c.write(Path(tmp)/'allocations/s020/BASELINE_OWNER_COMPLETE.json', {'passed': True})
            self.assertTrue(v2.adopted_complete(binding))
            c.write(Path(tmp)/'allocations/s020/RANK_0_FAILED.json', {'failed': True})
            with self.assertRaises(RuntimeError): v2.adopted_complete(binding)

    def test_resources(self):
        for count, gpus in ((1, 1), (2, 2), (3, 2), (4, 4), (119, 4)):
            self.assertEqual(c.resource_plan('generate', count), (gpus, 10800))
        self.assertEqual(c.resource_plan('extract', 0), (1, 1800))
        self.assertEqual(c.resource_plan('smoke', 0), (1, 600))
        for mode in ('train', 'gate2', 'retry'):
            with self.assertRaises(ValueError): c.resource_plan(mode, 1)
        with self.assertRaises(ValueError): c.resource_plan('generate', 0)

    def test_explicit_slurm_topology(self):
        for count in (1, 2, 4):
            cmd = c.allocation_command('s019', dict(mode='generate', gpus=count, seconds=10800))
            self.assertEqual(cmd[0], 'salloc')
            self.assertIn('srun', cmd)
            self.assertNotIn('sbatch', cmd)
            self.assertIn('--licenses=scratch', cmd)
            self.assertIn('--immediate=600', cmd)
            self.assertIn('--gpus-per-task=1', cmd)
            self.assertIn(f'--gpus={count}', cmd)
            self.assertIn('--qos='+('interactive' if count == 4 else 'shared_interactive'), cmd)

    def test_environment_isolation(self):
        with patch.dict(c.os.environ, {'PYTHONPATH': 'bad', 'PYTHONHOME': 'bad', 'LD_PRELOAD': 'bad'}):
            env = c.environment()
        self.assertEqual(env['PYTHONNOUSERSITE'], '1')
        self.assertFalse(any(k in env for k in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE', 'LD_PRELOAD')))

    def test_smoke_before_generation(self):
        self.assertEqual(c.choose_work(False, [], ['a'], 119, 73, 65), ('smoke', ['a'], 0))
        self.assertIsNone(c.choose_work(False, ['smoke'], ['a'], 119, 73, 65))
        with self.assertRaises(RuntimeError): c.choose_work(False, [], [], 119, 73, 65)

    def test_no_duplicate_extraction(self):
        backlog = [str(i) for i in range(20)]
        self.assertEqual(c.choose_work(True, [], backlog, 100, 92, 72), ('extract', backlog[:16], 0))
        self.assertEqual(c.choose_work(True, ['extract'], backlog, 100, 92, 72), ('generate', [], 100))
        self.assertIsNone(c.choose_work(True, ['extract'], backlog, 0, 192, 172))

    def test_bank_only_after_complete(self):
        self.assertEqual(c.choose_work(True, [], [], 0, 192, 192), ('bank', [], 0))
        for generated, extracted in ((191, 191), (192, 191), (191, 192)):
            with self.assertRaises(RuntimeError): c.choose_work(True, [], [], 0, generated, extracted)
        self.assertIsNone(c.choose_work(True, ['generate'], [], 0, 191, 191))
        self.assertEqual(c.choose_work(True, [], ['last'], 0, 192, 191), ('extract', ['last'], 0))

    def interrupted_fixture(self, root):
        key = 'ph014/NGC/seed29/d000'
        c.write(root/'MANIFEST.json', {'tasks': [{'key': key}]})
        c.write(root/'tasks'/key/'START.json', {'key': key})
        c.write(root/'tasks'/key/'density/BINDING.json', {'bound': True})
        c.write(root/'claims'/key/'OWNER.json', {'segment': 's001', 'job': '1'})
        c.write(root/'allocations/s001/ACCOUNTED.json', {'state': 'TIMEOUT', 'job': '1'})
        return key

    def test_retry_only_terminal_metadata(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(c, 'ROOT', Path(tmp)):
            key = self.interrupted_fixture(Path(tmp))
            rows = c.retry_candidates()
            self.assertEqual([r['key'] for r in rows], [key])
            self.assertEqual(len(rows[0]['original_files']), 3)
            c.write(Path(tmp)/'tasks'/key/'density/PAYLOAD.json', {'do_not_move': True})
            with self.assertRaises(RuntimeError): c.retry_candidates()

    def test_retry_rejects_live_or_wrong_job(self):
        for state, job in (('RUNNING', '1'), ('TIMEOUT', '2')):
            with tempfile.TemporaryDirectory() as tmp, patch.object(c, 'ROOT', Path(tmp)):
                self.interrupted_fixture(Path(tmp))
                original = c.read
                def altered(path):
                    if str(path).endswith('ACCOUNTED.json'): return {'state': state, 'job': job}
                    return original(path)
                with patch.object(c, 'read', side_effect=altered):
                    with self.assertRaises(RuntimeError): c.retry_candidates()

    def test_exclusive_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)/'receipt.json'; c.write(dest, {'preserve': True})
            with self.assertRaises(FileExistsError): c.write(dest, {'overwrite': True})
            self.assertEqual(c.read(dest), {'preserve': True})


if __name__ == '__main__':
    unittest.main()
