import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from workflows.sbi.e2e_wholefield_fullcap_batch import execute


class BatchTests(unittest.TestCase):
    def fake(self, mode='production', completed=None):
        def read(path):
            if path.name == 'LAUNCHER.json':
                return dict(mode=mode, sources=[{}], reservation={})
            if path.name == 'RESERVED.json':
                return dict(gpus=1)
            if path.name == 'WORKER_0.json':
                return dict(passed=True, completed=['task'] if completed is None else completed)
            raise AssertionError(path)
        return SimpleNamespace(compute=Mock(), read=read, verify_record=Mock(),
                               start=Mock(), write=Mock(), worker=Mock(), record=Mock(return_value={}))

    def run_with(self, gen, batch=True):
        env = dict(SLURM_JOB_ID='123', SLURM_NTASKS='1')
        if batch:
            env['SLURM_BATCH_HOST'] = 'nidtest'
        with patch.dict(os.environ, env, clear=True):
            execute(gen, 'test', Mock())

    def test_production_success(self):
        gen = self.fake(); self.run_with(gen)
        gen.worker.assert_called_once_with('test', 0)
        self.assertEqual(gen.write.call_args.args[0].name, 'BATCH_COMPLETE.json')
        self.assertTrue(gen.write.call_args.args[1]['scientific_completion'])

    def test_interactive_production_rejected(self):
        gen = self.fake()
        with self.assertRaisesRegex(RuntimeError, 'scheduler-owned'):
            self.run_with(gen, batch=False)
        gen.start.assert_not_called(); gen.worker.assert_not_called()

    def test_worker_failure_propagates(self):
        gen = self.fake(); gen.worker.side_effect = RuntimeError('injected')
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            self.run_with(gen)
        self.assertEqual(gen.write.call_args.args[0].name, 'BATCH_FAILED.json')

    def test_explicit_supervised_interactive_generates(self):
        gen = self.fake('supervised-interactive'); self.run_with(gen, batch=False)
        gen.worker.assert_called_once_with('test', 0)
        self.assertTrue(gen.write.call_args.args[1]['scientific_completion'])

    def test_supervised_mode_rejects_batch(self):
        gen = self.fake('supervised-interactive')
        with self.assertRaisesRegex(RuntimeError, 'cannot run as batch'):
            self.run_with(gen)
        gen.start.assert_not_called()

    def test_empty_worker_is_not_success(self):
        with self.assertRaisesRegex(RuntimeError, 'No completed'):
            self.run_with(self.fake(completed=[]))

    def test_smoke_never_generates(self):
        gen = self.fake('smoke-pass'); self.run_with(gen, batch=False)
        gen.worker.assert_not_called()
        self.assertFalse(gen.write.call_args.args[1]['scientific_completion'])

    def test_deliberate_failure(self):
        gen = self.fake('smoke-fail')
        with self.assertRaisesRegex(RuntimeError, 'Deliberate'):
            self.run_with(gen, batch=False)
        gen.worker.assert_not_called()
        self.assertEqual(gen.write.call_args.args[0].name, 'BATCH_FAILED.json')

    def test_changed_source_rejected(self):
        gen = self.fake(); gen.verify_record.side_effect = ValueError('changed')
        with self.assertRaises(ValueError):
            self.run_with(gen)
        gen.start.assert_not_called()


if __name__ == '__main__':
    unittest.main()
