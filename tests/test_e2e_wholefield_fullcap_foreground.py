from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from workflows.sbi.e2e_wholefield_fullcap_foreground import execute


class ForegroundTests(unittest.TestCase):
    def fake(self):
        return SimpleNamespace(compute=Mock(), read=Mock(return_value={'gpus': 1}),
                               start=Mock(), write=Mock(), worker=Mock(), verify_record=Mock())

    def test_success(self):
        gen = self.fake(); execute(gen, 's004', {'sha256': 'fixture'})
        gen.start.assert_called_once_with('s004')
        gen.worker.assert_called_once_with('s004', 0)
        self.assertEqual(gen.write.call_args.args[0].name, 'FOREGROUND_COMPLETE.json')

    def test_worker_failure_propagates(self):
        gen = self.fake(); gen.worker.side_effect = RuntimeError('fixture failure')
        with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
            execute(gen, 's004', {'sha256': 'fixture'})
        self.assertEqual(gen.write.call_args.args[0].name, 'FOREGROUND_FAILED.json')
        gen.verify_record.assert_not_called()

    def test_no_scale_up(self):
        gen = self.fake(); gen.read.return_value = {'gpus': 4}
        with self.assertRaises(ValueError):
            execute(gen, 's004', {})
        gen.start.assert_not_called(); gen.worker.assert_not_called()


if __name__ == '__main__':
    unittest.main()
