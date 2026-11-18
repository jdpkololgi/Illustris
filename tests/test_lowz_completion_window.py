import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from workflows.sbi import run_lowz_completion_window as runner


class CompletionWindowTests(unittest.TestCase):
    def test_november_continuation_allows_only_fifth_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for job in ('1','2','3','4'):
                (root/'completion_windows'/job).mkdir(parents=True)
            with patch.object(runner.c,'ROOT',root),patch.object(runner.c,'require_compute'),\
                    patch.object(runner.campaign,'authority'),patch.dict(os.environ,SLURM_JOB_ID='5'):
                self.assertEqual(runner.Window(300,continuation_20261102=True).allocation_limit,5)
            with patch.object(runner.c,'ROOT',root),patch.object(runner.c,'require_compute'),\
                    patch.object(runner.campaign,'authority'),patch.dict(os.environ,SLURM_JOB_ID='6'):
                with self.assertRaises(PermissionError):runner.Window(300,continuation_20261102=True)

    def test_new_continuation_is_one_window_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for job in ('60483945','60522523','60522907'):
                (root/'completion_windows'/job).mkdir(parents=True)
            with patch.object(runner.c,'ROOT',root),patch.object(runner.c,'require_compute'),\
                    patch.object(runner.campaign,'authority'),patch.dict(os.environ,SLURM_JOB_ID='fourth'):
                with self.assertRaises(PermissionError):runner.Window(13900,continuation_20261030=True)
                window=runner.Window(300,continuation_20261030=True)
                self.assertEqual(window.allocation_limit,4)
            with patch.object(runner.c,'ROOT',root),patch.object(runner.c,'require_compute'),\
                    patch.object(runner.campaign,'authority'),patch.dict(os.environ,SLURM_JOB_ID='fifth'):
                with self.assertRaises(PermissionError):runner.Window(300,continuation_20261030=True)

    def test_explicit_replacement_allows_only_one_extra_allocation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for job in ('60483945','60522523'):
                (root/'completion_windows'/job).mkdir(parents=True)
            with patch.object(runner.c,'ROOT',root),patch.object(runner.c,'require_compute'),\
                    patch.object(runner.campaign,'authority'),patch.dict(os.environ,SLURM_JOB_ID='replacement'):
                window=runner.Window(300,'60522523')
                self.assertEqual(window.allocation_limit,3)
            with patch.object(runner.c,'ROOT',root),patch.object(runner.c,'require_compute'),\
                    patch.object(runner.campaign,'authority'),patch.dict(os.environ,SLURM_JOB_ID='fourth'):
                with self.assertRaises(PermissionError):runner.Window(300,'60522523')
                with self.assertRaises(PermissionError):runner.Window(300,'unapproved')

    def test_waits_for_both_processes_even_when_first_fails(self):
        from workflows.sbi.launch_lowz_replacement import wait_for_all
        from unittest.mock import Mock
        first=Mock();first.wait.return_value=1
        second=Mock();second.wait.return_value=0
        self.assertEqual(wait_for_all({'p12':first,'cfm':second}),{'p12':1,'cfm':0})
        first.wait.assert_called_once_with();second.wait.assert_called_once_with()

    def test_allocation_count_includes_incomplete_windows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for job in ('1','2'):
                (root/'completion_windows'/job).mkdir(parents=True)
            with patch.object(runner.c,'ROOT',root),patch.object(runner.c,'require_compute'),\
                    patch.object(runner.campaign,'authority'),patch.dict(os.environ,SLURM_JOB_ID='3'):
                with self.assertRaises(PermissionError):runner.Window(300)
            with patch.object(runner.c,'ROOT',root),patch.object(runner.c,'require_compute'),\
                    patch.object(runner.campaign,'authority'),patch.dict(os.environ,SLURM_JOB_ID='2'):
                first=runner.Window(300);second=runner.Window(300)
                self.assertNotEqual(first.directory,second.directory)

    def test_failed_stage_is_never_relaunched(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with patch.object(runner.c,'ROOT',root),patch.object(runner.c,'require_compute'),\
                    patch.object(runner.campaign,'authority'),patch.dict(os.environ,SLURM_JOB_ID='1'):
                window=runner.Window(300)
                (window.registry/'failed_FAILED.json').write_text(json.dumps({'exitcode':1}))
                with patch.object(runner.subprocess,'run') as launch:
                    self.assertFalse(window.stage('failed','never.imported'))
                    launch.assert_not_called()
                self.assertIn('no automatic retry',window.stop['reason'])


if __name__=='__main__':unittest.main()
