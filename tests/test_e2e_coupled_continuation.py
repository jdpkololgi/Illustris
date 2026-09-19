from types import SimpleNamespace
from pathlib import Path
import unittest
from unittest.mock import patch
from workflows.sbi import e2e_coupled_post_allocation as launch


class ContinuationTests(unittest.TestCase):
    def test_wait_is_bounded_before_source_or_scheduler_access(self):
        for duration in (0, -1, 14401, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                launch.run('12',Path('/not-a-source'),wait_seconds=duration)

    def test_only_verified_success_or_checkpoint_pause_allows_successor(self):
        for state,code in [('COMPLETED','0:0'),('FAILED','75:0')]:
            with patch.object(launch.subprocess,'run',return_value=SimpleNamespace(stdout=f'12|{state}|{code}|\n')):
                launch.require_planned_terminal('12')
        for state,code in [('FAILED','1:0'),('TIMEOUT','0:15'),('RUNNING','0:0'),('CANCELLED','0:0')]:
            with patch.object(launch.subprocess,'run',return_value=SimpleNamespace(stdout=f'12|{state}|{code}|\n')):
                with self.assertRaises(RuntimeError): launch.require_planned_terminal('12')
        with patch.object(launch.subprocess,'run',return_value=SimpleNamespace(stdout='')):
            with self.assertRaises(RuntimeError): launch.require_planned_terminal('12')


if __name__=='__main__': unittest.main()
