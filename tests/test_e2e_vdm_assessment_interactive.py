import unittest
from pathlib import Path
from workflows.sbi import e2e_vdm_assessment_interactive as launch


class InteractiveLaunchTests(unittest.TestCase):
    def test_four_gpu_allocation_is_bounded(self):
        command = launch.allocation_command(Path('/tmp/root'),Path('/tmp/launcher.py'),0)
        self.assertEqual(command[0],'salloc')
        for flag in ('--gpus=4','--ntasks=4','--qos=interactive','--time=01:15:00','--immediate=600'):
            self.assertIn(flag,command)
        self.assertEqual(launch.SEGMENTS,2)

    def test_exclusive_single_gpu_workers_and_clean_pause(self):
        for branch in launch.BRANCHES:
            command = launch.worker_command(Path('/tmp/root'),branch)
            for flag in ('--gpus=1','--ntasks=1','--exclusive','--exact','--signal=USR1','--kill-after=120'):
                self.assertIn(flag,command)
        self.assertEqual(len(set(launch.BRANCHES)),4)
        with self.assertRaises(ValueError):
            launch.worker_command(Path('/tmp/root'),'not-a-branch')

    def test_only_clean_pause_can_resume(self):
        self.assertEqual(launch.disposition([0,0,0,0]),0)
        self.assertEqual(launch.disposition([0,75,75,0]),75)
        for bad in (1,124,137,-15):
            self.assertEqual(launch.disposition([75,75,75,bad]),1)


if __name__ == '__main__':
    unittest.main()
