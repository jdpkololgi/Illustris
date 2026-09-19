from pathlib import Path
import unittest
from workflows.sbi import e2e_coupled_gpu_allocation as launch


class TechnicalAllocationTests(unittest.TestCase):
    def test_no_third_job_or_premature_predecessor_handoff(self):
        self.assertFalse(launch.can_request([['12','RUNNING']], '12'))
        self.assertFalse(launch.can_request([['13','RUNNING'],['14','PENDING']], '12'))
        self.assertTrue(launch.can_request([['13','RUNNING']], '12'))
        self.assertTrue(launch.can_request([], '12'))

    def test_gpu_budget_counts_reserved_and_spent_not_cpu(self):
        usage={'allocations':[
            dict(kind='cpu',gpus=0,hours_cap=4,elapsed_seconds=3600,state='RUNNING'),
            dict(kind='gpu',gpus=1,hours_cap=1,elapsed_seconds=600,state='RUNNING'),
            dict(kind='gpu',gpus=1,hours_cap=1,elapsed_seconds=1800,state='FAILED')]}
        self.assertEqual(launch.reserved_gpu_hours(usage),1.5)

    def test_exact_single_hour_shared_gpu_no_batch_or_successor(self):
        command=launch.command(Path('/fixture'))
        self.assertEqual(command[0],'salloc')
        for flag in ('--gpus=1','--time=01:00:00','--qos=shared_interactive',
                     '--constraint=gpu&hbm80g','--account=desi_g','--licenses=scratch',
                     '--immediate=600'):
            self.assertIn(flag,command)
        self.assertTrue(command[-2].endswith('/e2e_coupled_gpu_benchmark_step.sh'))


if __name__ == '__main__':
    unittest.main()
