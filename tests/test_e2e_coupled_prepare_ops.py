import unittest
import re
from workflows.sbi import e2e_coupled_prepare_ops as ops


class PreparationResourceTests(unittest.TestCase):
    def test_final_data_step_covers_only_the_remaining_authorized_native_route(self):
        script=(ops.c.REPO/'workflows/sbi/e2e_coupled_final_data_step.sh').read_text()
        phases=set(re.search(r'--phases ([^\\\n]+)',script)[1].split())
        self.assertEqual(phases,set(ops.c.ROLES)-{'ph000','ph002','ph003'})
        self.assertIn('require_planned_terminal("58552205")',script)
        self.assertIn('--publish-data-release',script)

    def test_wrappers_guard_the_sum_of_all_simultaneous_steps(self):
        for name in ('cpu_continue', 'products_continue', 'final_data'):
            path = ops.c.REPO / 'workflows/sbi' / f'e2e_coupled_{name}_step.sh'
            script = path.read_text()
            requests = re.findall(r'srun --exact -N1 -n1 -c(\d+) --mem=(\d+)G', script)
            self.assertTrue(requests)
            cpus = sum(int(cpu) for cpu, _ in requests)
            memory = sum(int(mem) for _, mem in requests)
            self.assertIn(f'--memory-gib {memory} --cpus {cpus}', script)

    def test_actual_allocation_memory_not_physical_node_memory(self):
        info = 'JobId=12 JobState=RUNNING NumNodes=1 AllocTRES=cpu=256,mem=487802M,node=1'
        result = ops.validate_step_budget(info, 472, 124)
        self.assertEqual(result['requested_step_mib'], 483328)
        with self.assertRaises(RuntimeError):
            ops.validate_step_budget(info, 488, 124)
        with self.assertRaises(RuntimeError):
            ops.validate_step_budget(info, 472, 257)

    def test_units_and_running_single_node_requirement(self):
        info = 'JobState=RUNNING NumNodes=1 AllocTRES=cpu=128,mem=476.5G,node=1'
        self.assertGreater(ops.validate_step_budget(info, 472, 124)['allocated_mib'], 483328)
        for changed in (info.replace('RUNNING', 'PENDING'), info.replace('NumNodes=1', 'NumNodes=2')):
            with self.assertRaises(ValueError):
                ops.validate_step_budget(changed, 472, 124)


if __name__ == '__main__':
    unittest.main()
