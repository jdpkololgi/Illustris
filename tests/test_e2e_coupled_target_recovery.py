from pathlib import Path
import unittest
from workflows.sbi import e2e_coupled_target_recovery as recovery


def fixture():
    lines = ['58552205|FAILED|1:0|salloc']
    for suffix, module in recovery.EXPECTED.items():
        state, code = ('OUT_OF_MEMORY', '0:125') if suffix == '0' else ('COMPLETED', '0:0')
        lines.append(f'58552205.{suffix}|{state}|{code}|srun python -m workflows.sbi.{module} --seconds 13800')
    return '\n'.join(lines)


class TargetRecoveryTests(unittest.TestCase):
    def test_exact_reviewed_failure(self):
        self.assertTrue(recovery.review(fixture())['reviewed_target_oom'])
        self.assertTrue(recovery.review(fixture().replace('COMPLETED|0:0', 'FAILED|75:0')))

    def test_running_or_unexpected_parent_rejected(self):
        for state in ('RUNNING|0:0', 'FAILED|2:0', 'COMPLETED|0:0'):
            with self.assertRaises(RuntimeError):
                recovery.review(fixture().replace('FAILED|1:0', state))

    def test_additional_failure_or_wrong_oom_rejected(self):
        for old, new in [('COMPLETED|0:0', 'FAILED|1:0'),
                         ('OUT_OF_MEMORY|0:125', 'FAILED|1:0'),
                         ('--kind targets', '--kind observations')]:
            with self.assertRaises(RuntimeError):
                recovery.review(fixture().replace(old, new))

    def test_early_recovery_only_after_target_oom_and_healthy_other_workers(self):
        running = fixture().replace('FAILED|1:0', 'RUNNING|0:0').replace('COMPLETED|0:0', 'RUNNING|0:0')
        self.assertFalse(recovery.review(running, allow_running=True)['predecessor_terminal'])
        with self.assertRaises(RuntimeError):
            recovery.review(running)
        with self.assertRaises(RuntimeError):
            recovery.review(running.replace('OUT_OF_MEMORY|0:125', 'RUNNING|0:0'), allow_running=True)
        with self.assertRaises(RuntimeError):
            recovery.review(running.replace('58552205.1|RUNNING|0:0', '58552205.1|FAILED|1:0'), allow_running=True)

    def test_missing_or_extra_step_rejected(self):
        with self.assertRaises(RuntimeError):
            recovery.review('\n'.join(fixture().splitlines()[:-1]))
        with self.assertRaises(RuntimeError):
            recovery.review(fixture() + '\n58552205.6|COMPLETED|0:0|unknown')

    def test_single_two_hour_cpu_request(self):
        cmd = recovery.command(Path('/fixture'))
        self.assertEqual(cmd[0], 'salloc')
        for flag in ('--time=02:00:00', '--mem=0', '--account=desi',
                     '--qos=interactive', '--immediate=600', '--licenses=scratch'):
            self.assertIn(flag, cmd)

    def test_native_precedes_fft_and_concurrent_budget(self):
        script = (recovery.c.REPO / 'workflows/sbi/e2e_coupled_target_recovery_step.sh').read_text()
        self.assertLess(script.index('e2e_coupled_native_pool'), script.index('pids=()'))
        self.assertIn('--memory-gib 416 --cpus 72', script)
        self.assertIn('--mem=400G', script)
        self.assertIn('--publish-data-release', script)
        self.assertLess(script.index('wait_for_writers()'), script.index('e2e_coupled_audit_worker'))


if __name__ == '__main__':
    unittest.main()
