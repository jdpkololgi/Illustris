import unittest
from workflows.sbi import e2e_coupled_qualification_tail as tail


class QualificationHandoffTests(unittest.TestCase):
    expected = {'123.3': 'e2e_coupled_audit_worker',
                '123.4': 'e2e_coupled_normalization_worker'}

    def rows(self, states=(('COMPLETED', '0:0'), ('FAILED', '75:0'))):
        return '\n'.join(f'{step}|{state}|{code}|srun python -u -m workflows.sbi.{module} --seconds 10'
                         for (step, module), (state, code) in zip(self.expected.items(), states))

    def test_only_success_or_planned_checkpoint_pause(self):
        self.assertTrue(tail.predecessors_ready(self.rows(), self.expected))

    def test_live_missing_and_accounting_lag_wait(self):
        self.assertFalse(tail.predecessors_ready('', self.expected))
        self.assertFalse(tail.predecessors_ready(self.rows().splitlines()[0], self.expected))
        self.assertFalse(tail.predecessors_ready(
            self.rows((('RUNNING', '0:0'), ('FAILED', '75:0'))), self.expected))

    def test_failure_or_cancel_does_not_start_second_writer(self):
        for bad in (('FAILED', '1:0'), ('CANCELLED', '0:15'), ('TIMEOUT', '0:0')):
            with self.assertRaises(RuntimeError):
                tail.predecessors_ready(self.rows((bad, ('COMPLETED', '0:0'))), self.expected)

    def test_expected_worker_identity_and_explicit_step(self):
        with self.assertRaises(ValueError):
            tail.predecessors_ready(self.rows().replace('e2e_coupled_audit_worker', 'different'), self.expected)
        with self.assertRaises(ValueError):
            tail.predecessors_ready('', {'123.extern': 'e2e_coupled_audit_worker'})


if __name__ == '__main__':
    unittest.main()
