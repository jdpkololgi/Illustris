import signal
import unittest
from unittest.mock import patch
from workflows.sbi import e2e_coupled_products_lifetime_guard as guard


class ProductLifetimeGuardTests(unittest.TestCase):
    def record(self):
        return dict(pid=guard.PID, ppid=1, start_ticks=123,
                    argv=['/bin/bash', str(guard.SCRIPT), str(guard.SNAPSHOT)])

    def test_only_exact_wrapper_and_allocation_parent_are_accepted(self):
        r = self.record()
        p = dict(pid=1, argv=['salloc', '--job-name=coupled-products-01'])
        guard.validate_launcher(r, p)
        with self.assertRaises(PermissionError): guard.validate_launcher(dict(r, pid=1), p)
        with self.assertRaises(PermissionError): guard.validate_launcher(r, dict(p, pid=2))

    def test_error_after_stop_always_resumes_same_wrapper(self):
        r = self.record()
        with patch.object(guard, 'identity', return_value=r), patch.object(guard.os, 'kill') as kill, \
                patch.object(guard, 'step_states', side_effect=RuntimeError('test accounting failure')):
            with self.assertRaises(RuntimeError): guard.hold(r)
        self.assertEqual(kill.call_args_list[0].args, (guard.PID, signal.SIGSTOP))
        self.assertEqual(kill.call_args_list[-1].args, (guard.PID, signal.SIGCONT))

    def test_no_signal_after_identity_changes(self):
        with patch.object(guard, 'identity', return_value={}), patch.object(guard.os, 'kill') as kill:
            with self.assertRaises(RuntimeError): guard.hold(self.record())
            kill.assert_not_called()

    def test_every_attached_step_must_be_terminal(self):
        states = {step: ['COMPLETED', '0:0'] for step in guard.STEPS}
        self.assertTrue(guard.all_terminal(states))
        states[guard.JOB + '.8'] = ['RUNNING', '0:0']
        self.assertFalse(guard.all_terminal(states))
