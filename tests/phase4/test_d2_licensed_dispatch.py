import copy
import tempfile
from pathlib import Path
import unittest

from workflows.sbi.dispatch_p12f3_d2_after_primary import actions_for, digest, exclusive_json


class DispatchTests(unittest.TestCase):
    def inputs(self, passed=False, convergence=True):
        common = dict(pass_=True, ph001_opened=False, selected_arm='modern_base4',
                      selected_presentations=2500, selected_weights='ema', contract_digest='contract')
        common['pass'] = common.pop('pass_')
        decision = {**common, 'schema_version': 'p12f3-d2-ph006-seed-decision-v1',
                    'seed': 42, 'seed_role': 'primary', 'seed_pass': passed,
                    'sampler_convergence_nfe50_nfe100': {'pass': convergence},
                    'frozen_inputs': {'contract_digest': 'contract'}}
        decision['frozen_digest'] = digest(decision['frozen_inputs'])
        replication = {**common, 'schema_version': 'p12f3-d2-second-seed-license-v1',
                       'licensed': passed, 'seed42_decision_sha256': 'decision'}
        stochastic = {**common, 'schema_version': 'p12f3-d2-stochastic-control-license-v1',
                      'licensed': not convergence, 'seed_decision_sha256': 'decision'}
        return [decision, replication, stochastic, 'decision', 'contract']

    def test_science_failure_stops(self):
        self.assertEqual(actions_for(*self.inputs()), [])

    def test_numerical_failure_only_runs_diagnostic(self):
        self.assertEqual(actions_for(*self.inputs(convergence=False)), ['stochastic-control modern_base4'])

    def test_pass_runs_one_replication_and_combined_decision(self):
        actions = actions_for(*self.inputs(passed=True))
        self.assertEqual(len(actions), 7)
        self.assertTrue(actions[0].startswith('replicate modern_base4 '))
        self.assertEqual(actions[-1], 'decide-combined')

    def test_stale_or_unsafe_licence_rejected(self):
        for key, value in [('seed42_decision_sha256', 'stale'), ('licensed', True),
                           ('selected_arm', 'modern_base8'), ('ph001_opened', True)]:
            values = copy.deepcopy(self.inputs())
            values[1][key] = value
            with self.assertRaises(ValueError):
                actions_for(*values)

    def test_missing_scientific_pass_not_technical_pass(self):
        values = self.inputs()
        del values[0]['seed_pass']
        with self.assertRaises(ValueError):
            actions_for(*values)

    def test_duplicate_claim_refused(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'claim.json'
            exclusive_json(path, {'a': 1})
            with self.assertRaises(FileExistsError):
                exclusive_json(path, {'a': 2})


if __name__ == '__main__':
    unittest.main()
