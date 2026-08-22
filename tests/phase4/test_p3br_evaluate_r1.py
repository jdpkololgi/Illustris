import unittest

from workflows.abacus_tweb.p3br_evaluate_r1 import promotion_decision
from workflows.abacus_tweb.p8_deterministic_common import SHELL_NAMES


def report(macro, shells):
    return {
        "primary_macro_r2_lambda1": macro,
        "per_shell": {
            name: {"lambda1": {"r2": value}}
            for name, value in zip(SHELL_NAMES, shells)
        },
    }


class TestP3brEvaluateR1(unittest.TestCase):
    def test_large_clean_gain_opens_replication(self):
        decision = promotion_decision(
            report(0.50, [0.55, 0.53, 0.49, 0.43]),
            report(0.54, [0.59, 0.57, 0.53, 0.47]),
        )
        self.assertTrue(decision["seed42_promotion_candidate"])
        self.assertTrue(decision["second_seed_required_before_final_freeze"])

    def test_shell_degradation_vetoes_large_macro_gain(self):
        decision = promotion_decision(
            report(0.50, [0.55, 0.53, 0.49, 0.43]),
            report(0.54, [0.60, 0.58, 0.55, 0.41]),
        )
        self.assertFalse(decision["seed42_promotion_candidate"])
        self.assertEqual(decision["action"], "DO_NOT_PROMOTE_SUPPORTED_SHELL_DEGRADATION")

    def test_tiny_gain_retains_response_without_accuracy_claim(self):
        decision = promotion_decision(
            report(0.50, [0.55, 0.53, 0.49, 0.43]),
            report(0.505, [0.555, 0.535, 0.495, 0.435]),
        )
        self.assertFalse(decision["second_seed_required_before_final_freeze"])


if __name__ == "__main__":
    unittest.main()
