import unittest

import numpy as np

from workflows.abacus_tweb.p8_deterministic_common import evaluate_complete_phase
from workflows.abacus_tweb.p10_train_arm_a import frozen_arguments, source_contract


class P10ArmATrainingTests(unittest.TestCase):
    @staticmethod
    def assignment():
        dtype = [
            ("parent_node_id", "i8"),
            ("supervised_eligible", "?"),
            ("shell", "i1"),
            ("superblock_id", "i4"),
            ("distance_to_conservative_fold_boundary_mpc", "f4"),
        ]
        rows = np.zeros(12, dtype=dtype)
        rows["parent_node_id"] = np.arange(12)
        rows["supervised_eligible"] = True
        rows["shell"] = np.repeat(np.arange(4), 3)
        rows["superblock_id"] = np.tile(np.arange(3), 4)
        rows["distance_to_conservative_fold_boundary_mpc"] = np.linspace(5, 50, 12)
        return rows

    def test_complete_phase_evaluation_requires_exact_authoritative_rows(self):
        assignment = self.assignment()
        base = np.linspace(-0.4, 0.6, 12)
        truth = np.column_stack((base, base + 0.2, base + 0.5))
        order = np.asarray([9, 2, 7, 0, 11, 5, 1, 10, 3, 8, 6, 4])
        report = evaluate_complete_phase(
            parent_node_id=order,
            predicted_eigenvalues=truth[order],
            truth_by_parent=truth,
            assignment=assignment,
            phase="ph006",
        )
        self.assertTrue(report["complete_phase_coverage"])
        self.assertEqual(report["n_authoritative"], 12)
        self.assertAlmostEqual(report["primary_macro_r2_lambda1"], 1.0)
        with self.assertRaises(RuntimeError):
            evaluate_complete_phase(
                parent_node_id=order[:-1],
                predicted_eigenvalues=truth[order[:-1]],
                truth_by_parent=truth,
                assignment=assignment,
                phase="ph006",
            )

    def test_runtime_only_arguments_do_not_change_resume_contract(self):
        class Arguments:
            pass

        first = Arguments()
        first.model = "unet"
        first.epochs = 20
        first.auto_resume = False
        first.checkpoint_every = 250
        first.device = "cuda"
        first.loss_log_every = 25
        first.max_runtime_seconds = 6600.0
        first.validation_reserve_seconds = 1200.0
        first.scheduler_total_updates = 400_000
        first.gradient_clip = 5.0
        second = Arguments()
        second.__dict__.update(first.__dict__)
        second.auto_resume = True
        second.max_runtime_seconds = 3000.0
        second.checkpoint_every = 100
        self.assertEqual(frozen_arguments(first), frozen_arguments(second))
        self.assertEqual(frozen_arguments(first)["scheduler_total_updates"], 400_000)
        self.assertEqual(frozen_arguments(first)["gradient_clip"], 5.0)

    def test_source_contract_canonicalizes_home_and_u2_aliases(self):
        contract = source_contract()
        self.assertIn("workflows/abacus_tweb/p10_train_arm_a.py", contract)
        self.assertTrue(all(len(digest) == 64 for digest in contract.values()))


if __name__ == "__main__":
    unittest.main()
