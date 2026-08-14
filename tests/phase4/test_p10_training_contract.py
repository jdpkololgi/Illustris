import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from workflows.abacus_tweb.p10_training_contract import (
    BLIND_PHASE,
    TRAINING_PHASES,
    VALIDATION_PHASE,
    P10PhaseBalancedLoader,
    epoch_hash,
    phase_balanced_epoch,
    phase_equal_patch_objective,
    resume_state,
    validate_resume_state,
)


class P10TrainingContractTests(unittest.TestCase):
    def test_epoch_visits_every_phase_core_once_and_is_deterministic(self):
        ids = {
            "ph000": np.asarray([1, 2, 3]),
            "ph002": np.asarray([10, 11]),
            "ph003": np.asarray([20, 21, 22, 23]),
        }
        weights = {
            phase: np.linspace(1.0, 2.0, len(values))
            for phase, values in ids.items()
        }
        first = phase_balanced_epoch(ids, seed=42, epoch=3, core_weight_by_phase=weights)
        second = phase_balanced_epoch(ids, seed=42, epoch=3, core_weight_by_phase=weights)
        self.assertEqual(epoch_hash(first), epoch_hash(second))
        for phase, expected in ids.items():
            found = [row.core_id for row in first if row.phase == phase]
            self.assertEqual(sorted(found), sorted(expected.tolist()))
            self.assertEqual(len(found), len(set(found)))

    def test_round_robin_prefix_is_phase_balanced(self):
        ids = {
            phase: np.arange(index * 100, index * 100 + 8)
            for index, phase in enumerate(("ph000", "ph002", "ph003"))
        }
        refs = phase_balanced_epoch(ids, seed=7, epoch=0)
        for stop in range(3, len(refs) + 1, 3):
            counts = [
                sum(row.phase == phase for row in refs[:stop])
                for phase in ids
            ]
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_resume_reconstructs_identical_tail(self):
        ids = {
            "ph000": np.arange(5),
            "ph002": np.arange(7),
        }
        refs = phase_balanced_epoch(ids, seed=11, epoch=2)
        state = resume_state(seed=11, epoch=2, cursor=6, refs=refs)
        rebuilt = phase_balanced_epoch(ids, seed=11, epoch=2)
        validate_resume_state(state, rebuilt)
        self.assertEqual(refs[6:], rebuilt[6:])
        broken = dict(state)
        broken["epoch_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_resume_state(broken, rebuilt)

    def test_patch_objective_recovers_equal_phase_mean(self):
        ids = {
            "ph000": np.asarray([0, 1]),
            "ph002": np.asarray([0, 1, 2]),
        }
        refs = phase_balanced_epoch(ids, seed=5, epoch=0)
        numerators = {
            "ph000": np.asarray([1.0, 3.0]),
            "ph002": np.asarray([2.0, 4.0, 8.0]),
        }
        denominators = {"ph000": 2.0, "ph002": 7.0}
        reconstructed = np.mean([
            phase_equal_patch_objective(
                numerators[row.phase][row.core_id],
                phase_weight_denominator=denominators[row.phase],
                phase_objective_scale=row.phase_objective_scale,
            )
            for row in refs
        ])
        expected = 0.5 * (
            numerators["ph000"].sum() / denominators["ph000"]
            + numerators["ph002"].sum() / denominators["ph002"]
        )
        self.assertAlmostEqual(reconstructed, expected)

    def test_loader_roles_and_blind_truth_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = {
                "pass": True,
                "roles": {
                    "training": list(TRAINING_PHASES),
                    "validation_and_selection": VALIDATION_PHASE,
                    "sealed_blind_test": BLIND_PHASE,
                },
            }
            (root / "TRAINING_LOADER_READY.json").write_text(json.dumps(marker))
            for phase in TRAINING_PHASES + (VALIDATION_PHASE, BLIND_PHASE):
                phase_root = root / "phases" / phase
                phase_root.mkdir(parents=True)
                (phase_root / "phase_contract.json").write_text(json.dumps({
                    "phase": phase,
                    "truth_present": phase != BLIND_PHASE,
                }))
            loader = P10PhaseBalancedLoader(root, include_blind=True)
            self.assertEqual(loader.training_phases, TRAINING_PHASES)
            with self.assertRaises(PermissionError):
                loader.targets_by_parent(BLIND_PHASE)

    def test_registry_roles_match_frozen_code(self):
        registry = json.loads(Path("configs/p10_phase_registry_v1.json").read_text())
        roles = registry["model_phase_contract"]
        self.assertEqual(tuple(roles["training"]), TRAINING_PHASES)
        self.assertEqual(roles["validation_and_selection"], [VALIDATION_PHASE])
        self.assertEqual(roles["sealed_blind_test"], [BLIND_PHASE])
        self.assertFalse(roles["phase_is_model_input"])
        self.assertTrue(roles["fresh_initialization_required"])


if __name__ == "__main__":
    unittest.main()

