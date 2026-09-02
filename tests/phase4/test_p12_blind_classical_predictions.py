from __future__ import annotations

import unittest

import numpy as np

from workflows.sbi.p12_blind_classical_predictions import (
    authoritative_rows,
    validate_affine_report,
    validate_response_manifest,
)


class _Archive:
    files = ("parent_node_id", "targetid", "core_id", "cap", "shell", "supervised_eligible")

    def __init__(self):
        self.values = {
            "parent_node_id": np.asarray([9, 3, 8, 2]),
            "targetid": np.asarray([90, 30, 80, 20]),
            "core_id": np.asarray([1, 0, 1, 0]),
            "cap": np.asarray([1, 0, 1, 0]),
            "shell": np.asarray([0, 0, 1, 0]),
            "supervised_eligible": np.asarray([True, True, False, True]),
        }

    def __getitem__(self, key):
        return self.values[key]


class P12BlindClassicalPredictionsTest(unittest.TestCase):
    def report(self):
        return {
            "schema_version": "p10-cic-final-v1",
            "pass": True,
            "training_phases": ["ph000", "ph002", "ph003", "ph004", "ph005"],
            "validation_phase": "ph006",
            "sealed_blind_phase": "ph001",
            "blind_phase_opened": False,
            "affine": {
                "coefficients": [
                    {"slope": 1.0, "intercept": 0.0},
                    {"slope": 1.0, "intercept": 0.0},
                    {"slope": 1.0, "intercept": 0.0},
                ]
            },
        }

    def test_affine_contract_is_train_only_and_sealed(self):
        affine = validate_affine_report(self.report(), "cic")
        self.assertEqual(len(affine["coefficients"]), 3)
        opened = self.report()
        opened["blind_phase_opened"] = True
        with self.assertRaises(PermissionError):
            validate_affine_report(opened, "cic")

    def test_authoritative_rows_are_core_then_parent_ordered(self):
        parent, core, cap = authoritative_rows(_Archive())
        np.testing.assert_array_equal(parent, [2, 3, 9])
        np.testing.assert_array_equal(core, [0, 0, 1])
        np.testing.assert_array_equal(cap, [0, 0, 1])

    def test_response_must_be_authorized_and_sealed(self):
        valid = {
            "schema_version": "p3br-response-overlay-manifest-v1",
            "phase": "ph001",
            "pass": True,
            "ph001_opened": False,
            "blind_authority": {"sha256": "abc"},
            "components": {"NGC": {}, "SGC": {}},
        }
        validate_response_manifest(valid)
        valid["ph001_opened"] = True
        with self.assertRaises(PermissionError):
            validate_response_manifest(valid)


if __name__ == "__main__":
    unittest.main()
