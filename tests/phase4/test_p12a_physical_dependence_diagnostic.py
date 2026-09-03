from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from workflows.sbi.p12a_physical_dependence_diagnostic import (
    decision_summary,
    scaled_to_physical_eigenvalues,
    spatial_groups,
    validate_source_evidence,
)


class P12APhysicalDependenceDiagnosticTest(unittest.TestCase):
    def test_scaled_to_physical_outputs_ordered_eigenvalues(self) -> None:
        scaled = np.asarray([[[0.0, 0.0, 0.0], [1.0, -2.0, 2.0]]])
        eigen = scaled_to_physical_eigenvalues(
            scaled,
            np.asarray([-0.2, -1.0, -0.5]),
            np.asarray([0.3, 0.5, 0.25]),
        )
        self.assertEqual(eigen.shape, (1, 2, 3))
        self.assertTrue(np.all(np.diff(eigen, axis=-1) > 0))
        self.assertAlmostEqual(eigen[0, 0, 0], -0.2)

    def test_spatial_group_keeps_caps_disjoint(self) -> None:
        group = spatial_groups(np.asarray([0, 1]), np.asarray([7, 7]))
        self.assertNotEqual(int(group[0]), int(group[1]))

    def test_source_guard_rejects_open_blind_phase(self) -> None:
        payload = {
            "schema_version": "p12a-tarp-curve-v1",
            "selection_phase": "ph006",
            "sealed_phase": "ph001",
            "sealed_phase_opened": True,
            "rows": 50_000,
            "posterior_draws_per_row": 512,
        }
        with self.assertRaises(PermissionError):
            validate_source_evidence(payload)

    def test_source_guard_rejects_hash_change(self) -> None:
        payload = {
            "schema_version": "p12a-tarp-curve-v1",
            "selection_phase": "ph006",
            "sealed_phase": "ph001",
            "sealed_phase_opened": False,
            "rows": 50_000,
            "posterior_draws_per_row": 512,
            "provenance": {
                name: f"/tmp/{name}"
                for name in (
                    "audit_report",
                    "checkpoint",
                    "dataset_marker",
                    "evaluation_index",
                    "posterior_samples",
                    "validation_sample",
                )
            },
        }
        with patch("pathlib.Path.is_file", return_value=True), patch(
            "workflows.sbi.p12a_physical_dependence_diagnostic.sha256",
            return_value="actual",
        ):
            with self.assertRaises(RuntimeError):
                validate_source_evidence(payload)

    def test_decision_distinguishes_global_from_shell_scope(self) -> None:
        def curve(maximum: float) -> dict:
            return {"maximum_deviation": maximum}

        report = {
            "nested_draw_reports": {
                "512": {
                    "tarp": {
                        "ordered_eigenvalues": curve(0.02),
                        "eigengaps": curve(0.03),
                        "reference_seed_maxima": {
                            "ordered_p90": 0.025,
                            "eigengap_p90": 0.035,
                        },
                    }
                }
            },
            "shell_tarp_512_draws": {
                str(shell): {
                    "ordered_eigenvalues": curve(0.04),
                    "eigengaps": curve(0.08 if shell == 3 else 0.04),
                }
                for shell in range(4)
            },
        }
        decision = decision_summary(report)
        self.assertTrue(decision["global_physical_joint_pass"])
        self.assertEqual(decision["maxima"]["shell_eigengap_maximum"], 0.08)
        self.assertIn("within-galaxy", decision["scope"])


if __name__ == "__main__":
    unittest.main()
