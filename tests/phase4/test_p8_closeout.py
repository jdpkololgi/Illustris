import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from workflows.abacus_tweb.p8_closeout import REQUIRED_EVIDENCE, build_closeout


class P8CloseoutTest(unittest.TestCase):
    def test_closeout_requires_and_hashes_all_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            for relative in REQUIRED_EVIDENCE.values():
                path = repo / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = {}
                if relative.endswith("density_d0_darkai_like_rescore.json"):
                    payload = {
                        "grid_cell_classes": {
                            "predicted_fft": {"selected_cells": 11},
                            "darkai_sign_threshold": {
                                "balanced_accuracy": 0.7,
                                "exact_cell_accuracy": 0.8,
                                "recall": {"void": 0.6},
                            },
                        },
                        "spectra": {"mode_weighted_summary": {"band": {"r_k": 0.9}}},
                    }
                elif relative.endswith("density_first_rotation0_closeout.json"):
                    payload = {
                        "decision": {"primary_point_estimator": "NO_PROMOTION"},
                        "tidal": {
                            "z_observed_deployable": {
                                "raw_physical": {
                                    "macro_r2_lambda1": 0.47,
                                    "first_three_shell_macro_r2_lambda1": 0.53,
                                    "per_shell_r2_lambda1": [0.5, 0.5, 0.5, 0.3],
                                }
                            }
                        },
                    }
                elif relative.endswith("multitracer_mt4_decision.json"):
                    payload = {
                        "decision": {
                            "multitracer_information": "PASS",
                            "current_encoder_adoption": "NO_GO",
                        }
                    }
                elif relative.endswith("ucic_v2_closeout.json"):
                    payload = {"decision": "NO_GO"}
                path.write_text(json.dumps(payload))

            with mock.patch(
                "workflows.abacus_tweb.p8_closeout.git_revision", return_value="abc123"
            ):
                result = build_closeout(repo)

            self.assertEqual(
                result["status"], "P8_COMPLETE_PH000_DETERMINISTIC_DEVELOPMENT_FROZEN"
            )
            self.assertEqual(result["darkai_like_diagnostic"]["selected_equal_volume_cells"], 11)
            self.assertEqual(set(result["evidence"]), set(REQUIRED_EVIDENCE))
            self.assertEqual(
                result["closed_without_run"]["density_d1_auxiliary"]["status"],
                "NOT_RUN_SUPERSEDED_BY_PH000_FREEZE",
            )


if __name__ == "__main__":
    unittest.main()
