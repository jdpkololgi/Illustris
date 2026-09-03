import copy
import unittest

from workflows.sbi.p12f3_freeze_lowmode_decision import build_decision


class P12F3LowmodeDecisionTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "expected_methods": [
                "g1_wide_crop_h8", "g1_wide_h24",
                "hybrid_local_h8", "hybrid_wide_h24",
            ],
            "expected_cores": 2,
            "expected_draws": 64,
            "registered_low_bands": 2,
            "bootstrap_replicates": 100,
            "bootstrap_seed": 2,
            "gates": {
                "low_band_error_improvement_vs_local": 0.2,
                "eigengap_tarp_maximum_deviation": 0.05,
                "global_coverage_error": 0.05,
                "conditional_coverage_error": 0.10,
                "ordered_eigen_tarp_absolute_worsening": 0.01,
                "proper_score_fractional_worsening": 0.01,
            },
        }
        rows = [
            {"core_id": 1, "energy": 1.0, "coarse_energy": 1.0, "variogram_p0p5": 1.0},
            {"core_id": 2, "energy": 1.1, "coarse_energy": 1.1, "variogram_p0p5": 1.1},
        ]
        base = {
            "schema_version": "p12f-common-evaluation-report-v1",
            "phase": "ph006", "cores": 2, "draws": 64,
            "conditioning_contract_sha256": "condition", "target_scaler_sha256": "scale",
            "ph001_opened": False, "truth_files_read": ["ph006 density/T-web"],
            "physics_closure": {"all_finite": True, "all_ordered": True},
            "global_coverage_error": {"0.68": 0.02, "0.90": 0.03},
            "maximum_conditional_coverage_error": 0.08,
            "proper_scores": {"energy": 1.05, "primary_joint": 1.05, "coarse_energy": 1.05,
                              "variogram_p0p5": 1.05, "marginal_crps": 1.05},
            "per_core_proper_scores": rows,
        }
        self.reports = {}
        for method in self.config["expected_methods"]:
            report = copy.deepcopy(base)
            report["method"] = method
            self.reports[method] = report
        # Make wide decisively better by core and in aggregate.
        wide = self.reports["hybrid_wide_h24"]
        wide["proper_scores"] = {key: value * 0.95 for key, value in wide["proper_scores"].items()}
        for row in wide["per_core_proper_scores"]:
            for key in ("energy", "coarse_energy", "variogram_p0p5"):
                row[key] *= 0.95
        self.visual = {"phase": "ph006", "ph001_opened": False, "methods": {}}
        for method in self.config["expected_methods"]:
            self.visual["methods"][method] = {
                "posterior_to_truth_power": [0.8, 0.8],
                "eigen_tarp": {"maximum_deviation": 0.03},
                "gap_tarp": {"maximum_deviation": 0.04},
            }
        self.visual["methods"]["hybrid_local_h8"]["posterior_to_truth_power"] = [0.7, 0.7]
        self.visual["methods"]["hybrid_wide_h24"]["posterior_to_truth_power"] = [0.9, 0.9]

    def test_passing_registered_decision(self):
        result = build_decision(self.config, self.reports, self.visual)
        self.assertTrue(result["pass"])
        self.assertEqual(result["decision"], "advance_to_f3h")
        self.assertFalse(result["ph001_opened"])

    def test_longest_band_overshoot_blocks_promotion(self):
        self.visual["methods"]["hybrid_wide_h24"]["posterior_to_truth_power"][0] = 1.4
        result = build_decision(self.config, self.reports, self.visual)
        self.assertFalse(result["pass"])
        self.assertFalse(result["gate_groups"]["low_band_power"])

    def test_eigen_tarp_and_score_worsening_block_promotion(self):
        self.visual["methods"]["hybrid_wide_h24"]["eigen_tarp"]["maximum_deviation"] = 0.2
        self.reports["hybrid_wide_h24"]["proper_scores"]["variogram_p0p5"] = 1.2
        result = build_decision(self.config, self.reports, self.visual)
        self.assertFalse(result["pass"])
        self.assertFalse(result["gate_groups"]["tarp_and_coverage"])
        self.assertFalse(result["gate_groups"]["proper_scores"])

    def test_blind_provenance_fails_closed(self):
        self.reports["hybrid_wide_h24"]["ph001_opened"] = True
        with self.assertRaises(PermissionError):
            build_decision(self.config, self.reports, self.visual)


if __name__ == "__main__":
    unittest.main()
