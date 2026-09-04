import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from workflows.sbi.p12f3_freeze_diffusion_license import loss_plateau, science_gates


class P12F3DiffusionLicenseTests(unittest.TestCase):
    def test_plateau_accepts_no_final_window_improvement(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "loss.jsonl"
            rows = []
            for update in range(25, 10_001, 25):
                loss = 1.0 if update <= 7_500 else 1.01
                rows.append(json.dumps({"update": update, "loss": loss}))
            path.write_text("\n".join(rows) + "\n")
            result = loss_plateau(path, window=2_500, maximum_improvement=0.01)
        self.assertTrue(result["pass"])
        self.assertLess(result["fractional_improvement"], 0.0)

    def test_science_gate_fails_conditional_coverage(self):
        core = [
            {"core_id": index, "energy": 0.9, "coarse_energy": 1.0, "variogram_p0p5": 1.0}
            for index in range(256)
        ]
        reference_core = [dict(row, energy=1.0) for row in core]
        report = {
            "tarp": {
                "ordered_eigenvalues": {"full_max_abs_ecp_minus_alpha": 0.02},
                "eigengaps": {"full_max_abs_ecp_minus_alpha": 0.03},
            },
            "global_coverage_error": {"0.68": 0.02, "0.90": 0.03},
            "maximum_conditional_coverage_error": 0.12,
            "proper_scores": {
                "energy": 0.9, "coarse_energy": 0.9,
                "marginal_crps": 0.9, "variogram_p0p5": 0.9,
            },
            "per_core_proper_scores": core,
        }
        reference = {
            "proper_scores": {
                "energy": 1.0, "coarse_energy": 1.0,
                "marginal_crps": 1.0, "variogram_p0p5": 1.0,
            },
            "per_core_proper_scores": reference_core,
        }
        shear = {"joint_tarp_blocked": {"full_max_abs_ecp_minus_alpha": 0.02}}
        visual = {"posterior_to_truth_power": [1.0, 1.0]}
        contract = {
            "science_gates": {
                "low_band_power_ratio_absolute_tolerance": 0.10,
                "ordered_eigenvalue_tarp_maximum": 0.05,
                "eigengap_tarp_maximum": 0.05,
                "five_shear_tarp_maximum": 0.05,
                "global_coverage_error_maximum": 0.05,
                "conditional_coverage_error_maximum": 0.10,
                "proper_score_worsening_maximum": 0.01,
                "primary_energy_paired_bootstrap_repeats": 100,
            }
        }
        gates = science_gates(report, shear, visual, reference, contract, seed=4)
        self.assertFalse(gates["conditional_coverage"]["pass"])
        self.assertFalse(gates["all_pass"])


if __name__ == "__main__":
    unittest.main()
