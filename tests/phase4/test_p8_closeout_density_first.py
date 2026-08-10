import unittest

from workflows.abacus_tweb.p8_closeout_density_first import build_decision


def score(macro, shells, balanced=0.6, void=0.6, knot=0.4):
    names = ("0p15_0p25", "0p25_0p35", "0p35_0p45", "0p45_0p55")
    classification = {
        "balanced_accuracy": balanced,
        "macro_f1": balanced,
        "void_recall": void,
        "knot_recall": knot,
    }
    return {
        "primary_macro_r2_lambda1": macro,
        "diagnostic_first_three_shell_macro_r2_lambda1": sum(shells[:3]) / 3,
        "pooled": {"lambda1": {"r2": macro}, "classification": classification},
        "per_shell": {
            name: {"lambda1": {"r2": value}} for name, value in zip(names, shells)
        },
        "spatial_block_interval": {"p16": macro - 0.01, "p50": macro, "p84": macro + 0.01},
    }


class DensityFirstCloseoutTests(unittest.TestCase):
    def test_unique_tensor_and_class_benefit_can_authorize_rotation2(self):
        raw = score(0.47, [0.53, 0.58, 0.49, 0.29], 0.66, 0.81, 0.50)
        affine = score(0.51, [0.57, 0.61, 0.52, 0.34])
        coordinate = {
            "raw_physical": raw,
            "train_fold_affine_diagnostic": affine,
            "affine": {"fit_split": "training cores", "coefficients": []},
        }
        evaluation = {
            "field_metrics": {
                "overall": {"r2": 0.62, "prediction_std": 0.37, "truth_std": 0.46,
                            "tails": {"3.0": {"count_ratio_prediction_to_truth": 0.2},
                                      "-0.5": {"count_ratio_prediction_to_truth": 0.4}}},
                "macro_shell_r2_delta_r7": 0.68,
                "by_shell": {name: {"r2": 0.6} for name in (
                    "0p15_0p25", "0p25_0p35", "0p35_0p45", "0p45_0p55")},
            },
            "spectra": {"pooled_caps": {
                "k_centres_h_mpc": [0.03, 0.1], "mode_count": [100, 100],
                "cross_correlation_r": [0.9, 0.85], "cross_transfer": [0.8, 0.7],
            }},
            "tidal": {
                "coordinates": {"z_cosmo_oracle": coordinate,
                                "z_observed_deployable": coordinate},
                "predicted_vs_windowed_true_tensor_components_z_cosmo": {
                    name: {"r2": 0.85} for name in ("t00", "t01", "t02", "t11", "t12", "t22")
                },
                "orientation_z_cosmo": {"bins": {"0": {}, "3": {}}},
            },
        }
        reference = score(0.507, [0.62, 0.57, 0.49, 0.345], 0.63, 0.68, 0.42)
        decision = build_decision(
            {"epochs_completed": 20, "best_epoch": 16,
             "best_macro_shell_r2_delta_r7": 0.697},
            {"status": "PASS", "checkpoint_epoch": 16,
             "support_coverage": {}, "trained_patch_parity": {}},
            evaluation,
            {"anchors": {"n": 24}, "radii": {}},
            {"status": "PASS"},
            reference,
        )
        self.assertFalse(decision["registered_gates"]["within_0p03_macro"])
        self.assertTrue(decision["registered_gates"]["unique_tensor_and_raw_class_benefit"])
        self.assertEqual(
            decision["decision"]["rotation2_continuation"],
            "GO_UNIQUE_TENSOR_AND_RAW_CLASS_BENEFIT",
        )


if __name__ == "__main__":
    unittest.main()
