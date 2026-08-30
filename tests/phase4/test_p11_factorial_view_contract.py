import json
from pathlib import Path
import unittest


REPO = Path(__file__).resolve().parents[2]


class P11FactorialViewContractTests(unittest.TestCase):
    def setUp(self):
        self.contract = json.loads(
            (REPO / "configs/p11_factorial_views_v1.json").read_text()
        )

    def test_phase_roles_are_disjoint_and_blind_is_sealed(self):
        split = self.contract["phase_split"]
        train = set(split["training"])
        validation = set(split["validation_and_selection"])
        blind = set(split["sealed_blind_test"])
        self.assertFalse(train & validation)
        self.assertFalse(train & blind)
        self.assertFalse(validation & blind)
        self.assertEqual(blind, {"ph001"})
        self.assertNotIn("ph000", train)
        exclusion = self.contract["phase_exclusions"]["ph000"]
        self.assertEqual(exclusion["scope"], "P11 factorial-view branch only")
        self.assertIn("not TARGETID-nested", exclusion["reason"])

    def test_factorial_axes_and_heldout_recipe_are_explicit(self):
        self.assertEqual(
            set(self.contract["observation_stage_axis"]),
            {"V_dense", "V_assign", "V_final"},
        )
        self.assertEqual(
            set(self.contract["tracer_axis"]),
            {"bright_only", "bright_plus_faint_context"},
        )
        heldout = [
            name
            for name, row in self.contract["stochastic_response_axis"].items()
            if row["role"] == "held_out_degradation_recipe"
        ]
        self.assertEqual(heldout, ["tileloc_correlated_thinning"])

    def test_jepa_is_not_posterior(self):
        self.assertIn(
            "never replaces posterior calibration",
            self.contract["production_rule"],
        )


if __name__ == "__main__":
    unittest.main()
