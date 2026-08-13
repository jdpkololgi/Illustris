import unittest
from pathlib import Path

import numpy as np

from workflows.abacus_tweb.p10_build_observed_truth import (
    default_lss,
    default_output,
    observed_success_mask,
)
from workflows.abacus_tweb.p10_target_contract import stored_class_consistency


class P10ObservedTruthTests(unittest.TestCase):
    def test_observed_success_requires_positive_finite_z_and_zwarn_zero(self):
        table = np.array(
            [(0.2, 0), (np.nan, 0), (-1.0, 0), (0.3, 1)],
            dtype=[("Z_not4clus", "f8"), ("ZWARN", "i8")],
        )
        self.assertEqual(observed_success_mask(table).tolist(), [True, False, False, False])

    def test_default_lss_uses_full_bright_not_mock_number_suffix(self):
        registry = {
            "path_templates": {
                "lss": "/data/mock{mock}/LSScats",
                "phase_output": "/scratch/{phase}",
            },
            "phases": {"ph002": {"mock": 2}},
        }
        self.assertEqual(
            default_lss(registry, "ph002"),
            Path("/data/mock2/LSScats/BGS_BRIGHT_full_HPmapcut.dat.fits"),
        )
        self.assertNotIn("BGS_BRIGHT-02", str(default_lss(registry, "ph002")))

    def test_default_output_names_full_observation_contract(self):
        registry = {
            "path_templates": {"phase_output": "/scratch/{phase}"},
            "phases": {"ph002": {"mock": 2}},
        }
        self.assertEqual(
            default_output(registry, "ph002"),
            Path("/scratch/ph002/catalogues/observed/"
                 "ph002_bgs_bright_full_observed_with_tweb.fits"),
        )

    def test_native_class_mismatch_is_allowed_only_at_float32_threshold(self):
        at_threshold = np.float32(0.2)
        eigenvalues = np.asarray(
            [[at_threshold, 0.6, 0.9], [0.1, 0.6, 0.9]], dtype=np.float32
        )
        result = stored_class_consistency(eigenvalues, np.asarray([3, 3]))
        self.assertEqual(result["mismatch"].tolist(), [True, True])
        self.assertEqual(result["boundary_ambiguous"].tolist(), [True, False])
        self.assertEqual(result["nonboundary_mismatch"].tolist(), [False, True])


if __name__ == "__main__":
    unittest.main()
