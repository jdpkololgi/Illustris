import unittest

import numpy as np
import torch

from workflows.abacus_tweb.p8_evaluate_stitched_density import (
    DistributionAccumulator,
    spectra_report,
    spectral_sums,
)
from workflows.abacus_tweb.p8_infer_stitched_density import convergence_metrics


class DensityFieldEvaluationTests(unittest.TestCase):
    def test_patch_convergence_metrics_are_scale_normalized(self):
        reference = np.linspace(-2.0, 2.0, 101)
        candidate = reference + 1e-3
        report = convergence_metrics(candidate, reference)
        self.assertAlmostEqual(report["rmse"], 1e-3)
        self.assertAlmostEqual(report["p95_abs_over_std"], 1e-3 / reference.std())
        self.assertAlmostEqual(report["max_abs"], 1e-3)

    def test_distribution_accumulator_reports_perfect_field_and_tails(self):
        truth = np.array([-0.9, -0.6, 0.0, 1.5, 3.5, 7.0])
        accumulator = DistributionAccumulator()
        accumulator.add(truth[:3], truth[:3])
        accumulator.add(truth[3:], truth[3:])
        report = accumulator.report()
        self.assertAlmostEqual(report["r2"], 1.0)
        self.assertEqual(report["tails"]["6.0"]["truth_count"], 1)
        self.assertEqual(report["tails"]["6.0"]["prediction_count"], 1)
        self.assertAlmostEqual(report["tails"]["6.0"]["count_ratio_prediction_to_truth"], 1.0)

    def test_spectral_identity_has_unit_r_transfer_and_power(self):
        generator = torch.Generator().manual_seed(11)
        field = torch.randn((12, 10, 8), generator=generator)
        edges = np.geomspace(0.01, 1.0, 12)
        sums = spectral_sums(field, field, cell_mpc=5.0, edges_h_mpc=edges)
        report = spectra_report(sums, edges)
        used = np.asarray(report["mode_count"]) > 0
        np.testing.assert_allclose(np.asarray(report["cross_correlation_r"])[used], 1.0)
        np.testing.assert_allclose(np.asarray(report["cross_transfer"])[used], 1.0)
        np.testing.assert_allclose(np.asarray(report["power_ratio"])[used], 1.0)


if __name__ == "__main__":
    unittest.main()
