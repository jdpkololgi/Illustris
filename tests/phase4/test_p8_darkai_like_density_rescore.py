import unittest

import numpy as np
import torch

from workflows.abacus_tweb.p8_darkai_like_density_rescore import (
    class_recall_from_components,
    spectral_table,
)
from workflows.abacus_tweb.p8_evaluate_stitched_density import (
    spectral_sums,
    spectra_report,
)


def diagonal_components(eigenvalues):
    values = np.asarray(eigenvalues, dtype=np.float32)
    result = np.zeros((len(values), 6), dtype=np.float32)
    result[:, 0] = values[:, 0]
    result[:, 3] = values[:, 1]
    result[:, 5] = values[:, 2]
    return result


class DarkAILikeDensityRescoreTests(unittest.TestCase):
    def test_spectral_amplitude_and_phase_definitions(self):
        generator = torch.Generator().manual_seed(19)
        truth = torch.randn((12, 10, 8), generator=generator)
        prediction = 0.7 * truth
        edges = np.geomspace(0.01, 1.0, 12)
        report = spectra_report(
            spectral_sums(prediction, truth, cell_mpc=5.0, edges_h_mpc=edges),
            edges,
        )
        used = np.asarray(report["mode_count"]) > 0
        np.testing.assert_allclose(
            np.asarray(report["cross_transfer"])[used], 0.7, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(report["cross_correlation_r"])[used], 1.0, rtol=1e-5, atol=1e-6
        )
        rows = spectral_table(report)
        self.assertIn("p_cross_over_p_true", rows[0])
        self.assertIn("r_k", rows[0])

    def test_sign_class_recall_is_true_row_normalized(self):
        truth = diagonal_components([
            [-3, -2, -1], [-2, -1, 1], [-1, 1, 2], [1, 2, 3],
        ])
        prediction = diagonal_components([
            [-3, -2, -1], [-2, -1, 1], [-1, -0.5, 2], [1, 2, 3],
        ])
        report = class_recall_from_components(
            prediction, truth, threshold=0.0, chunk=2
        )
        self.assertEqual(report["class_order"], ["void", "sheet", "filament", "knot"])
        self.assertEqual(report["confusion_true_rows_predicted_columns"][2][1], 1)
        self.assertEqual(report["recall"]["void"], 1.0)
        self.assertEqual(report["recall"]["sheet"], 1.0)
        self.assertEqual(report["recall"]["filament"], 0.0)
        self.assertEqual(report["recall"]["knot"], 1.0)
        self.assertAlmostEqual(report["balanced_accuracy"], 0.75)


if __name__ == "__main__":
    unittest.main()
