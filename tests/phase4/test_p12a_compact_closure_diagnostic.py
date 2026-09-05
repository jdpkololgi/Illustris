import unittest

import numpy as np

from workflows.sbi.diagnose_p12a_compact_closure import closure_diagnostic


class CompactClosureDiagnosticTests(unittest.TestCase):
    def test_rounding_disagreement_reproduced_without_truth(self):
        eigen = np.array([[-0.1, 0.2, 0.5]], dtype=np.float32)
        cweb = (eigen.astype(np.float64) > 0.2).sum(axis=1)
        report = closure_diagnostic(eigen, cweb)
        self.assertEqual(report["float32_comparison_class_mismatches"], 1)
        self.assertEqual(report["float64_comparison_class_mismatches"], 0)
        self.assertTrue(report["all_float32_mismatches_explained_by_threshold_rounding"])
        self.assertTrue(report["source_values_preserved_by_float32"])

    def test_ordinary_values_have_no_disagreement(self):
        eigen = np.array([[-0.1, 0.1, 0.5], [-0.4, -0.3, -0.2]], dtype=np.float32)
        report = closure_diagnostic(eigen, np.array([1, 0]))
        self.assertEqual(report["float32_comparison_class_mismatches"], 0)
        self.assertEqual(report["float64_comparison_class_mismatches"], 0)
        self.assertEqual(report["rows_at_rounded_threshold"], 0)

    def test_genuine_class_disagreement_is_not_excused(self):
        report = closure_diagnostic(np.array([[-0.1, 0.1, 0.5]], dtype=np.float32), np.array([0]))
        self.assertFalse(report["all_float32_mismatches_explained_by_threshold_rounding"])
        self.assertEqual(report["float64_comparison_class_mismatches"], 1)

    def test_invalid_eigenvalues_remain_invalid(self):
        report = closure_diagnostic(np.array([[0.5, 0.1, -0.1]], dtype=np.float32), np.array([1]))
        self.assertFalse(report["ordered"])
        report = closure_diagnostic(np.array([[0.1, 0.2, np.nan]], dtype=np.float32), np.array([1]))
        self.assertFalse(report["finite"])


if __name__ == "__main__":
    unittest.main()
