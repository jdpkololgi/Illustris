import unittest

import numpy as np

from workflows.abacus_tweb.p10_evaluate_multitracer import (
    aggregate_core_shell,
    r2_from_sums,
    scores,
)


class P10EvaluateMultitracerTests(unittest.TestCase):
    def test_sufficient_statistics_match_direct_r2(self):
        core = np.asarray([0, 0, 1, 1, 2, 2])
        shell = np.asarray([0, 0, 0, 0, 1, 1])
        weight = np.asarray([1, 2, 1, 1, 1, 3], dtype=float)
        truth = np.asarray([0, 1, 2, 3, 4, 7], dtype=float)
        prediction = truth + np.asarray([0, 0.2, -0.3, 0, 0.5, -0.5])
        common, method = aggregate_core_shell(
            core=core, shell=shell, weight=weight, truth=truth,
            predictions={"model": prediction},
        )
        result = scores(common, method["model"])["shell_r2_lambda1"]
        for shell_id in (0, 1):
            chosen = shell == shell_id
            w, y, p = weight[chosen], truth[chosen], prediction[chosen]
            mean = np.sum(w * y) / np.sum(w)
            direct = 1 - np.sum(w * (p - y) ** 2) / np.sum(w * (y - mean) ** 2)
            self.assertAlmostEqual(result[shell_id], direct)

    def test_r2_rejects_empty_weight(self):
        self.assertTrue(np.isnan(r2_from_sums(0, 0, 0, 0)))


if __name__ == "__main__":
    unittest.main()
