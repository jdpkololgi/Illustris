import unittest

import numpy as np

from workflows.sbi.p12f3_observable_proxy_autopsy import (
    metrics_for_labels,
    self_consistency,
    tidal_features,
)


class ObservableProxyAutopsyTests(unittest.TestCase):
    def test_exchangeable_split_draw_reference_is_near_nominal_globally(self):
        rng = np.random.default_rng(3)
        draws = rng.normal(size=(64, 4000))
        truth = rng.normal(size=4000)
        labels = {"response": np.arange(4000) % 4}
        result = self_consistency(draws, truth, labels, repetitions=8, seed=8)
        for level in (68, 90):
            values = np.asarray(result["response"][f"pseudo{level}"]["mean"])
            # Finite 32-draw intervals have discrete empirical coverage and the
            # fixed eight-repeat Monte Carlo reference fluctuates at the few
            # percent level.  This is a functionality check, not a science
            # calibration gate.
            self.assertLess(np.max(np.abs(values - level / 100)), 0.06)

    def test_metrics_separate_location_and_scale(self):
        rng = np.random.default_rng(4)
        draws = rng.normal(size=(64, 800))
        truth = rng.normal(loc=0.5, size=800)
        labels = np.arange(800) % 4
        result = metrics_for_labels(draws, truth, labels)
        self.assertGreater(result["0"]["mean_truth_minus_mean_over_std"], 0.2)
        self.assertIn("coverage68", result["3"])

    def test_tidal_features_zero_field(self):
        field = np.zeros((8, 8, 8), dtype=np.float32)
        valid = np.arange(16)
        shear, web = tidal_features(field, (slice(None),) * 3, valid)
        np.testing.assert_array_equal(shear, 0)
        np.testing.assert_array_equal(web, 0)


if __name__ == "__main__":
    unittest.main()
