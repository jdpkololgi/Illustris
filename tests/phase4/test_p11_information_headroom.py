import unittest

import numpy as np

from workflows.sbi.p11_information_headroom import (
    aggregate_by_block_shell,
    metric_set_from_aggregate,
    metrics_from_sums,
    weighted_components,
)


class TestP11InformationHeadroom(unittest.TestCase):
    def test_calibrated_posterior_satisfies_bayes_identity(self):
        truth = np.array([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]])
        posterior_mean = np.ones_like(truth) * np.array([1.0, 2.0, 3.0])
        posterior_variance = np.ones_like(truth)
        base = posterior_mean.copy()
        weight = np.ones(2)
        component = weighted_components(
            truth, posterior_mean, posterior_variance, base, weight
        )
        result = metrics_from_sums(
            component["weight"].sum(),
            component["weighted_truth"].sum(axis=0),
            component["weighted_truth2"].sum(axis=0),
            component["weighted_posterior_variance"].sum(axis=0),
            component["weighted_posterior_squared_error"].sum(axis=0),
            component["weighted_base_squared_error"].sum(axis=0),
        )
        np.testing.assert_allclose(result["bayes_identity_gap_r2"], 0.0)
        np.testing.assert_allclose(result["posterior_variance_r2_estimate"], 0.0)

    def test_underdispersed_posterior_exposes_positive_headroom_gap(self):
        truth = np.array([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]])
        posterior_mean = np.ones_like(truth) * np.array([1.0, 2.0, 3.0])
        posterior_variance = np.full_like(truth, 0.25)
        component = weighted_components(
            truth, posterior_mean, posterior_variance, posterior_mean, np.ones(2)
        )
        result = metrics_from_sums(
            component["weight"].sum(),
            component["weighted_truth"].sum(axis=0),
            component["weighted_truth2"].sum(axis=0),
            component["weighted_posterior_variance"].sum(axis=0),
            component["weighted_posterior_squared_error"].sum(axis=0),
            component["weighted_base_squared_error"].sum(axis=0),
        )
        np.testing.assert_allclose(result["bayes_identity_gap_r2"], 0.75)
        np.testing.assert_allclose(result["nonnegative_same_summary_headroom_r2"], 0.75)

    def test_block_shell_aggregation_preserves_point_metrics(self):
        rng = np.random.default_rng(7)
        rows = 80
        truth = rng.normal(size=(rows, 3))
        posterior_mean = truth + rng.normal(scale=0.5, size=(rows, 3))
        posterior_variance = np.full((rows, 3), 0.25)
        base = posterior_mean + rng.normal(scale=0.1, size=(rows, 3))
        weight = rng.uniform(0.5, 1.5, size=rows)
        group = np.repeat(np.arange(20), 4)
        shell = np.tile(np.arange(4), 20)
        components = weighted_components(
            truth, posterior_mean, posterior_variance, base, weight
        )
        aggregate, unique = aggregate_by_block_shell(components, group, shell)
        global_result, shell_result, macro = metric_set_from_aggregate(
            aggregate, np.ones(len(unique), dtype=int)
        )
        direct = metrics_from_sums(
            components["weight"].sum(),
            components["weighted_truth"].sum(axis=0),
            components["weighted_truth2"].sum(axis=0),
            components["weighted_posterior_variance"].sum(axis=0),
            components["weighted_posterior_squared_error"].sum(axis=0),
            components["weighted_base_squared_error"].sum(axis=0),
        )
        for key in direct:
            np.testing.assert_allclose(global_result[key], direct[key])
            np.testing.assert_allclose(
                macro[key], np.mean([value[key] for value in shell_result], axis=0)
            )


if __name__ == "__main__":
    unittest.main()
