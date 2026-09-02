from __future__ import annotations

import unittest

import numpy as np
import torch

from workflows.sbi.p12f_challenger_common import (
    ConditionedView,
    FieldSampleContract,
    core_joint_scores,
    energy_score,
    freeze_method_selection,
    paired_core_bootstrap,
    select_truth_free_panel,
    variogram_score,
)
from workflows.sbi.p12f_gaussian_controls import (
    correlated_unit_residuals,
    fit_radial_residual_filter,
    gaussian_nll,
    sample_correlated_gaussian,
)


class P12FChallengerTest(unittest.TestCase):
    def test_truth_free_panel_has_32_per_shell_and_is_reproducible(self):
        core = np.arange(800)
        shell = np.repeat(np.arange(4), 200)
        cap = core % 2
        response = np.sin(core / 20.0)
        boundary = core % 71
        first = select_truth_free_panel(
            core_id=core,
            shell=shell,
            cap=cap,
            response=response,
            boundary_distance=boundary,
            seed=11,
        )
        second = select_truth_free_panel(
            core_id=core,
            shell=shell,
            cap=cap,
            response=response,
            boundary_distance=boundary,
            seed=11,
        )
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 128)
        np.testing.assert_array_equal(np.bincount(shell[first]), [32, 32, 32, 32])

    def test_field_contract_rejects_degenerate_samples(self):
        truth = np.zeros((5, 5, 5))
        support = np.ones_like(truth, dtype=bool)
        with self.assertRaises(ValueError):
            FieldSampleContract(
                method="bad",
                core_id=1,
                samples=np.zeros((8, 5, 5, 5)),
                truth=truth,
                support=support,
            ).validate()

    def test_energy_and_variogram_prefer_truth_centred_ensemble(self):
        rng = np.random.default_rng(1)
        truth = rng.normal(size=20)
        good = truth[None] + rng.normal(scale=0.2, size=(32, 20))
        bad = truth[None] + 3.0 + rng.normal(scale=0.2, size=(32, 20))
        pairs = np.stack((np.arange(10), np.arange(10, 20)), axis=1)
        self.assertLess(energy_score(good, truth), energy_score(bad, truth))
        self.assertLessEqual(
            variogram_score(good, truth, pairs),
            variogram_score(bad * np.linspace(1.0, 3.0, 20), truth, pairs),
        )

    def test_joint_scores_are_fixed_index_and_finite(self):
        rng = np.random.default_rng(3)
        truth = rng.normal(size=(8, 8, 8))
        samples = truth[None] + rng.normal(scale=0.3, size=(16, 8, 8, 8))
        first = core_joint_scores(samples, truth, np.ones_like(truth, bool), seed=6)
        second = core_joint_scores(samples, truth, np.ones_like(truth, bool), seed=6)
        self.assertEqual(first, second)
        self.assertTrue(all(np.isfinite(list(first.values()))))

    def test_gaussian_nll_uses_supported_cells_only(self):
        mean = torch.zeros((1, 1, 2, 2, 2))
        log_std = torch.zeros_like(mean)
        target = torch.ones_like(mean)
        support = torch.zeros_like(mean, dtype=torch.bool)
        support[..., 0, 0, 0] = True
        value = gaussian_nll(mean, log_std, target, support)
        self.assertAlmostEqual(float(value), 0.5)
        with self.assertRaises(ValueError):
            gaussian_nll(mean, log_std, target, torch.zeros_like(support))

    def test_correlated_residual_filter_is_real_and_repeatable(self):
        rng = np.random.default_rng(4)
        residual = rng.normal(size=(12, 8, 8, 8))
        contract = fit_radial_residual_filter(residual, bins=6)
        first = correlated_unit_residuals(contract, draws=10, seed=9)
        second = correlated_unit_residuals(contract, draws=10, seed=9)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.isrealobj(first))
        self.assertEqual(first.shape, (10, 8, 8, 8))
        sample = sample_correlated_gaussian(
            np.zeros((8, 8, 8)),
            np.ones((8, 8, 8)),
            contract,
            draws=10,
            seed=9,
        )
        self.assertTrue(np.all(np.isfinite(sample)))

    def test_bootstrap_is_by_core_and_selection_can_write_no_finalist(self):
        reference = np.ones(20)
        candidate = np.full(20, 0.95)
        result = paired_core_bootstrap(candidate, reference, seed=2)
        self.assertEqual(result["resampling_unit"], "authoritative patch core")
        self.assertFalse(result["voxel_independent_resampling"])
        thresholds = {
            "tarp": 0.05,
            "global_coverage": 0.05,
            "conditional_coverage": 0.10,
            "joint_improvement": 0.02,
            "other_score_worsening": 0.01,
        }
        reference_report = {
            "proper_scores": {
                "primary_joint": 1.0,
                "energy": 1.0,
                "variogram": 1.0,
            }
        }
        failing = {
            "finite_non_degenerate": True,
            "tarp_maximum_deviation": 0.2,
            "global_coverage_error": {"0.68": 0.1, "0.90": 0.1},
            "maximum_conditional_coverage_error": 0.2,
            "joint_score_vs_g1_bootstrap": {
                "fractional_improvement": 0.01,
                "interval95": [-0.1, 0.1],
            },
            "proper_scores": {
                "primary_joint": 0.99,
                "energy": 1.0,
                "variogram": 1.0,
            },
        }
        marker = freeze_method_selection(
            {
                "gaussian_correlated_g1": reference_report,
                "rectified_flow_f1b": failing,
            },
            thresholds=thresholds,
        )
        self.assertEqual(marker["schema_version"], "p12f-no-field-finalist-v1")
        self.assertIsNone(marker["field_finalist"])

    def test_factorial_views_preserve_truth_and_single_core(self):
        truth = np.ones((4, 4, 4))
        from workflows.sbi.p12f_challenger_common import choose_factorial_view

        views = [
            ConditionedView(
                delta_r7=truth,
                observation=np.zeros((3, 4, 4, 4)),
                response=np.ones((2, 4, 4, 4)),
                support=np.ones_like(truth, bool),
                core_id=7,
                view=name,
            )
            for name in ("dense", "final")
        ]
        first = choose_factorial_view(views, seed=4, update=8)
        second = choose_factorial_view(views, seed=4, update=8)
        self.assertEqual(first.view, second.view)
        bad = list(views)
        bad[1] = ConditionedView(
            delta_r7=np.zeros_like(truth),
            observation=bad[1].observation,
            response=bad[1].response,
            support=bad[1].support,
            core_id=7,
            view="bad",
        )
        with self.assertRaises(ValueError):
            choose_factorial_view(bad, seed=4, update=8)


if __name__ == "__main__":
    unittest.main()
