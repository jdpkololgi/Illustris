from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from workflows.abacus_tweb.p6_field_patch_utils import trilinear_sample
from workflows.sbi.p12f_dependency_rescue_evaluator import (
    _spatial_pair_diagnostics,
    _subpanel_labels,
    aggregate_compact,
    tarp_curve,
    tidal_eigenvalues_at_galaxies,
)
from workflows.sbi.p12f_field_posterior_diagnostics import fixed_tidal_eigenvalues
from workflows.sbi.plot_p12f_dependency_rescue import (
    render_dependency,
    render_nested_tarp,
    render_seed_and_response_stability,
    validate_report,
)


class P12FDependencyRescueEvaluatorTest(unittest.TestCase):
    def test_sparse_eigendecomposition_matches_full_grid_interpolation(self):
        generator = torch.Generator().manual_seed(7)
        delta = torch.randn((3, 7, 8, 6), generator=generator)
        coordinates = np.asarray(
            [[0.2, 1.7, 3.1], [6.8, -0.4, 5.7], [3.0, 4.0, 2.0]],
            dtype=np.float32,
        )
        expected_grid = fixed_tidal_eigenvalues(delta).numpy()
        expected = trilinear_sample(
            np.moveaxis(expected_grid, -1, 1).reshape(9, 7, 8, 6),
            coordinates,
        ).reshape(len(coordinates), 3, 3).transpose(1, 0, 2)
        actual, closure = tidal_eigenvalues_at_galaxies(delta, coordinates)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
        self.assertTrue(closure["all_finite"])
        self.assertTrue(closure["all_ordered"])
        self.assertLess(closure["trace_max_abs"], 2e-6)

    def test_subpanels_are_shell_balanced_and_disjoint(self):
        core = np.arange(40, dtype=np.int64)[::-1]
        shell = np.repeat(np.arange(4, dtype=np.int8), 10)
        labels = _subpanel_labels(core, shell)
        self.assertEqual(set(labels.tolist()), {0, 1, 2, 3})
        for value in range(4):
            counts = np.bincount(shell[labels == value], minlength=4)
            self.assertLessEqual(int(counts.max() - counts.min()), 1)

    def test_sparse_eigendecomposition_accepts_no_galaxies(self):
        delta = torch.zeros((2, 4, 4, 4), dtype=torch.float32)
        actual, closure = tidal_eigenvalues_at_galaxies(
            delta, np.empty((0, 3), dtype=np.float32)
        )
        self.assertEqual(actual.shape, (2, 0, 3))
        self.assertTrue(closure["all_finite"])

    def test_spatial_diagnostic_handles_single_supported_voxel(self):
        samples = np.zeros((4, 2, 2, 2), dtype=np.float32)
        truth = np.zeros((2, 2, 2), dtype=np.float32)
        support = np.zeros_like(truth, dtype=bool)
        support[0, 0, 0] = True
        report = _spatial_pair_diagnostics(samples, truth, support, seed=1)
        self.assertTrue(all(value.size == 0 for value in report.values()))

    def test_tarp_curve_is_near_diagonal_for_exchangeable_draws(self):
        rng = np.random.default_rng(3)
        ensemble = rng.normal(size=(128, 4000, 2)).astype(np.float32)
        truth = rng.normal(size=(4000, 2)).astype(np.float32)
        report = tarp_curve(ensemble, truth, seed=11)
        self.assertLess(report["maximum_deviation"], 0.06)
        self.assertEqual(report["alpha"][0], 0.0)

    def test_plot_contract_rejects_opened_or_incomplete_report(self):
        report = {
            "schema_version": "p12f-dependency-rescue-evaluation-v2",
            "phase": "ph006",
            "ph001_opened": False,
            "nested_draw_reports": {str(value): {} for value in (64, 128, 256)},
            "subpanel_reports_256_draws": {str(value): {} for value in range(4)},
            "physics_closure": {"all_finite": True, "all_ordered": True},
        }
        validate_report(report)
        report["ph001_opened"] = True
        with self.assertRaises(PermissionError):
            validate_report(report)

    def test_compact_aggregation_and_core_plots_complete(self):
        rng = np.random.default_rng(17)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = []
            for core_id in range(16):
                truth = rng.normal(size=12).astype(np.float32)
                samples = (truth[None] + rng.normal(scale=0.4, size=(256, 12))).astype(np.float32)
                lambda_truth = np.sort(rng.normal(size=(8, 3)), axis=1).astype(np.float32)
                lambda_samples = np.sort(
                    lambda_truth[None] + rng.normal(scale=0.3, size=(256, 8, 3)), axis=2
                ).astype(np.float32)
                path = root / f"core_{core_id}.npz"
                np.savez_compressed(
                    path,
                    voxel_samples=samples,
                    voxel_truth=truth,
                    voxel_response=np.linspace(0.2, 1.0, 12, dtype=np.float32),
                    voxel_boundary=np.linspace(0.0, 20.0, 12, dtype=np.float32),
                    voxel_tracer=np.linspace(0.0, 0.2, 12, dtype=np.float32),
                    lambda_samples=lambda_samples,
                    lambda_truth=lambda_truth,
                    pair_distance_voxels=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
                    truth_residual_product=np.asarray([0.2, 0.1, 0.0], dtype=np.float32),
                    posterior_residual_covariance=np.asarray([0.18, 0.08, 0.01], dtype=np.float32),
                    truth_residual_variogram=np.asarray([0.3, 0.5, 0.7], dtype=np.float32),
                    posterior_residual_variogram=np.asarray([0.32, 0.52, 0.68], dtype=np.float32),
                    spectral_edges=np.linspace(0.0, np.sqrt(0.75), 13, dtype=np.float32),
                    spectral_count=np.ones(12, dtype=np.int64),
                    truth_innovation_power_sum=np.linspace(1.0, 0.1, 12),
                    posterior_residual_power_sum=np.linspace(0.9, 0.11, 12),
                )
                entries.append(
                    {
                        "core_id": core_id,
                        "shell": core_id // 4,
                        "cap": core_id % 2,
                        "voxel_rows": 12,
                        "galaxy_rows": 8,
                        "compact_path": str(path),
                        "sample_physics": {"trace_max_abs": 1e-7, "trace_rmse": 1e-8, "all_finite": True, "all_ordered": True},
                        "truth_physics": {"trace_max_abs": 1e-7, "trace_rmse": 1e-8, "all_finite": True, "all_ordered": True},
                    }
                )
            report = aggregate_compact(entries, {})
            validate_report(report)
            render_nested_tarp(report, root / "nested")
            render_dependency(report, root / "dependency")
            render_seed_and_response_stability(report, root / "stability")
            for stem in ("nested", "dependency", "stability"):
                self.assertTrue((root / f"{stem}.png").is_file())


if __name__ == "__main__":
    unittest.main()
