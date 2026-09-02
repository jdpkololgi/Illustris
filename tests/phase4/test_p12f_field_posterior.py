import json
from pathlib import Path
import unittest

import numpy as np
import torch

from workflows.abacus_tweb.p12f_build_field_targets import validate_visible_phase
from workflows.abacus_tweb.p12f_validate_field_targets import (
    read_flat_points,
    sample_indices,
)
from workflows.sbi.p12f_field_posterior_diagnostics import (
    central_coverage,
    fixed_tidal_eigenvalues,
    fixed_tidal_tensor,
    physics_closure_report,
    randomized_ranks,
    scalar_posterior_report,
)
from workflows.sbi.p12f_train_conditional_field_flow import (
    _fourier_parts,
    rectified_flow_training_pair,
    sample_heun,
)
from workflows.sbi.p11_latent_physics_diagnostic import pc1, weighted_correlation


REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs/p12f_conditional_field_flow_v1.json"


class ConstantVelocity(torch.nn.Module):
    def __init__(self, velocity: float):
        super().__init__()
        self.velocity = float(velocity)

    def forward(self, state, time_value, condition):
        return torch.full_like(state, self.velocity)


class P12FFieldPosteriorTest(unittest.TestCase):
    def test_contract_is_field_posterior_and_seals_blind_phase(self):
        contract = json.loads(CONFIG.read_text())
        self.assertEqual(
            contract["estimand"],
            "p(delta_R7 patch | BGS_BRIGHT final view, random-derived response, H_fid)",
        )
        self.assertEqual(contract["roles"]["sealed_blind_test"], "ph001")
        self.assertNotIn("ph001", contract["roles"]["training"])
        self.assertTrue(contract["target"]["double_smoothing_forbidden"])
        self.assertIn(
            "posterior-mean R2(delta_R7)",
            contract["diagnostics_not_promotion_gates"],
        )
        with self.assertRaises(PermissionError):
            validate_visible_phase("ph001")
        self.assertEqual(validate_visible_phase("ph006"), "ph006")

    def test_rectified_flow_pair_has_exact_endpoints_and_velocity(self):
        target = torch.arange(64, dtype=torch.float32).reshape(1, 1, 4, 4, 4)
        generator = torch.Generator().manual_seed(11)
        state, time_value, velocity, noise = rectified_flow_training_pair(
            target, generator=generator
        )
        blend = time_value.view(-1, 1, 1, 1, 1)
        torch.testing.assert_close(state, (1.0 - blend) * noise + blend * target)
        torch.testing.assert_close(velocity, target - noise)
        torch.testing.assert_close(noise + velocity, target)

    def test_heun_integrates_constant_velocity(self):
        condition = torch.zeros(1, 3, 4, 4, 4)
        generator_a = torch.Generator().manual_seed(4)
        generator_b = torch.Generator().manual_seed(4)
        initial = torch.randn((3, 1, 4, 4, 4), generator=generator_a)[:, 0]
        result = sample_heun(
            ConstantVelocity(2.5),
            condition,
            draws=3,
            steps=5,
            generator=generator_b,
        )
        torch.testing.assert_close(result, initial + 2.5)

    def test_fixed_physics_closes_trace_without_extra_smoothing(self):
        generator = torch.Generator().manual_seed(13)
        delta = torch.randn((2, 8, 10, 12), generator=generator)
        tensor = fixed_tidal_tensor(delta)
        eigen = fixed_tidal_eigenvalues(delta)
        eigen_small_chunks = fixed_tidal_eigenvalues(delta, matrix_chunk_size=17)
        self.assertEqual(tuple(tensor.shape), (2, 8, 10, 12, 3, 3))
        self.assertEqual(tuple(eigen.shape), (2, 8, 10, 12, 3))
        torch.testing.assert_close(eigen, eigen_small_chunks)
        trace = torch.diagonal(tensor, dim1=-2, dim2=-1).sum(dim=-1)
        centered = delta - delta.mean(dim=(-3, -2, -1), keepdim=True)
        torch.testing.assert_close(trace, centered, atol=2e-5, rtol=2e-5)
        self.assertTrue(torch.all(eigen[..., 1:] >= eigen[..., :-1]))
        report = physics_closure_report(delta)
        self.assertTrue(report["all_finite"])
        self.assertTrue(report["ordered"])
        self.assertFalse(report["additional_gaussian_smoothing"])

    def test_posterior_rank_and_coverage_shapes(self):
        truth = np.linspace(-1.0, 1.0, 101, dtype=np.float64)
        offsets = np.linspace(-2.0, 2.0, 41, dtype=np.float64)
        samples = truth[None, :] + offsets[:, None]
        ranks = randomized_ranks(samples, truth, seed=7)
        self.assertEqual(ranks.shape, truth.shape)
        self.assertTrue(np.all((ranks >= 0.0) & (ranks <= 1.0)))
        coverage = central_coverage(samples, truth)
        self.assertEqual(set(coverage), {"0.50", "0.68", "0.90"})
        report = scalar_posterior_report(samples, truth, seed=8)
        self.assertAlmostEqual(report["posterior_mean_r2_diagnostic"], 1.0)
        self.assertGreater(report["posterior_width_median"], 0.0)

    def test_fourier_rows_preserve_draw_axis(self):
        rng = np.random.default_rng(9)
        samples = rng.normal(size=(5, 8, 6, 10)).astype(np.float32)
        truth = rng.normal(size=(8, 6, 10)).astype(np.float32)
        sample_modes, truth_modes, kmag = _fourier_parts(samples, truth)
        self.assertEqual(sample_modes.shape[0], 5)
        self.assertEqual(sample_modes.shape[1], truth_modes.shape[0])
        self.assertEqual(truth_modes.shape, kmag.shape)
        self.assertTrue(np.all(kmag > 0.0))

    def test_weighted_pc1_recovers_one_dimensional_signal(self):
        signal = np.linspace(-2.0, 2.0, 50)
        values = np.stack((signal, 2.0 * signal, np.zeros_like(signal)), axis=1)
        weight = np.ones(len(signal))
        score, explained = pc1(values, weight)
        self.assertAlmostEqual(explained, 1.0, places=12)
        self.assertAlmostEqual(abs(weighted_correlation(score, signal, weight)), 1.0)

    def test_field_target_audit_samples_hdf5_by_flat_identity(self):
        import h5py
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.h5"
            expected = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
            with h5py.File(path, "w") as handle:
                handle.create_dataset("values", data=expected)
            flat = sample_indices(expected.shape, 31, seed=8)
            with h5py.File(path, "r") as handle:
                observed = read_flat_points(handle["values"], flat)
            np.testing.assert_array_equal(observed, expected.ravel()[flat])


if __name__ == "__main__":
    unittest.main()
