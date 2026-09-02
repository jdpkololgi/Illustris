from __future__ import annotations

import unittest

import torch

from workflows.sbi.p12f_score_diffusion import (
    ConditionalVDiffusionUNet,
    cosine_alpha_sigma,
    diffusion_training_pair,
    recover_x0_epsilon,
    sample_ddim,
    sampler_comparison_contract,
    v_parameterization,
)


class ZeroVDiffusion(torch.nn.Module):
    condition_channels = 3

    def forward(self, noisy, time_value, condition):
        del time_value, condition
        return torch.zeros_like(noisy)


class P12FScoreDiffusionTest(unittest.TestCase):
    def test_cosine_schedule_is_variance_preserving(self):
        time = torch.linspace(0.0, 1.0, 101)
        alpha, sigma = cosine_alpha_sigma(time)
        torch.testing.assert_close(alpha.square() + sigma.square(), torch.ones_like(time))
        self.assertTrue(torch.all(alpha[:-1] >= alpha[1:]))
        self.assertTrue(torch.all(sigma[:-1] <= sigma[1:]))

    def test_v_algebra_recovers_target_and_noise(self):
        generator = torch.Generator().manual_seed(8)
        target = torch.randn((3, 1, 4, 4, 4), generator=generator)
        noise = torch.randn(target.shape, generator=generator)
        time = torch.tensor([0.1, 0.5, 0.9])
        noisy, velocity = v_parameterization(target, noise, time)
        recovered_target, recovered_noise = recover_x0_epsilon(noisy, velocity, time)
        torch.testing.assert_close(recovered_target, target)
        torch.testing.assert_close(recovered_noise, noise)

    def test_training_pair_is_repeatable(self):
        target = torch.ones((2, 1, 3, 3, 3))
        first = diffusion_training_pair(
            target, generator=torch.Generator().manual_seed(3)
        )
        second = diffusion_training_pair(
            target, generator=torch.Generator().manual_seed(3)
        )
        for left, right in zip(first, second, strict=True):
            torch.testing.assert_close(left, right)

    def test_model_geometry_and_ddim_repeatability(self):
        model = ZeroVDiffusion()
        condition = torch.zeros((1, 3, 4, 4, 4))
        first = sample_ddim(
            model,
            condition,
            draws=3,
            steps=5,
            generator=torch.Generator().manual_seed(9),
        )
        second = sample_ddim(
            model,
            condition,
            draws=3,
            steps=5,
            generator=torch.Generator().manual_seed(9),
        )
        torch.testing.assert_close(first, second)
        self.assertEqual(first.shape, (3, 4, 4, 4))

    def test_sampler_contract_counts_network_evaluations(self):
        contract = sampler_comparison_contract()
        self.assertEqual(contract["rectified_flow_reference"]["network_evaluations"], 24)
        self.assertEqual(contract["diffusion_matched"]["steps"], 24)
        self.assertIn("not a universal", contract["interpretation"])

    def test_real_unet_validates_condition_geometry(self):
        model = ConditionalVDiffusionUNet(condition_channels=3, base=2)
        state = torch.zeros((1, 1, 8, 8, 8))
        condition = torch.zeros((1, 3, 8, 8, 8))
        output = model(state, torch.tensor([0.5]), condition)
        self.assertEqual(output.shape, state.shape)
        with self.assertRaises(ValueError):
            model(state, torch.tensor([0.5]), condition[:, :2])


if __name__ == "__main__":
    unittest.main()
