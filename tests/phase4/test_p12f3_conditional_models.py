import unittest

import numpy as np
import torch

from workflows.sbi.p12f3_conditional_models import (
    ConditionalLowModeGaussianUNet,
    conditional_gaussian_nll,
    cosine_alpha_sigma,
    fourier_v_pair,
)


class ConditionalModelsTests(unittest.TestCase):
    def test_gaussian_shapes_and_finite_loss(self):
        model = ConditionalLowModeGaussianUNet(base=2)
        condition = torch.randn(1, 7, 9, 10, 11)
        location, scale = model(condition)
        self.assertEqual(location.shape, (1, 1, 9, 10, 11))
        mask = torch.ones_like(location, dtype=torch.bool)
        loss = conditional_gaussian_nll(location, scale, torch.zeros_like(location), mask)
        self.assertTrue(torch.isfinite(loss))

    def test_fourier_v_algebra(self):
        target = torch.randn(3, 12)
        state, time, velocity = fourier_v_pair(target, generator=torch.Generator().manual_seed(7))
        alpha, sigma = cosine_alpha_sigma(time)
        recovered = alpha[:, None] * state - sigma[:, None] * velocity
        torch.testing.assert_close(recovered, target)

    def test_cosine_endpoints(self):
        alpha, sigma = cosine_alpha_sigma(torch.tensor([0.0, 1.0]))
        np.testing.assert_allclose(alpha.numpy(), [1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(sigma.numpy(), [0.0, 1.0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
