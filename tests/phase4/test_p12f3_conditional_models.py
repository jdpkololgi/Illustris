import unittest

import numpy as np
import torch
import torch.nn as nn

from workflows.abacus_tweb.p6_field_patch_utils import FieldPatch
from workflows.sbi.p12f3_conditional_models import (
    ConditionalLowModeGaussianUNet,
    conditional_gaussian_nll,
    cosine_alpha_sigma,
    fourier_v_pair,
    proxy_condition,
)


class ConditionalModelsTests(unittest.TestCase):
    def test_frozen_proxy_output_supports_downstream_backpropagation(self):
        shape = (9, 10, 11)
        patch = FieldPatch(
            core_id=0,
            fold=0,
            cap=0,
            channel_names=(
                "counts", "exposure_apodized", "log_count_ratio",
                "distance_to_support_boundary",
            ),
            values=np.stack((
                np.ones(shape, dtype=np.float32),
                np.ones(shape, dtype=np.float32),
                np.zeros(shape, dtype=np.float32),
                np.full(shape, 60.0, dtype=np.float32),
            )),
            context_start=np.zeros(3, dtype=np.int64),
            context_stop=np.asarray(shape),
            core_start=np.zeros(3, dtype=np.int64),
            core_stop=np.asarray(shape),
            core_slice=(slice(None),) * 3,
            authoritative_parent_id=np.empty(0, dtype=np.int64),
            authoritative_frac_index_global=np.empty((0, 3), dtype=np.float32),
            authoritative_frac_index_local=np.empty((0, 3), dtype=np.float32),
            available_halo_low=np.zeros(3, dtype=np.int64),
            available_halo_high=np.zeros(3, dtype=np.int64),
        )
        normalization = {"channels": {
            name: {"policy": "identity"}
            for name in ("counts", "exposure_apodized", "log_count_ratio")
        }}
        class FrozenG1(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Conv3d(3, 2, 1)

            def forward(self, values):
                output = self.net(values)
                return output[:, :1], output[:, 1:]

        frozen_g1 = FrozenG1().eval().requires_grad_(False)
        condition, _, _ = proxy_condition(
            patch, normalization, frozen_g1, device="cpu", arm="proxy7"
        )
        model = ConditionalLowModeGaussianUNet(base=2)
        location, log_scale = model(condition)
        mask = torch.ones_like(location, dtype=torch.bool)
        conditional_gaussian_nll(location, log_scale, torch.zeros_like(location), mask).backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

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
