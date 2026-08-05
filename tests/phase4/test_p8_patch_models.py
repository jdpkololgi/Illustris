import unittest

import numpy as np
import torch

from workflows.abacus_tweb.p8_classical_fullcap import _grid_coords
from workflows.abacus_tweb.p8_train_unet_patch import ChannelLayerNorm3d, UPatch, grid_coordinates
from workflows.abacus_tweb.p8_train_unet_cic_residual import (
    UCICResidual,
    checkpoint_zero_parity,
    physical_to_scaled,
)


class P8PatchModelTests(unittest.TestCase):
    def test_channel_layer_norm_is_spatially_local(self):
        torch.manual_seed(3)
        layer = ChannelLayerNorm3d(5)
        small = torch.randn(1, 5, 4, 5, 6)
        large = torch.randn(1, 5, 8, 9, 10)
        large[:, :, 2:6, 2:7, 2:8] = small
        got_small = layer(small)
        got_large = layer(large)[:, :, 2:6, 2:7, 2:8]
        torch.testing.assert_close(got_small, got_large)

    def test_unet_has_no_spatial_normalizer(self):
        model = UPatch(base=4, latent_channels=8)
        forbidden = (torch.nn.GroupNorm, torch.nn.InstanceNorm3d, torch.nn.BatchNorm3d)
        self.assertFalse(any(isinstance(module, forbidden) for module in model.modules()))

    def test_grid_coordinate_conventions_agree(self):
        frac = np.array([[0.0, 2.0, 4.0], [4.0, 3.0, 0.0]], dtype=np.float64)
        shape = (5, 4, 5)
        field = torch.arange(np.prod(shape), dtype=torch.float32).reshape(1, 1, *shape)
        for maker in (_grid_coords, grid_coordinates):
            coordinates = maker(frac, shape, "cpu")
            sampled = torch.nn.functional.grid_sample(
                field, coordinates, mode="bilinear", align_corners=True,
                padding_mode="border",
            )[0, 0, 0, 0].numpy()
            expected = np.array([field[0, 0, 0, 2, 4], field[0, 0, 4, 3, 0]])
            np.testing.assert_allclose(sampled, expected)

    def test_u_cic_residual_zero_is_exact_and_always_ordered(self):
        scaler = {"mean": [-0.1, 0.2, 0.25], "std": [0.3, 0.1, 0.12]}
        model = UCICResidual(scaler, base=4, latent_channels=8, head_width=16)
        values = torch.randn(1, 3, 8, 8, 8)
        points = torch.tensor([[[[[-0.5, 0.0, 0.5], [0.1, -0.2, 0.3]]]]])
        cic = torch.tensor([[-0.4, 0.1, 0.3], [0.2, 0.25, 0.9]])
        parity = checkpoint_zero_parity(model, values, points, cic)
        self.assertTrue(parity["pass"])
        self.assertLessEqual(parity["maximum_absolute_eigenvalue_difference"], 2.0e-6)

        with torch.no_grad():
            model.head[-1].bias[:] = torch.tensor([1.0, -5.0, 5.0])
        scaled, predicted, correction = model(values, points, cic)
        self.assertTrue(torch.all(predicted[:, 1] > predicted[:, 0]))
        self.assertTrue(torch.all(predicted[:, 2] > predicted[:, 1]))
        self.assertEqual(tuple(scaled.shape), (2, 3))
        self.assertEqual(tuple(correction.shape), (2, 3))

    def test_u_cic_scaled_null_matches_common_increment_contract(self):
        scaler = {"mean": [-0.1, 0.2, 0.25], "std": [0.3, 0.1, 0.12]}
        model = UCICResidual(scaler, base=4, latent_channels=8, head_width=16)
        values = torch.randn(1, 3, 8, 8, 8)
        points = torch.zeros(1, 1, 1, 2, 3)
        cic_np = np.asarray([[-0.4, 0.1, 0.3], [0.2, 0.25, 0.9]], dtype=np.float32)
        with torch.no_grad():
            scaled, _, _ = model(values, points, torch.from_numpy(cic_np))
        np.testing.assert_allclose(
            scaled.numpy(), physical_to_scaled(cic_np, scaler), rtol=1e-6, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
