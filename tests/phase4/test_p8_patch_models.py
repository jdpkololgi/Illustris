import unittest

import numpy as np
import torch

from workflows.abacus_tweb.p8_classical_fullcap import _grid_coords
from workflows.abacus_tweb.p8_train_unet_patch import ChannelLayerNorm3d, UPatch, grid_coordinates


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


if __name__ == "__main__":
    unittest.main()
