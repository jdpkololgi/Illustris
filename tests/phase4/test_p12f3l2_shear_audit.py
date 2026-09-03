import unittest

import torch

from workflows.sbi.p12f3l2_shear_audit import (
    sample_shear_at_galaxies,
    traceless_components,
)


class TestP12F3L2ShearAudit(unittest.TestCase):
    def test_five_components_are_traceless_tensor_coordinates(self):
        tensor = torch.zeros(2, 3, 4, 5, 3, 3)
        tensor[..., 0, 0] = 4.0
        tensor[..., 1, 1] = 1.0
        tensor[..., 2, 2] = -2.0
        tensor[..., 0, 1] = tensor[..., 1, 0] = 0.5
        tensor[..., 0, 2] = tensor[..., 2, 0] = -0.25
        tensor[..., 1, 2] = tensor[..., 2, 1] = 0.75
        value = traceless_components(tensor)
        self.assertEqual(tuple(value.shape), (2, 5, 3, 4, 5))
        self.assertTrue(torch.allclose(value[:, 0], torch.full((2, 3, 4, 5), 3.0)))
        self.assertTrue(torch.allclose(value[:, 1], torch.zeros(2, 3, 4, 5)))
        self.assertTrue(torch.allclose(value[:, 2], torch.full((2, 3, 4, 5), 0.5)))
        self.assertTrue(torch.allclose(value[:, 3], torch.full((2, 3, 4, 5), -0.25)))
        self.assertTrue(torch.allclose(value[:, 4], torch.full((2, 3, 4, 5), 0.75)))

    def test_numpy_sampler_boundary_accepts_torch_physics_output(self):
        fields = torch.zeros(2, 8, 8, 8).numpy()
        coordinates = torch.tensor([[1.5, 2.5, 3.5], [4.0, 4.0, 4.0]]).numpy()
        value = sample_shear_at_galaxies(
            fields, coordinates, device="cpu", batch=1
        )
        self.assertEqual(value.shape, (2, 2, 5))
        self.assertTrue((value == 0).all())


if __name__ == "__main__":
    unittest.main()
