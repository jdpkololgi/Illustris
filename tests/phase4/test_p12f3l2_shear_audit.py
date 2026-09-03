import unittest

import torch

from workflows.sbi.p12f3l2_shear_audit import traceless_components


class TestP12F3L2ShearAudit(unittest.TestCase):
    def test_five_components_are_traceless_tensor_coordinates(self):
        tensor = torch.zeros(2, 3, 4, 3, 3)
        tensor[..., 0, 0] = 4.0
        tensor[..., 1, 1] = 1.0
        tensor[..., 2, 2] = -2.0
        tensor[..., 0, 1] = tensor[..., 1, 0] = 0.5
        tensor[..., 0, 2] = tensor[..., 2, 0] = -0.25
        tensor[..., 1, 2] = tensor[..., 2, 1] = 0.75
        value = traceless_components(tensor)
        self.assertEqual(tuple(value.shape), (2, 5, 3, 4))
        self.assertTrue(torch.allclose(value[:, 0], torch.full((2, 3, 4), 3.0)))
        self.assertTrue(torch.allclose(value[:, 1], torch.zeros(2, 3, 4)))
        self.assertTrue(torch.allclose(value[:, 2], torch.full((2, 3, 4), 0.5)))
        self.assertTrue(torch.allclose(value[:, 3], torch.full((2, 3, 4), -0.25)))
        self.assertTrue(torch.allclose(value[:, 4], torch.full((2, 3, 4), 0.75)))


if __name__ == "__main__":
    unittest.main()
