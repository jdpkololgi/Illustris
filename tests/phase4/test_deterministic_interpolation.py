import unittest
import torch
from torch.nn import functional as F
from shared.deterministic_interpolation import resize_trilinear, sample_points_trilinear, max_pool3d_stride2


class DeterministicInterpolationTests(unittest.TestCase):
    def test_resize_matches_values_and_gradients(self):
        torch.manual_seed(7)
        for shape, target in [((2, 3, 3, 4, 5), (7, 9, 11)), ((1, 2, 1, 2, 3), (2, 3, 5))]:
            x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
            expected = F.interpolate(x, size=target, mode='trilinear', align_corners=False)
            actual = resize_trilinear(x, target)
            torch.testing.assert_close(actual, expected, atol=3e-15, rtol=3e-15)
            weight = torch.randn_like(actual)
            a = torch.autograd.grad((actual*weight).sum(), x, retain_graph=True)[0]
            b = torch.autograd.grad((expected*weight).sum(), x)[0]
            torch.testing.assert_close(a, b, atol=3e-14, rtol=3e-14)

    def test_point_sampling_matches_native_border_and_repeated_gradients(self):
        torch.manual_seed(8)
        x = torch.randn(2, 3, 3, 4, 5, dtype=torch.float64, requires_grad=True)
        points = torch.randn(2, 1, 1, 21, 3, dtype=torch.float64)
        points[..., :4, :] = 0
        points[..., 4, :] = -1
        points[..., 5, :] = 1
        expected = F.grid_sample(x, points, mode='bilinear', padding_mode='border', align_corners=True).flatten(2)
        actual = sample_points_trilinear(x, points)
        torch.testing.assert_close(actual, expected, atol=2e-15, rtol=2e-15)
        weight = torch.randn_like(actual)
        a = torch.autograd.grad((actual*weight).sum(), x, retain_graph=True)[0]
        b = torch.autograd.grad((expected*weight).sum(), x)[0]
        torch.testing.assert_close(a, b, atol=3e-15, rtol=3e-15)

    def test_pool_matches_values_gradients_and_ties(self):
        torch.manual_seed(9)
        for shape in [(2, 3, 5, 6, 7), (1, 2, 1, 3, 2)]:
            for tied in (False, True):
                x = (torch.ones(shape, dtype=torch.float64) if tied else torch.randn(shape, dtype=torch.float64)).requires_grad_()
                actual = max_pool3d_stride2(x)
                expected = F.max_pool3d(x, 2, ceil_mode=True)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                weight = torch.randn_like(actual)
                a = torch.autograd.grad((actual*weight).sum(), x, retain_graph=True)[0]
                b = torch.autograd.grad((expected*weight).sum(), x)[0]
                torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_reject_trainable_geometry(self):
        with self.assertRaises(ValueError):
            sample_points_trilinear(torch.zeros(1, 1, 2, 2, 2), torch.zeros(1, 1, 3, requires_grad=True))

if __name__ == '__main__':
    unittest.main()
