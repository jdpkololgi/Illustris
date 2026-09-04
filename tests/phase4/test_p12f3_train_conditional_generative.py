import unittest

import torch

from workflows.sbi.p12f3_conditional_models import fourier_v_pair


class ConditionalGenerativeTrainingTests(unittest.TestCase):
    def test_v_target_is_finite_and_shape_preserving(self):
        target = torch.randn(1, 41)
        state, time, velocity = fourier_v_pair(target, generator=torch.Generator().manual_seed(4))
        self.assertEqual(state.shape, target.shape)
        self.assertEqual(velocity.shape, target.shape)
        self.assertEqual(time.shape, (1,))
        self.assertTrue(torch.isfinite(velocity).all())


if __name__ == "__main__":
    unittest.main()
