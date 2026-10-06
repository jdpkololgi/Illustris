import unittest

import numpy as np
import torch

from workflows.sbi.e2e_cfm_conditioning_model import corrupt_endpoint, augment_condition
from workflows.sbi.e2e_cfm_conditioning_train import step_inputs, profile
from workflows.sbi.e2e_cfm_conditioning_gates import classify, oracle_screen
from workflows.sbi.e2e_coupled_benchmark_models import FieldCondition
from workflows.sbi.e2e_vdm_context_models import replicate, decode_density, block_mean


class ConditioningTests(unittest.TestCase):
    def test_bridge_and_input_immutability(self):
        c = torch.ones(2, 1, 3, 4, 5); old = c.clone(); eta = torch.full_like(c, 3)
        out = corrupt_endpoint(c, torch.tensor([0., .2]), eta)
        self.assertTrue(torch.equal(out[0], c[0]))
        torch.testing.assert_close(out[1], torch.full_like(c[1], 1.4))
        self.assertTrue(torch.equal(c, old))

    def test_strength_rejections(self):
        c = torch.ones(2, 1, 2, 2, 2)
        for s in (torch.tensor([-.1, 0.]), torch.tensor([0., .21]),
                  torch.tensor([0., float('nan')]), torch.tensor([0.])):
            with self.assertRaises(ValueError):
                corrupt_endpoint(c, s, c)

    @staticmethod
    def condition():
        wide = torch.arange(48**3, dtype=torch.float32).reshape(1, 1, 48, 48, 48)/48**3
        return FieldCondition(torch.zeros(1), torch.zeros(1), torch.zeros(1, 3),
            'joint', replicate(wide[:, :, 16:32, 18:30, 18:30]), wide, 'training_truth')

    def test_aligned_crop_and_zero_identity(self):
        c = self.condition(); eta = torch.ones_like(c.coarse_wide)
        clean = augment_condition(c, torch.zeros(1), eta)
        self.assertTrue(torch.equal(clean.coarse_joint, c.coarse_joint))
        noisy = augment_condition(c, torch.tensor([.1]), eta)
        self.assertTrue(torch.equal(noisy.coarse_joint,
            replicate(noisy.coarse_wide[:, :, 16:32, 18:30, 18:30])))

    def test_paired_fine_rng_and_repeatability(self):
        x = torch.zeros(1, 1, 8, 8, 8); c = self.condition(); p = profile()
        a = step_inputs(x, c, 'control', 12, p)
        b = step_inputs(x, c, 'augmented', 12, p)
        repeat = step_inputs(x, c, 'augmented', 12, p)
        for first, second in zip(a[:3], b[:3]):
            self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(b[3].coarse_wide, repeat[3].coarse_wide))
        self.assertTrue(torch.equal(a[3].coarse_wide, c.coarse_wide))

    def test_uncorrupted_mass_decoder(self):
        mass = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8.], dtype=torch.float64).reshape(1,1,2,2,2)
        u = torch.linspace(-1, 1, 8**3, dtype=torch.float64).reshape(1,1,8,8,8)
        rho = decode_density(mass, u)
        torch.testing.assert_close(block_mean(rho), mass, rtol=1e-14, atol=1e-14)

    def test_gate_three_way(self):
        self.assertEqual(classify(1, .99, 1.01, .95, 1.05), 'pass')
        self.assertEqual(classify(1, .94, 1.01, .95, 1.05), 'inconclusive')
        self.assertEqual(classify(.9, .89, 1.01, .95, 1.05), 'fail')

    def test_oracle_gate_rejects_overshoot_and_regression(self):
        p = np.ones((32,2,2)); w = np.ones((32,2))*.997
        refs = dict(parent=w, control=w)
        r = oracle_screen(p, w, np.ones((2,2)), np.ones(2), refs)
        self.assertEqual(r['decision'], 'pass')
        self.assertEqual(oracle_screen(p*1.1,w,np.ones((2,2)),np.ones(2),refs)['decision'], 'fail')
        self.assertEqual(oracle_screen(p,w*1.02,np.ones((2,2)),np.ones(2),refs)['decision'], 'fail')


if __name__ == '__main__':
    unittest.main()
