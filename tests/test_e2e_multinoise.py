import json
import math
import unittest
import torch
from workflows.sbi.e2e_wide_models import ConditionalFieldNet
from workflows.sbi.e2e_multinoise_models import (build, coefficients, haar_matrix, haar, inverse_haar,
                                               pack_voxels, unpack_voxels, partition, unpartition,
                                               WindowBlock, loss_for)
from workflows.sbi.e2e_multinoise_test import CONFIG, exposure


class MultiNoiseTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(33); torch.set_num_threads(1)
        self.x = torch.randn(1, 1, 8, 8, 8)
        self.c = torch.randn(1, 2, 8, 8, 8); self.w = torch.randn(1, 2, 8, 8, 8)

    def model(self, arm):
        torch.manual_seed(123)
        return build(arm, ConditionalFieldNet(2, base_channels=4, levels=3, wide_condition_channels=2))

    def test_endpoint_coefficients(self):
        a, b, d = coefficients(torch.tensor([0., .5, 1.]))
        self.assertTrue(torch.isfinite(1/d).all())
        self.assertLessEqual(float((1/d).max()), 20.)
        torch.testing.assert_close(a.square()+b.square(), torch.ones_like(a))

    def test_initial_output_and_backward_all_arms(self):
        for arm in json.loads(CONFIG.read_text())['arms']:
            model = self.model(arm)
            for t in (0., .03, 1.):
                v = model(self.x, torch.tensor([t]), self.c, wide_condition=self.w)
                torch.testing.assert_close(v, torch.zeros_like(self.x))
            loss, _, _ = loss_for(model, self.x, torch.randn_like(self.x), .03, self.c, self.w)
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            self.assertTrue(any(float(p.grad.abs().sum()) > 0 for p in model.parameters() if p.grad is not None))

    def test_v_loss_identity_and_gradient(self):
        t = torch.tensor([.0318]); a, b, d = coefficients(t)
        noise = torch.randn_like(self.x); raw = torch.randn_like(self.x, requires_grad=True)
        state = a*self.x+b*noise
        clean = a*state+(b/d)*raw
        loss1 = ((-raw/d)-(a*noise-b*self.x)).square().mean()
        loss2 = ((clean-self.x)/b).square().mean()
        torch.testing.assert_close(loss1, loss2)
        torch.testing.assert_close(torch.autograd.grad(loss1, raw, retain_graph=True)[0], torch.autograd.grad(loss2, raw)[0])

    def test_haar_roundtrip_energy_and_grad(self):
        x = torch.randn(2, 3, 6, 8, 10, requires_grad=True); matrix = haar_matrix()
        z = haar(x, matrix); restored = inverse_haar(z, matrix)
        torch.testing.assert_close(restored, x)
        torch.testing.assert_close(z.square().sum(), x.square().sum())
        restored.sum().backward(); torch.testing.assert_close(x.grad, torch.ones_like(x))
        torch.testing.assert_close(unpack_voxels(pack_voxels(x)), x)

    def test_window_roundtrip_padding_no_wrap(self):
        for shape in ((1, 3, 5, 7, 4), (2, 8, 8, 8, 4), (1, 1, 1, 1, 4)):
            x = torch.randn(shape)
            for offset in (0, 2):
                z, mask, meta = partition(x, offset=offset)
                torch.testing.assert_close(unpartition(z, meta), x)
                self.assertEqual(int(mask.sum()), math.prod(shape[:-1]))
                self.assertTrue(mask.any(dim=1).all())
                self.assertTrue((z[~mask] == 0).all())

    def test_attention_gradients_and_padding(self):
        block = WindowBlock(16, offset=2)
        x = torch.randn(1, 3, 5, 7, 16, requires_grad=True)
        result = block(x, torch.randn(1, 64)); result.square().mean().backward()
        self.assertTrue(torch.isfinite(result).all())
        self.assertGreater(float(block.qkv.weight.grad.abs().sum()), 0.)

    def test_film_hidden_initialization_matches(self):
        a, b = self.model('unet_residual'), self.model('unet_film')
        for key, value in a.net.base.state_dict().items():
            torch.testing.assert_close(value, b.net.base.state_dict()[key])

    def test_dropout_and_clean_objective(self):
        model = self.model('unet_film_drop')
        # Nonzero FiLM/head to exercise actual dependence on present and inputs.
        torch.nn.init.normal_(model.net.film.weight, std=.1)
        torch.nn.init.normal_(model.net.base.output.weight, std=.1)
        noise = torch.randn_like(self.x)
        a = loss_for(model, self.x, noise, .03, self.c, self.w, .1, True)
        b = loss_for(model, self.x, noise, .03, self.c*23, self.w*11, .1, True)
        torch.testing.assert_close(a[0], b[0])
        torch.testing.assert_close(a[0].detach(), a[1]+.1*a[2])
        self.assertGreater(float(a[2]), 0.)

    def test_exposure_phase_coverage_reproducible(self):
        cfg = json.loads(CONFIG.read_text())
        for step in range(18):
            t, ratio, _ = exposure(step, cfg)
            low, high = cfg['noise_bins'][(step//3) % 6]
            self.assertTrue(low <= ratio <= high)
            self.assertEqual(exposure(step, cfg), exposure(step, cfg))
            self.assertTrue(0 < t < 1)
        self.assertEqual(exposure(63, cfg)[:2], (1., None))
        self.assertEqual(cfg['evaluate_at'][-1], cfg['updates'])


if __name__ == '__main__':
    unittest.main()
