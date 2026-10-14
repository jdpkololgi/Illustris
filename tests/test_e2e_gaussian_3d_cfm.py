from pathlib import Path
import sys
import unittest
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'workflows/sbi'))
import e2e_gaussian_3d_cfm as g


class Gaussian3DTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_backbone_layers_are_the_actual_project_layers(self):
        from workflows.sbi.e2e_coupled_benchmark_models import CoupledBackbone
        a = g.GaussianBackbone(); b = CoupledBackbone('coarse', 'wide', base=24, levels=3)
        self.assertEqual({k: v.shape for k, v in a.state_dict().items()},
                         {k: v.shape for k, v in b.state_dict().items()})

    def test_full_exterior_has_no_updated_target_leakage(self):
        model = g.GaussianBackbone().eval()
        x = torch.randn(2, 1, 16, 16, 16); observation = torch.randn(2, 3, 16, 16, 16)
        with torch.no_grad():
            first = model.prepare(observation, x, (0, 0, 0), 8)
            x[:, :, :8, :8, :8] += 100
            second = model.prepare(observation, x, (0, 0, 0), 8)
            for a, b in zip(first, second): torch.testing.assert_close(a, b, rtol=0, atol=0)
            x[:, :, -1, -1, -1] += 100
            third = model.prepare(observation, x, (0, 0, 0), 8)
            self.assertFalse(torch.equal(second[1], third[1]))

    def test_prior_fft_matches_dense_covariance(self):
        n = 8; data = g.JointGenerator(n, 'cpu')
        identity = torch.eye(n**3).reshape(n**3, n, n, n)
        covariance = torch.fft.ifftn(torch.fft.fftn(identity, dim=(-3, -2, -1))*data.factor.square(),
            dim=(-3, -2, -1)).real.reshape(n**3, n**3).numpy()+.8
        np.testing.assert_allclose(covariance, g.context.prior(n, 'long_range'), atol=2e-7)

    def test_one_block_training_is_identical(self):
        data = g.JointGenerator(8, 'cpu'); states = []
        for arm in ('whole', 'block'):
            torch.manual_seed(412); model = g.GaussianBackbone()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            generator = torch.Generator().manual_seed(21)
            g.train_step(model, optimizer, data, generator, arm, 0, batch=2)
            states.append(model.state_dict())
        for key in states[0]: torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)

    def test_zero_output_residual_heun_and_fixed_exterior(self):
        model = g.GaussianBackbone().eval(); x = torch.randn(2, 1, 16, 16, 16)
        before = x.clone(); observation = torch.randn(2, 3, 16, 16, 16)
        noise = torch.randn(2, 1, 8, 8, 8)
        result = g.draw(model, observation, x, (0, 0, 0), 8, noise, 8)
        torch.testing.assert_close(result, noise*(1+.25+.25**2/2)**4)
        torch.testing.assert_close(x, before, rtol=0, atol=0)


if __name__ == '__main__': unittest.main()
