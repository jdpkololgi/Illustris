from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'workflows/sbi'))
import e2e_gaussian_transformer as tr
import e2e_gaussian_3d_cfm as g


class TransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        torch.manual_seed(451)
        return tr.GaussianTransformer(width=48, depth=2, heads=4)

    def test_patch_roundtrip_and_channel_order(self):
        x = torch.arange(2*3*8**3).reshape(2, 3, 8, 8, 8)
        tokens = tr.patchify(x)
        torch.testing.assert_close(tr.unpatchify(tokens, 8), x, rtol=0, atol=0)
        torch.testing.assert_close(tokens[0, 0], x[0, :, :2, :2, :2].reshape(-1))
        torch.testing.assert_close(tokens[1, 1], x[1, :, :2, :2, 2:4].reshape(-1))

    def test_geometry_rejects_invalid(self):
        with self.assertRaises(ValueError): tr.patchify(torch.zeros(1, 1, 7, 7, 7))
        with self.assertRaises(ValueError): tr.unpatchify(torch.zeros(1, 9, 8), 8)
        with self.assertRaises(ValueError): tr.GaussianTransformer(width=49, heads=4)

    def test_no_convolutions_and_near_matched_parameters(self):
        model = tr.GaussianTransformer()
        self.assertFalse(any(isinstance(m, torch.nn.Conv3d) for m in model.modules()))
        self.assertLess(abs(sum(p.numel() for p in model.parameters())/8398387-1), .03)
        self.assertEqual(len(model.blocks), 8)

    def test_no_truth_or_exterior_leakage(self):
        model = self.model()
        obs = torch.randn(2, 3, 8, 8, 8)
        exterior = torch.randn(2, 1, 8, 8, 8, requires_grad=True)
        first = model.prepare(obs, exterior, (0, 0, 0), 8)
        second = model.prepare(obs, exterior+1000, (0, 0, 0), 8)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertIsNone(torch.autograd.grad(first.sum(), exterior, allow_unused=True)[0])
        with self.assertRaises(ValueError): model.prepare(obs, exterior, (0, 0, 0), 4)

    def test_zero_head_matches_baseline_initial_flow(self):
        model = self.model().eval()
        noise = torch.randn(2, 1, 8, 8, 8)
        obs = torch.randn(2, 3, 8, 8, 8)
        result = g.draw(model, obs, noise, (0, 0, 0), 8, noise, 8)
        torch.testing.assert_close(result, noise*(1+.25+.25**2/2)**4)

    def test_remote_observation_and_state_can_affect_output(self):
        model = self.model()
        with torch.no_grad():
            model.output.weight.normal_(0, .02)
            for block in model.blocks:
                block.modulation[-1].bias[2*48:3*48] = 1
                block.modulation[-1].bias[5*48:6*48] = 1
        obs = torch.randn(1, 3, 8, 8, 8, requires_grad=True)
        z = torch.randn(1, 1, 8, 8, 8, requires_grad=True)
        value = model.velocity(z, torch.tensor([.4]), model.prepare(obs, z, (0, 0, 0), 8))[0, 0, 0, 0, 0]
        dobs, dz = torch.autograd.grad(value, (obs, z))
        self.assertGreater(float(dobs[0, :, -2:, -2:, -2:].abs().sum()), 0)
        self.assertGreater(float(dz[0, :, -2:, -2:, -2:].abs().sum()), 0)

    def test_training_reproducible_and_rng_budget_unchanged(self):
        states = []
        for _ in range(2):
            model = self.model()
            data = g.JointGenerator(8, 'cpu')
            data_gen = torch.Generator().manual_seed(32)
            bridge_gen = torch.Generator().manual_seed(33)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            for update in range(3): g.train_step(model, opt, data, data_gen, 'whole', update, 2, bridge_gen)
            states.append(model.state_dict())
            reference_data = torch.Generator().manual_seed(32)
            reference_bridge = torch.Generator().manual_seed(33)
            for _ in range(3):
                data.batch(2, reference_data)
                torch.randn((2, 1, 8, 8, 8), generator=reference_bridge)
                torch.rand((), generator=reference_bridge)
            self.assertTrue(torch.equal(data_gen.get_state(), reference_data.get_state()))
            self.assertTrue(torch.equal(bridge_gen.get_state(), reference_bridge.get_state()))
            self.assertGreater(float(model.blocks[0].qkv.weight.grad.abs().sum()), 0)
        for name in states[0]: torch.testing.assert_close(states[0][name], states[1][name], rtol=0, atol=0)


if __name__ == '__main__': unittest.main()
