"""Tiny synthetic engineering checks, not native training/science validation."""
import unittest
import torch
from workflows.sbi.e2e_wide_models import (
    ConditionalFieldNet, crop_children, diffusion_loss, flow_matching_loss, sample_field,
    _downsample_mean, _WideSpatialMean,
)


class WideModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_matched_losses_and_gradients(self):
        model = ConditionalFieldNet(2, base_channels=4, levels=2)
        target, condition = torch.ones(1, 1, 8, 8, 8), torch.ones(1, 2, 8, 8, 8)
        for loss_fn in (flow_matching_loss, diffusion_loss):
            model.zero_grad(set_to_none=True)
            loss = loss_fn(model, target, condition, torch.Generator().manual_seed(3))
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertGreater(sum(p.grad.abs().sum().item() for p in model.parameters()), 0)

    def test_deterministic_parent_and_overlap(self):
        model = ConditionalFieldNet(2, base_channels=4, levels=2)
        condition = torch.ones(1, 2, 8, 8, 8)
        for method in ("cfm", "diffusion"):
            kwargs = dict(model=model, condition=condition, method=method, steps=2)
            a = sample_field(**kwargs, generator=torch.Generator().manual_seed(7))
            b = sample_field(**kwargs, generator=torch.Generator().manual_seed(7))
            self.assertTrue(torch.equal(a, b))
            self.assertTrue(model.training)
            left, right = crop_children(a, [(0, 0, 0), (2, 0, 0)], 6)
            self.assertTrue(torch.equal(left[:, :, 2:], right[:, :, :4]))
            self.assertTrue(torch.equal(crop_children(a, [(2, 0, 0)], 6)[0], right))

    def test_generator_resume(self):
        model = ConditionalFieldNet(1, base_channels=4, levels=1)
        condition = torch.zeros(1, 1, 8, 8, 8)
        generator = torch.Generator().manual_seed(4)
        saved = generator.get_state()
        a = sample_field(model, condition, "cfm", 1, generator, solver="euler")
        generator.set_state(saved)
        b = sample_field(model, condition, "cfm", 1, generator, solver="euler")
        self.assertTrue(torch.equal(a, b))

    def test_deterministic_reductions_even_and_odd(self):
        for shape in ((8, 8, 8), (9, 7, 5)):
            x = torch.arange(2*shape[0]*shape[1]*shape[2], dtype=torch.float32).reshape(1, 2, *shape)
            torch.testing.assert_close(_downsample_mean(x), torch.nn.functional.avg_pool3d(x, 2))
            torch.testing.assert_close(_WideSpatialMean()(x), torch.nn.functional.adaptive_avg_pool3d(x, 2))
        previous = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            model = ConditionalFieldNet(2, base_channels=4, levels=3, wide_condition_channels=3)
            for shape in ((8, 8, 8), (9, 7, 5)):
                state = torch.ones(1, 1, *shape)
                result = model(state, torch.zeros(1), torch.ones(1, 2, *shape),
                               wide_condition=torch.ones(1, 3, 9, 7, 5))
                self.assertEqual(result.shape, state.shape)
                result.square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        finally:
            torch.use_deterministic_algorithms(previous)

    def test_wide_observations_reach_fine_model(self):
        model = ConditionalFieldNet(2, base_channels=4, wide_condition_channels=3)
        target, condition = torch.ones(1, 1, 8, 8, 8), torch.zeros(1, 2, 8, 8, 8)
        wide = torch.zeros(1, 3, 8, 8, 8)
        other = torch.ones_like(wide)
        for objective in (flow_matching_loss, diffusion_loss):
            a = objective(model, target, condition, torch.Generator().manual_seed(2), wide_condition=wide)
            b = objective(model, target, condition, torch.Generator().manual_seed(2), wide_condition=other)
            self.assertNotEqual(a.item(), b.item())
            b.backward()
            self.assertTrue(all(p.grad is not None for p in model.wide_encoder.parameters()))
        for method in ("cfm", "diffusion"):
            a = sample_field(model, condition, method, 2, torch.Generator().manual_seed(2), wide_condition=wide)
            b = sample_field(model, condition, method, 2, torch.Generator().manual_seed(2), wide_condition=other)
            self.assertFalse(torch.equal(a, b))

    def test_exact_endpoint_conventions(self):
        class ConstantVelocity(torch.nn.Module):
            def forward(self, state, time, condition):
                return torch.ones_like(state)
        condition = torch.zeros(1, 1, 2, 2, 2)
        noise = torch.randn(condition.shape, generator=torch.Generator().manual_seed(1))
        for solver in ("euler", "heun"):
            result = sample_field(ConstantVelocity(), condition, "cfm", 4,
                                  torch.Generator().manual_seed(1), solver=solver)
            torch.testing.assert_close(result, noise + 1)
        # At t=1 a constant v=1 implies clean=-1, no division by alpha=0.
        result = sample_field(ConstantVelocity(), condition, "diffusion", 1,
                              torch.Generator().manual_seed(1))
        torch.testing.assert_close(result, -torch.ones_like(result))

    def test_invalid_shapes_and_crops(self):
        model = ConditionalFieldNet(2, base_channels=4)
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 1, 8, 8, 8), torch.zeros(1), torch.zeros(1, 1, 8, 8, 8))
        with self.assertRaises(ValueError):
            crop_children(torch.zeros(1, 1, 8, 8, 8), [(5, 0, 0)], 4)


if __name__ == "__main__":
    unittest.main()
