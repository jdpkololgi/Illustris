"""Small CPU checks; real U-Net patch/backward smoke is compute-node only."""
import unittest
import numpy as np
import torch
from workflows.sbi.p12b_unet_representation_common import (
    safe_path, theta_from_eigen, eigen_from_theta, physical_log_jacobian,
    context_boxes, disjoint_candidates, balanced_cores, build_flow,
    heun_sample, heun_log_prob, row_scores, clustered_difference,
)


class TestP12B(unittest.TestCase):
    def test_no_blind_path(self):
        with self.assertRaises(PermissionError):
            safe_path("/tmp/ph001/anything.npy")

    def test_roundtrip(self):
        truth = np.array([[-1., -.99999, .4], [.2, .7, 32.]])
        np.testing.assert_allclose(eigen_from_theta(theta_from_eigen(truth)), truth, atol=1e-13)

    def test_invalid_target_not_clipped(self):
        for value in ([0., 0., 1.], [0., -1., 1.], [0., np.nan, 1.]):
            with self.assertRaises((ValueError, FloatingPointError)):
                theta_from_eigen(np.array([value]))

    def test_physical_jacobian(self):
        eigen = np.array([[.1, .4, .9]])
        std = np.array([.3, .6, .8])
        eps = 1e-6
        jac = np.column_stack([((theta_from_eigen(eigen+eps*np.eye(3)[d])-theta_from_eigen(eigen-eps*np.eye(3)[d]))/(2*eps)/std)[0] for d in range(3)])
        self.assertAlmostEqual(float(physical_log_jacobian(eigen, std)[0]), float(np.linalg.slogdet(jac)[1]), places=8)

    def test_context_overlap_and_cap(self):
        low, high = context_boxes([[0,0,0], [15,0,0], [80,0,0], [15,0,0]],
                                  [[10,10,10], [25,10,10], [90,10,10], [25,10,10]], 8, 8)
        selected = disjoint_candidates(np.array([1,2,3]), np.array([0,0,0,1]), low, high, [0])
        np.testing.assert_array_equal(selected, [2,3])

    def test_balanced_selection(self):
        cap = np.repeat([0,1], 12)
        shell = np.tile(np.repeat(np.arange(4), 3), 2)
        selected = balanced_cores(np.arange(24), cap, shell, 16, np.random.default_rng(42))
        self.assertEqual(len(np.unique(selected)), 16)
        for c in (0,1):
            for s in range(4):
                self.assertEqual(int(np.sum((cap[selected] == c) & (shell[selected] == s))), 2)

    def test_heun_direction_translation(self):
        context = torch.tensor([[1., 2., -1.], [.2, -.1, .3]], dtype=torch.float64)
        velocity = lambda x, c, t: c + x*0
        sample = heun_sample(velocity, context, torch.zeros_like(context), 8)
        torch.testing.assert_close(sample, -context)
        density = heun_log_prob(velocity, context, sample, 8)
        torch.testing.assert_close(density, torch.full((2,), -1.5*np.log(2*np.pi), dtype=torch.float64))

    def test_log_density_divergence_sign(self):
        theta = torch.tensor([[.2,.3,-.5], [1.,-.1,.4]], dtype=torch.float64)
        velocity = lambda x, c, t: .4*x
        logp = heun_log_prob(velocity, torch.zeros_like(theta), theta, 256)
        expected = -.5*((theta*np.exp(.4))**2+np.log(2*np.pi)).sum(-1)+1.2
        torch.testing.assert_close(logp, expected, atol=1e-6, rtol=1e-6)

    def test_sbi_condition_gradient_and_seed(self):
        torch.set_num_threads(1)
        cfg = dict(seed=42, head_hidden_features=16, head_layers=2, context_dimensions=39)
        model = build_flow(cfg, "cpu")
        condition = torch.randn(8,39, requires_grad=True)
        target = torch.randn(8,3)
        torch.manual_seed(5)
        loss = model.loss(target, condition).mean()
        torch.manual_seed(5)
        replay = model.loss(target, condition).mean()
        torch.testing.assert_close(loss, replay, rtol=0, atol=0)
        loss.backward()
        # Installed SBI initializes its output at zero; condition gradients open
        # after the first shared optimizer update, without changing architecture.
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        condition.grad = None
        torch.manual_seed(6)
        model.loss(target, condition).mean().backward()
        self.assertGreater(float(condition.grad.abs().max()), 0)

    def test_scores_and_cluster_units(self):
        scores = row_scores(np.zeros((4,8,3)), np.ones((4,3)))
        np.testing.assert_allclose(scores["energy"], np.sqrt(3))
        np.testing.assert_allclose(scores["crps"], 1.)
        clusters = np.array([[0,1],[0,1],[1,1],[1,1]])
        result = clustered_difference(np.ones(4), np.zeros(4), clusters, 20, 42)
        self.assertEqual(result["cluster_count"], 2)
        self.assertEqual(result["interval95"], [1.,1.])


if __name__ == "__main__":
    unittest.main()
