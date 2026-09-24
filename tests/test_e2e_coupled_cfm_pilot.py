import unittest
import numpy as np
import torch
from workflows.sbi.e2e_coupled_cfm_pilot import spectral_weights,weighted_loss,pair
from workflows.sbi.e2e_vdm_context_metrics import central_order_interval
from workflows.sbi.e2e_vdm_context_models import project,block_mean


class PilotTests(unittest.TestCase):
    def test_parseval_and_normalization(self):
        x=torch.arange(64,dtype=torch.float64).reshape(1,1,4,4,4)/10
        w=torch.as_tensor(spectral_weights(np.ones((4,4,4)),.3))
        self.assertAlmostEqual(float(weighted_loss(x,w)),float(x.square().mean()),places=10)
        weights=spectral_weights(np.arange(64).reshape(4,4,4),.3)
        self.assertTrue(np.isfinite(weights).all())
        self.assertAlmostEqual(float(np.mean(weights**2)),1.)
    def test_nontrain_access_rejected_before_io(self):
        for phase in ['ph012','ph014','ph001','ph004']:
            with self.assertRaises(PermissionError):pair((phase,'not_a_real_pair'),'bad')
    def test_projected_bridge_and_velocity(self):
        x=project(torch.randn(2,1,8,8,8));eps=project(torch.randn_like(x))
        for value in [.0,.3,1.]:
            self.assertLess(float(block_mean((1-value)*eps+value*x).abs().max()),1e-6)
        self.assertLess(float(block_mean(x-eps).abs().max()),1e-6)
    def test_finite_ensemble_reference(self):
        self.assertAlmostEqual(central_order_interval(128,.9)[2],117/129)


if __name__=='__main__':unittest.main()
