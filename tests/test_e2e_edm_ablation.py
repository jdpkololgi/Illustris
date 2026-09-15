import copy
import math
import unittest
import torch

from workflows.sbi.e2e_wide_models import ConditionalFieldNet
from workflows.sbi.e2e_edm_ablation_models import NoiseAdaptedNet, edm_coefficients, edm_loss
from workflows.sbi.e2e_fine_learning_test import prediction_loss


class EDMAblationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(42)
        self.base=ConditionalFieldNet(2,base_channels=2,levels=2,wide_condition_channels=2)
        self.x=torch.randn(1,1,8,8,8);self.condition=torch.randn(1,2,8,8,8)
        self.noise=torch.randn_like(self.x)

    def test_identity_adapters_and_finite_endpoints(self):
        for film,rf in [(False,False),(True,False),(True,True)]:
            model=NoiseAdaptedNet(copy.deepcopy(self.base),film=film,receptive=rf)
            for t in (0.,.01,.5,1.):
                time=self.x.new_tensor([t])
                ref=self.base(self.x,time,self.condition,self.condition)
                got=model(self.x,time,self.condition,self.condition)
                torch.testing.assert_close(got,ref,atol=0,rtol=0)
                self.assertTrue(torch.isfinite(got).all())

    def test_edm_v_denoiser_loss_and_gradient_equivalence(self):
        model=NoiseAdaptedNet(self.base)
        for sigma in (.05,.2,1.,5.,20.):
            model.zero_grad()
            t=2*math.atan(sigma)/math.pi
            loss=edm_loss(model,self.x,self.noise,sigma,self.condition,self.condition)
            loss.backward();grad=self.base.output.weight.grad.clone()
            model.zero_grad()
            ref=prediction_loss(model,self.x,self.noise,t,self.condition,self.condition,'diffusion')
            ref.backward()
            torch.testing.assert_close(loss,ref,atol=2e-6,rtol=2e-6)
            torch.testing.assert_close(grad,self.base.output.weight.grad,atol=2e-6,rtol=2e-6)
            z=self.x+sigma*self.noise
            clean=model.denoise(z,sigma,self.condition,self.condition)
            weighted=((clean-self.x)**2).mean()*(1+sigma**2)/sigma**2
            torch.testing.assert_close(weighted,loss,atol=2e-6,rtol=2e-6)

    def test_new_parameters_receive_gradients(self):
        model=NoiseAdaptedNet(self.base,film=True,receptive=True)
        edm_loss(model,self.x,self.noise,.2,self.condition,self.condition).backward()
        self.assertGreater(float(model.film[-1].weight.grad.abs().sum()),0)
        self.assertGreater(float(model.receptive[-1].weight.grad.abs().sum()),0)

    def test_coefficient_identity_and_invalid_sigma(self):
        s=torch.tensor([.05,.2,1.,5.,20.])
        skip,out,scale=edm_coefficients(s)
        torch.testing.assert_close(skip,scale**2)
        torch.testing.assert_close(out,s*scale)
        model=NoiseAdaptedNet(self.base)
        with self.assertRaises(ValueError):
            model.denoise(self.x,0.,self.condition,self.condition)


if __name__=='__main__':
    unittest.main()
