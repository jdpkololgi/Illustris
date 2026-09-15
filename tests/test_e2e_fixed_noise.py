import math
import unittest
import torch
from torch import nn

from workflows.sbi.e2e_wide_models import ConditionalFieldNet
from workflows.sbi.e2e_fixed_noise_models import ResidualAdapter, HighResolutionNet, LinearReference, fixed_loss
from workflows.sbi.e2e_fine_learning_test import prediction_loss


class Zero(nn.Module):
    def forward(self,state,time,condition,wide_condition=None):
        return state*0


class FixedNoiseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(42)
        self.y=torch.randn(1,1,8,8,8); self.e=torch.randn_like(self.y)
        self.c=torch.randn(1,2,8,8,8)

    def test_residual_skip_and_fixed_loss_identity(self):
        net=ConditionalFieldNet(2,base_channels=2,levels=2,wide_condition_channels=2)
        model=ResidualAdapter(net)
        for s in (.05,.2):
            a=1/math.sqrt(1+s*s); b=s*a; x=a*self.y+b*self.e
            t=self.y.new_tensor([2*math.atan(s)/math.pi])
            correction=net(x,t,self.c,self.c)
            torch.testing.assert_close(a*x-b*model(x,t,self.c,self.c),x/a+correction)
            loss=fixed_loss(model,self.y,self.e,s,self.c,self.c,True)
            torch.testing.assert_close(loss,((correction/s+self.e)**2).mean())
            ref=prediction_loss(net,self.y,self.e,float(t),self.c,self.c,'diffusion')
            torch.testing.assert_close(fixed_loss(net,self.y,self.e,s,self.c,self.c),ref)

    def test_identity_references_and_highres_initialization(self):
        base=ConditionalFieldNet(2,base_channels=2,levels=2,wide_condition_channels=2)
        net=HighResolutionNet(base,width=4,blocks=2)
        for model in (ResidualAdapter(Zero()),ResidualAdapter(net),LinearReference(n=8,identity=True)):
            s=.2; a=1/math.sqrt(1+s*s); b=s*a; t=self.y.new_tensor([2*math.atan(s)/math.pi])
            torch.testing.assert_close(a*self.y-b*model(self.y,t,self.c,self.c),self.y/a)
        fixed_loss(ResidualAdapter(net),self.y,self.e,.2,self.c,self.c,True).backward()
        self.assertGreater(float(net.output.weight.grad.abs().sum()),0)

    def test_nonperiodic_reference_preserves_constant_and_attenuates_noise(self):
        ref=LinearReference(n=8,cell=3.383)
        torch.testing.assert_close(ref.clean(torch.ones_like(self.y)),torch.ones_like(self.y))
        self.assertLess(float(ref.clean(self.e).square().mean()),float(self.e.square().mean()))
        with self.assertRaises(ValueError):
            ref.clean(torch.ones(1,1,4,4,4))


if __name__=='__main__':
    unittest.main()
