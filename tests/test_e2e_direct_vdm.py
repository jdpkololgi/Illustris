import math
import unittest
import numpy as np
import torch
from workflows.sbi.e2e_direct_vdm import ConditionalVDM, LinearSchedule, vlb, sample
from workflows.sbi.e2e_direct_experiment import observation, prepare, metrics
from workflows.sbi.e2e_wide_denoising_audit import Bands


class DirectVDMTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(73)

    def test_coefficients_and_schedule(self):
        s=LinearSchedule();g=s(torch.tensor([0.,.5,1.]));a,b=s.coefficients(g)
        torch.testing.assert_close(a.square()+b.square(),torch.ones(3))
        self.assertTrue(torch.all(g[1:]>g[:-1]))
        self.assertFalse(LinearSchedule(learned=False).low.requires_grad)

    def test_shape_skip_gradient(self):
        model=ConditionalVDM(condition_channels=2,base=8,levels=2)
        x=torch.randn(2,1,8,8,8);c=torch.randn(2,2,8,8,8)
        torch.testing.assert_close(model(x,torch.zeros(2),c),x)
        loss,terms=vlb(model,x,c,torch.Generator().manual_seed(2));loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(model.schedule.low.grad))
        self.assertGreater(float(model.output.weight.grad.norm()),0)
        self.assertEqual(set(terms),{'diffusion','decoder','prior'})

    def test_fixed_learned_initial_loss_parity(self):
        torch.manual_seed(9);a=ConditionalVDM(2,8,2,True)
        torch.manual_seed(9);b=ConditionalVDM(2,8,2,False)
        x=torch.randn(2,1,8,8,8);c=torch.randn(2,2,8,8,8)
        la,_=vlb(a,x,c,torch.Generator().manual_seed(4));lb,_=vlb(b,x,c,torch.Generator().manual_seed(4))
        torch.testing.assert_close(la,lb,rtol=0,atol=0)

    def test_analytic_decoder(self):
        model=ConditionalVDM(2,8,2);x=torch.zeros(2,1,8,8,8);c=torch.zeros(2,2,8,8,8)
        _,terms=vlb(model,x,c,torch.Generator().manual_seed(3))
        expected=(.5*math.log(2*math.pi*1e-6)+.5*math.exp(-13.3)/1e-6)/math.log(2)
        self.assertAlmostEqual(float(terms['decoder']),expected,places=5)

    def test_ancestral_gaussian_oracle(self):
        # Finite-step DDPM variance is NOT exact Gaussian posterior variance.
        # Compare to the analytic discrete recurrence, not a false unit-variance
        # assertion that would blame the network for sampler discretization.
        class Oracle(torch.nn.Module):
            def __init__(self):
                super().__init__();self.schedule=LinearSchedule(learned=False)
            def forward(self,z,g,c):
                return self.schedule.coefficients(g)[1][:,None,None,None,None]*z
        model=Oracle();model.train();c=torch.zeros(128,1,8,8,8)
        out=sample(model,c,16,torch.Generator().manual_seed(11))
        variance=1.
        for i in range(16):
            gt=13.3-26.6*i/16;gs=13.3-26.6*(i+1)/16
            at2=1/(1+math.exp(gt));ass2=1/(1+math.exp(gs))
            st2=1-at2;ss2=1-ass2;change=-math.expm1(gs-gt)
            multiplier=math.sqrt(ass2/at2)*(1-change*st2)
            variance=multiplier**2*variance+ss2*change
        self.assertTrue(model.training)
        self.assertLess(abs(float(out.mean())),.015)
        self.assertLess(abs(float(out.var())-variance),.02)
        torch.testing.assert_close(out,sample(model,c,16,torch.Generator().manual_seed(11)),rtol=0,atol=0)

    def test_observation_excludes_coarse_truth(self):
        item=dict(condition=torch.randn(1,13,8,8,8),wide=torch.randn(1,12,8,8,8))
        before=observation(item);item['condition'][:,-1]=float('nan')
        torch.testing.assert_close(observation(item),before,rtol=0,atol=0)
        self.assertEqual(before.shape,(1,24,4,4,4))

    def test_train_only_normalization(self):
        train=[dict(anchor_id=str(i),phase='ph000' if i<5 else 'ph002') for i in range(10)]
        extra=dict(anchor_id='test',phase='ph003')
        prepared=dict(selection=dict(train=train,transfer=[extra]),original_normalization=dict(targets=dict(
            fine=dict(mean=0,std=1),coarse=dict(mean=0,std=1))))
        items={r['anchor_id']:dict(target=torch.rand(1,1,8,8,8),condition=torch.zeros(1,13,8,8,8),
                                 wide=torch.zeros(1,12,8,8,8),phase=r['phase']) for r in train+[extra]}
        spec=dict(train_phases=['ph000','ph002'],development_phase='ph003')
        _,a=prepare(prepared,items,spec);items['test']['target']+=200;items['test']['condition'][:,:12]+=400
        _,b=prepare(prepared,items,spec);self.assertEqual(a,b)

    def test_physical_metric_identity(self):
        x=np.random.default_rng(7).normal(size=(8,8,8))
        m=metrics(x,x,Bands(8,1,[0,1,2,4,np.inf]));self.assertEqual(m['rmse'],0)
        np.testing.assert_allclose(m['spectrum']['power_ratio'],1)
        self.assertEqual(m['regional_density'],m['truth_regional_density'])


if __name__=='__main__':unittest.main()
