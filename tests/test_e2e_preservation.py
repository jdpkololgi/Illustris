import copy
import json
import math
import unittest
import torch
from torch import nn
from workflows.sbi.e2e_preservation_loss import objective, residual_terms, auxiliary_ratio
from workflows.sbi.e2e_preservation_experiment import spec, gate, response_gate
from workflows.sbi.e2e_multinoise_models import coefficients, loss_for
from workflows.sbi.e2e_diversity_norm import field_for


class Toy(nn.Module):
    tau=.05
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.tensor(.3,dtype=torch.float64))
    def forward(self,x,t,c,wide_condition=None,context_present=None):
        return self.weight*x+.01*c[:,:1]+t[:,None,None,None,None]


class PreservationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(52)
        self.y=torch.randn(1,1,4,4,4,dtype=torch.float64)
        self.eps=torch.randn_like(self.y);self.aux=torch.randn_like(self.y)
        self.c=torch.randn_like(self.y);self.w=torch.randn_like(self.y);self.m=Toy()

    def run_objective(self,**kwargs):
        args=dict(ratio=.01,fraction=.25,identity_weight=.1,response_weight=.1);args.update(kwargs)
        return objective(self.m,self.y,self.eps,.3,self.c,self.w,self.aux,**args)

    def test_zero_weights_match_original_loss_and_gradient(self):
        old,_,_=loss_for(self.m,self.y,self.eps,.3,self.c,self.w)
        new,_=self.run_objective(identity_weight=0.,response_weight=0.)
        self.assertTrue(torch.equal(old,new))
        a=torch.autograd.grad(old,self.m.weight)[0];b=torch.autograd.grad(new,self.m.weight)[0]
        self.assertTrue(torch.equal(a,b))

    def test_terms_equal_physical_clean_even_odd_errors(self):
        t=self.y.new_tensor([2*math.atan(.01)/math.pi]);a,b,_=coefficients(t)
        v0=self.m(a*self.y,t,self.c,self.w)
        vp=self.m(a*self.y+b*.25*self.aux,t,self.c,self.w)
        vm=self.m(a*self.y-b*.25*self.aux,t,self.c,self.w)
        terms=residual_terms(v0,vp,vm,self.y,self.aux,a,b,.25)
        d0=a*a*self.y-b*v0
        dp=a*(a*self.y+b*.25*self.aux)-b*vp
        dm=a*(a*self.y-b*.25*self.aux)-b*vm
        expected={'identity':((d0-self.y)/b).square().mean(),
                  'even_consistency':(((dp+dm)/2-d0)/b).square().mean(),
                  'odd_noise_response':(((dp-dm)/2)/b).square().mean()}
        for name in terms:torch.testing.assert_close(terms[name],expected[name],atol=1e-10,rtol=1e-8)

    def test_perfect_response_has_zero_auxiliary_terms(self):
        a,b,_=coefficients(self.y.new_tensor([.03]))
        v0=-b*self.y;vp=v0+a*.25*self.aux;vm=v0-a*.25*self.aux
        terms=residual_terms(v0,vp,vm,self.y,self.aux,a,b,.25)
        self.assertLess(sum(float(v) for v in terms.values()),1e-25)

    def test_identity_only_does_not_satisfy_noise_response(self):
        a,b,_=coefficients(self.y.new_tensor([.03]))
        v0=-b*self.y;vp=v0-b*b/a*.25*self.aux;vm=v0+b*b/a*.25*self.aux
        terms=residual_terms(v0,vp,vm,self.y,self.aux,a,b,.25)
        self.assertEqual(float(terms['identity']),0.)
        self.assertGreater(float(terms['odd_noise_response']),.01)

    def test_antithetic_sign_invariance(self):
        a,_=self.run_objective();self.aux=-self.aux;b,_=self.run_objective()
        torch.testing.assert_close(a,b)

    def test_auxiliary_gradients_not_detached(self):
        plain,_=self.run_objective(identity_weight=0.,response_weight=0.)
        augmented,_=self.run_objective()
        self.assertNotEqual(float(torch.autograd.grad(plain,self.m.weight)[0]),
                            float(torch.autograd.grad(augmented,self.m.weight)[0]))

    def test_low_noise_finite_and_zero_rejected(self):
        for ratio in (1e-5,.001,.05):
            loss,_=self.run_objective(ratio=ratio)
            self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(torch.autograd.grad(loss,self.m.weight)[0]))
        for ratio in (0.,-.001,.2):
            with self.assertRaises(ValueError):self.run_objective(ratio=ratio)

    def test_every_field_visits_every_auxiliary_bin(self):
        s=spec();seen=set()
        for u in range(24576,24576+180):
            q=auxiliary_ratio(u,91531,s);lo,hi=s['auxiliary_bins'][(u//3)%6]
            self.assertTrue(lo<=q<=hi);seen.add((field_for(u,15,list(range(15))),(u//3)%6))
        self.assertEqual(len(seen),90)

    def fixture(self):
        before=[];after=[];parent={}
        for ratio in (0.,.001,.01,.05,.2):
            a=dict(anchor_id='a',phase='ph000',kind='clean',ratio=ratio,rms=0. if ratio==0 else 1.,
                bias=0.,max_abs=0.,rms_over_injected=1.,metrics=dict(error_power=[1.]*4,gain=[1.]*4))
            b=copy.deepcopy(a);b['rms']*=.5;before.append(a);after.append(b)
            if ratio:
                for rep in range(2):
                    a=dict(anchor_id='a',phase='ph000',kind='noisy',ratio=ratio,rep=rep,physical_mse=1.,
                        metrics=dict(error_power=[1.]*4,gain=[1.]*4,noise_amplitude=[.1]*4))
                    b=copy.deepcopy(a);b['physical_mse']=.8;before.append(a);after.append(b)
                    parent['a',ratio,rep]=dict(metrics=dict(error_power=[10.]*4))
        return before,after,parent

    def test_joint_gate_rejects_identity_cheating(self):
        a,b,parent=self.fixture();self.assertTrue(gate(a,b,['a'],spec()['gate'],parent)['passed'])
        for row in b:
            if row['kind']=='clean':row['rms']=0.
            else:row['physical_mse']=1.2
        self.assertFalse(gate(a,b,['a'],spec()['gate'],parent)['passed'])

    def test_gate_requires_legacy_noise_removal(self):
        a,b,parent=self.fixture()
        for row in b:
            if row['kind']=='noisy':row['metrics']['noise_amplitude']=[1.]*4
        self.assertFalse(gate(a,b,['a'],spec()['gate'],parent)['passed'])

    def test_response_pairing_and_nonregression(self):
        a=[dict(anchor_id='a',phase='ph000',ratio=.01,fraction=.1,rep=i,
                noisy_mse=1.,even_drift_mse=.2,odd_remaining_mse=.5) for i in range(2)]
        b=copy.deepcopy(a)
        self.assertTrue(response_gate(a,b,['a'],spec()['gate'])['passed'])
        for row in b:row['noisy_mse']=1.2
        self.assertFalse(response_gate(a,b,['a'],spec()['gate'])['passed'])
        with self.assertRaises(ValueError):response_gate(a,b[:1],['a'],spec()['gate'])


if __name__=='__main__':unittest.main()
