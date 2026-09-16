import math
import unittest
import torch
from workflows.sbi.e2e_analytic_fields import (spectrum,filt,gaussian_mean,log_density_mean,
    GaussianVelocity,iid_lognormal_density_mean,min_snr_v_weight)
from workflows.sbi.e2e_loss_conflict import comparison,remove_conflicting_component,gradient_vectors
from workflows.sbi.e2e_oracle_solver import sample_vp_heun
from workflows.sbi.e2e_gradient_step_probe import assign_gradient


class AnalyticFieldsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.p=spectrum(4);self.x=torch.arange(64,dtype=torch.float64).reshape(1,1,4,4,4)/64

    def test_spectrum_variance_and_symmetry(self):
        self.assertAlmostEqual(float(self.p.mean()),.49)
        torch.testing.assert_close(filt(self.x,torch.ones_like(self.p)),self.x)

    def test_exact_zero_identity(self):
        torch.testing.assert_close(gaussian_mean(self.x,self.p,0),self.x)
        torch.testing.assert_close(log_density_mean(self.x,self.p,0),torch.expm1(self.x-.245))

    def test_white_prior_shrinkage(self):
        p=torch.full_like(self.p,2.)
        torch.testing.assert_close(gaussian_mean(self.x,p,.5),self.x*2/2.25)

    def test_v_oracle_conversion(self):
        t=torch.tensor([.2],dtype=torch.float64);a=math.cos(.2*math.pi/2);b=math.sin(.2*math.pi/2)
        mean=torch.full_like(self.x,.3);v=GaussianVelocity(self.p,'diffusion')(self.x,t,mean)
        expected=mean+filt(self.x-a*mean,a*self.p/(a*a*self.p+b*b))
        torch.testing.assert_close(a*self.x-b*v,expected)

    def test_log_mean_jensen_correction(self):
        q=.2;v=(self.p*q*q/(self.p+q*q)).mean()
        naive=torch.exp(gaussian_mean(self.x,self.p,q)-self.p.mean()/2)
        torch.testing.assert_close(log_density_mean(self.x,self.p,q)+1,naive*torch.exp(v/2))

    def test_quadrature_tolerance(self):
        for y in (-.9,0.,3.):
            for q in (.005,.2,1.):
                a=iid_lognormal_density_mean(y,q,tol=1e-9)
                b=iid_lognormal_density_mean(y,q,tol=1e-11)
                self.assertAlmostEqual(a,b,places=7);self.assertGreater(a,-1)

    def test_min_snr_v_chart(self):
        q=torch.tensor([.001,.01,.05,1.,20.],dtype=torch.float64);snr=1/q.square()
        torch.testing.assert_close(min_snr_v_weight(q),snr.clamp(max=5)/(snr+1))
        self.assertEqual(float(min_snr_v_weight(torch.tensor(0.))),0)

    def test_projection_not_identity_cheat(self):
        a=torch.tensor([1.,0.]);b=torch.tensor([-1.,2.]);c=remove_conflicting_component(a,b)
        torch.testing.assert_close(c,torch.tensor([0.,2.]));self.assertEqual(comparison(a,c)['dot'],0.)

    def test_gradient_metrics(self):
        x=torch.nn.Parameter(torch.tensor([1.,2.]));v=gradient_vectors({'a':x.square().sum(),'b':-x.sum()},[x])
        torch.testing.assert_close(v['a'],torch.tensor([2.,4.],dtype=torch.float64))
        self.assertLess(comparison(v['a'],v['b'])['cosine'],0)

    def test_gradient_assignment(self):
        a=torch.nn.Parameter(torch.zeros(2));b=torch.nn.Parameter(torch.zeros(1,2))
        assign_gradient([a,b],torch.tensor([1.,2.,3.,4.],dtype=torch.float64))
        torch.testing.assert_close(a.grad,torch.tensor([1.,2.]))
        torch.testing.assert_close(b.grad,torch.tensor([[3.,4.]]))

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):gaussian_mean(self.x,self.p,-1)
        with self.assertRaises(ValueError):iid_lognormal_density_mean(-2,.1)
        with self.assertRaises(ValueError):min_snr_v_weight(torch.tensor(-.1))

    def test_heun_exact_stationary_standard_gaussian(self):
        p=torch.ones_like(self.p);condition=torch.zeros_like(self.x);model=GaussianVelocity(p,'diffusion')
        gen=lambda:torch.Generator().manual_seed(7)
        expected=torch.randn(self.x.shape,dtype=self.x.dtype,generator=gen())
        actual=sample_vp_heun(model,condition,8,gen())
        torch.testing.assert_close(actual,expected,atol=1e-14,rtol=1e-14)
        self.assertTrue(model.training)

    def test_heun_refines_and_restores_mode(self):
        p=torch.full_like(self.p,.2);condition=torch.ones_like(self.x)*.1
        model=GaussianVelocity(p,'diffusion');model.eval();gen=lambda:torch.Generator().manual_seed(8)
        expected=condition+torch.randn(self.x.shape,dtype=self.x.dtype,generator=gen())*math.sqrt(.2)
        coarse=sample_vp_heun(model,condition,8,gen());fine=sample_vp_heun(model,condition,32,gen())
        self.assertLess(float((fine-expected).square().mean()),float((coarse-expected).square().mean())/20)
        self.assertFalse(model.training)

if __name__=='__main__':unittest.main()
