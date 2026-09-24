import unittest
import numpy as np
from workflows.sbi.e2e_lognormal_reference import Posterior,laplace,sample


class LognormalTests(unittest.TestCase):
    def test_gradient_and_hessian(self):
        p=Posterior(np.array([[1.,.3],[.3,1.]]),np.array([.5,2.]),np.array([0,3]))
        z=np.array([.1,-.2]);eps=1e-5;eye=np.eye(2)
        grad=np.array([(p.value_grad(z+eps*d)[0]-p.value_grad(z-eps*d)[0])/(2*eps) for d in eye])
        h=np.stack([(p.value_grad(z+eps*d)[1]-p.value_grad(z-eps*d)[1])/(2*eps) for d in eye],axis=1)
        np.testing.assert_allclose(grad,p.value_grad(z)[1],rtol=1e-7)
        np.testing.assert_allclose(h,p.hessian(z),rtol=1e-7)
    def test_mask_and_prior(self):
        p=Posterior(np.eye(2),np.zeros(2),np.zeros(2));z=np.array([1.,2.])
        self.assertEqual(p.value_grad(z)[0],2.5);np.testing.assert_array_equal(p.value_grad(z)[1],z)
        with self.assertRaises(ValueError):Posterior(np.eye(2),np.zeros(2),np.ones(2))
    def test_map(self):
        p=Posterior(np.eye(2),np.ones(2),np.array([0,3]));m,f,d=laplace(p)
        self.assertLess(d['gradient_max'],1e-6)
        np.testing.assert_allclose(f@f.T,np.linalg.inv(p.hessian(m)),atol=1e-12)
    def test_gaussian_sampler(self):
        p=Posterior(np.eye(2),np.zeros(2),np.zeros(2));m,f,_=laplace(p)
        z,_=sample(p,m,f,11,draws=4096,warmup=512)
        np.testing.assert_allclose(z.mean((0,1)),0,atol=.06)
        np.testing.assert_allclose(z.var((0,1)),1,atol=.08)


if __name__=='__main__':unittest.main()
