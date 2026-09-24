import unittest
import numpy as np
from workflows.sbi.e2e_classical_hybrid import wiener_setup,constrained_draws,hybrid_setup,hybrid,low_basis
from workflows.sbi.e2e_conditional_reference_math import posterior,prior


class ClassicalHybridTests(unittest.TestCase):
    def test_constrained_covariance(self):
        p=np.array([[2.,.8],[.8,1.]])
        c=posterior(p,np.array([1.,0.]),np.ones(2)*.4,np.array([.5,0.]))
        setup=wiener_setup(p,c);o,n,b=setup
        a=np.eye(2);a[:,o]-=b
        np.testing.assert_allclose(a@p@a.T+(b*n)@(b*n).T,c['sigma'],atol=1e-12)
        x=constrained_draws(np.linalg.cholesky(p),c,setup,np.random.default_rng(1),10000)
        np.testing.assert_allclose(x.mean(0),c['mu'],atol=.025)
    def test_schur_restores_joint(self):
        cov=np.array([[2.,.8],[.8,1.]])
        c=dict(mu=np.zeros(2),sigma=cov);mu,a,l,_=hybrid_setup(c,np.eye(2),1)
        rebuilt=np.block([[a@cov[1:,1:]@a.T+l@l.T,a@cov[1:,1:]],[cov[1:,1:]@a.T,cov[1:,1:]]])
        np.testing.assert_allclose(rebuilt,cov,atol=1e-12)
    def test_high_modes_unchanged(self):
        c=dict(mu=np.zeros(2),sigma=np.array([[2.,.8],[.8,1.]]))
        x=np.random.default_rng(2).normal(size=(20,2))
        y=hybrid(x,np.eye(2),1,hybrid_setup(c,np.eye(2),1),np.random.default_rng(3))
        np.testing.assert_array_equal(x[:,1],y[:,1])
    def test_basis(self):
        _,_,r=prior(4);b,k=low_basis(r)
        self.assertEqual(k,7);np.testing.assert_allclose(b.T@b,np.eye(64),atol=1e-12)


if __name__=='__main__':unittest.main()
