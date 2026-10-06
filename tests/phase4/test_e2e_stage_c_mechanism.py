import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'workflows/sbi'))
import e2e_stage_c_mechanism_metrics as m


class MechanismTests(unittest.TestCase):
    def test_accounting_direct(self):
        rng=np.random.default_rng(1);x=rng.normal(size=(7,32,3));y=rng.normal(size=(7,3))
        q=m.moments(x,y);s=sum(np.cov(v,rowvar=False) for v in x)/7
        e=x.mean(1)-y;b=e.mean(0)
        np.testing.assert_allclose(q['predictive_covariance'],s)
        np.testing.assert_allclose(q['adjusted_error_second_moment'],sum(np.outer(v,v) for v in e)/7-s/32)
        np.testing.assert_allclose(np.asarray(q['centered_adjusted_error_second_moment'])+np.outer(b,b),q['adjusted_error_second_moment'])

    def test_negative_adjustments_retained(self):
        x=np.tile(np.array([-1.,1.])[None,:,None],(4,1,1));q=m.moments(x,np.zeros((4,1)))
        self.assertLess(q['adjusted_mse'][0],0)

    def test_gaussian_bias_and_spread(self):
        rng=np.random.default_rng(99);x=rng.normal(size=(6000,32,2));y=rng.normal(size=(6000,2))
        q=m.moments(x,y);shift=m.moments(x+1,y);narrow=m.moments(x*.5,y)
        self.assertAlmostEqual(q['trace_adjusted_mse_over_variance'],1,delta=.06)
        self.assertGreater(shift['bias_squared_fraction_of_total_mse'],.4)
        self.assertAlmostEqual(shift['trace_centered_adjusted_mse_over_variance'],q['trace_centered_adjusted_mse_over_variance'])
        self.assertGreater(narrow['trace_adjusted_mse_over_variance'],3.7)

    def test_dependence_only_failure(self):
        rng=np.random.default_rng(2);x=rng.normal(size=(6000,32,2));y=rng.normal(size=(6000,2));y[:,1]=.9*y[:,0]+np.sqrt(.19)*y[:,1]
        q=m.moments(x,y)
        np.testing.assert_allclose(q['adjusted_mse_over_variance'],1,atol=.07)
        self.assertGreater(q['generalized_eigenvalues'][-1],1.7)
        self.assertLess(q['generalized_eigenvalues'][0],.2)

    def test_affine_not_clipped(self):
        x=np.array([[[[1.,0,0,0],[0,1.,0,0]],[[0,1.,0,0],[1.,0,0,0]]]])
        z,q=m.affine_probe(x,np.zeros(4),2)
        self.assertEqual(q['out_of_simplex_rate'],1)
        self.assertEqual(z.min(),-.5)
        np.testing.assert_allclose(z.mean(1),x.mean(1))

    def test_affine_identity_preserves_discrete_ties(self):
        rng=np.random.default_rng(51);x=rng.multinomial(13,[.1,.2,.3,.4],size=(4,32,2))/13
        z,_=m.affine_probe(x,np.zeros(4),1)
        np.testing.assert_array_equal(x,z)

    def test_cyclic_marginals_and_zero_average_cross_covariance(self):
        rng=np.random.default_rng(3);x=rng.dirichlet([1,2,3,4],size=(32,2));cross=[]
        for offset,z in m.cyclic_pairings(x):
            for side in (0,1):np.testing.assert_array_equal(np.sort(z[:,side],axis=0),np.sort(x[:,side],axis=0))
            cross.append(np.cov(z.reshape(32,8),rowvar=False)[:4,4:])
        np.testing.assert_allclose(np.mean(cross,0),0,atol=1e-16)

    def test_sum_difference_variance_identity(self):
        rng=np.random.default_rng(4);x=rng.normal(size=(9,32,2,3));y=rng.normal(size=(9,2,3));q=m.cross_core(x,y)
        vl=np.array(q['left']['predictive_variance']);vr=np.array(q['right']['predictive_variance']);c=np.array(q['same_class_predictive_cross_covariance'])
        np.testing.assert_allclose(q['mean']['predictive_variance'],(vl+vr+2*c)/4)
        np.testing.assert_allclose(q['difference']['predictive_variance'],vl+vr-2*c)

    def test_guards_and_small_groups(self):
        with self.assertRaises(ValueError):m.moments(np.zeros((3,1,2)),np.zeros((3,2)))
        with self.assertRaises(ValueError):m.affine_probe(np.zeros((2,32,2,4)),np.ones(4),1)
        q=m.moments(np.zeros((2,32,3)),np.zeros((2,3)))
        self.assertEqual(q['matrix_status'],'unavailable_fewer_than_four_cases')
        self.assertIsNone(q['trace_adjusted_mse_over_variance'])


if __name__=='__main__':unittest.main()
