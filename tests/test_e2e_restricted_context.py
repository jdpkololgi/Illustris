from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.linalg import cho_factor,cho_solve

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workflows/sbi'))
import e2e_gaussian_block_sampler as block
import e2e_restricted_context as context


class RestrictedContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = context.problem(4,'long_range',1,8841)
        cls.indices = context.patches(4)[0]

    def test_full_covariance_conditional_equals_precision(self):
        c = self.case; ix = self.indices
        exterior = np.setdiff1d(np.arange(64),ix)
        cross = c['sigma'][np.ix_(ix,exterior)]
        gain = cho_solve(cho_factor(c['sigma'][np.ix_(exterior,exterior)]),cross.T).T
        update = context.conditional(c,ix,'full')
        np.testing.assert_allclose(update.gain[:,exterior],gain,atol=1e-12)
        np.testing.assert_allclose(update.covariance,c['sigma'][np.ix_(ix,ix)]-gain@cross.T,atol=1e-12)

    def test_restricted_conditional_matches_joint_summary_identity(self):
        c = context.problem(6,'spectral',0,8842); ix = context.patches(6)[0]
        u = context.context_basis(6,ix,'collar1_modes')
        update = context.conditional(c,ix,'collar1_modes')
        np.testing.assert_allclose(update.gain@c['sigma']@u,c['sigma'][ix]@u,atol=1e-11)
        np.testing.assert_allclose(update.covariance+update.gain@c['sigma']@update.gain.T,
                                   c['sigma'][np.ix_(ix,ix)],atol=1e-11)
        self.assertGreaterEqual(context.information_loss(update,context.conditional(c,ix,'full')),0)

    def test_no_exterior_is_posterior_marginal(self):
        u = context.conditional(self.case,self.indices,'none')
        np.testing.assert_array_equal(u.gain,0)
        np.testing.assert_allclose(u.offset,self.case['mu'][self.indices])
        np.testing.assert_allclose(u.covariance,self.case['sigma'][np.ix_(self.indices,self.indices)])

    def test_dense_composition_and_actual_step(self):
        c = self.case; updates = [context.conditional(c,i,'collar1') for i in context.patches(4)]
        m,b,q = block.transition(updates)
        mm,bb,qq = np.eye(64),np.zeros(64),np.zeros((64,64))
        states = np.arange(128,dtype=float).reshape(2,64)
        actual = states.copy()
        for u in updates:
            a,d,r = np.eye(64),np.zeros(64),np.zeros((64,64))
            a[u.indices] = u.gain; d[u.indices] = u.offset
            r[np.ix_(u.indices,u.indices)] = u.covariance
            mm,bb,qq = a@mm,a@bb+d,a@qq@a.T+r
            actual = u.step(actual,noise=np.zeros((2,len(u.indices))))
        for x,y in zip((m,b,q),(mm,bb,qq)):
            np.testing.assert_allclose(x,y,atol=1e-12)
        np.testing.assert_allclose(actual,states@m.T+b,atol=1e-11)

    def test_full_invariance_and_marginal_failure(self):
        c = self.case; probes,_names,_omitted = block.probe_bank(4,np.eye(64))
        for name,expected in [('full',True),('none',False)]:
            affine = block.transition([context.conditional(c,i,name) for i in context.patches(4)])
            result,_,_ = context.stationary(affine,c,probes)
            self.assertEqual(result['accuracy_passed'],expected)

    def test_independent_target_all_contexts(self):
        c = context.problem(4,'independent',0,8843)
        for name in context.CONTEXTS:
            affine = block.transition([context.conditional(c,i,name) for i in context.patches(4)])
            self.assertLess(block.invariance(affine,c['mu'],c['sigma'])['covariance_relative'],1e-12)

    def test_geometry_and_no_block_leakage(self):
        for n in (4,6,8):
            for shift in (0,1):
                patches = context.patches(n,shift=shift)
                for name in context.CONTEXTS:
                    u = context.context_basis(n,patches[0],name)
                    np.testing.assert_allclose(u[patches[0]],0,atol=1e-14)
        ix = context.patches(6)[0]
        periodic = context.context_basis(6,ix,'collar1',True)
        opened = context.context_basis(6,ix,'collar1',False)
        self.assertGreater(periodic.shape[1],opened.shape[1])


if __name__ == '__main__':
    unittest.main()
