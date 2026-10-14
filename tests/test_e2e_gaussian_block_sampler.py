"""Independent conditional identities, invariance and sampler safeguards."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workflows/sbi'))
import e2e_gaussian_block_sampler as block


class BlockSamplerTests(unittest.TestCase):
    def setUp(self):
        self.cov = np.array([[2.,.8,.3],[.8,1.5,.4],[.3,.4,1.]])
        self.mean = np.array([1.,-2.,.5]); self.precision = np.linalg.inv(self.cov)

    def test_precision_conditional_matches_covariance_identity(self):
        update = block.ConditionalUpdate.make(self.precision,self.mean,np.eye(3)[:,:2])
        expected = self.cov[:2,:2]-np.outer(self.cov[:2,2],self.cov[2,:2])/self.cov[2,2]
        np.testing.assert_allclose(update.covariance,expected,atol=1e-14)
        state = np.array([[12.,-9.,4.]])
        got = update.step(state,noise=np.zeros((1,2)))
        mean = self.mean[:2]+self.cov[:2,2]/self.cov[2,2]*(4.-self.mean[2])
        np.testing.assert_allclose(got[0,:2],mean,atol=1e-14)
        self.assertEqual(got[0,2],4.)

    def test_full_update_is_independent_exact_posterior_draw(self):
        update = block.ConditionalUpdate.make(self.precision,self.mean,np.eye(3))
        matrix,offset,cov = block.transition([update])
        np.testing.assert_allclose(matrix,0,atol=1e-14)
        np.testing.assert_allclose(offset,self.mean,atol=1e-14)
        np.testing.assert_allclose(cov,self.cov,atol=1e-14)

    def test_composition_matches_independent_dense_transition(self):
        updates = [block.ConditionalUpdate.make(self.precision,self.mean,np.eye(3)[:,[i]]) for i in (0,1,2)]
        matrix,offset,cov = np.eye(3),np.zeros(3),np.zeros((3,3))
        for update in updates:
            a = np.eye(3)-update.basis@update.gain
            b = update.basis@update.gain@self.mean
            matrix,offset,cov = a@matrix,a@offset+b,a@cov@a.T+update.basis@update.covariance@update.basis.T
        got = block.transition(updates)
        for a,b in zip(got,(matrix,offset,cov)): np.testing.assert_allclose(a,b,atol=1e-14)
        inv = block.invariance(got,self.mean,self.cov)
        self.assertLess(inv['covariance_relative'],1e-13)
        curve = block.convergence(got,self.mean,self.cov,np.eye(3))
        self.assertTrue(curve[-1]['passed'])

    def test_actual_noise_free_sweep_matches_affine_map(self):
        updates = [block.ConditionalUpdate.make(self.precision,self.mean,np.eye(3)[:,[i]]) for i in (2,1,0)]
        matrix,offset,_ = block.transition(updates)
        state = np.arange(15,dtype=float).reshape(5,3); actual = state.copy()
        for update in updates: actual = update.step(actual,noise=np.zeros((5,1)))
        np.testing.assert_allclose(actual,state@matrix.T+offset,atol=1e-14)

    def test_independent_marginal_reset_fails_correlated_target(self):
        wrong = (np.zeros((3,3)),self.mean,np.diag(np.diag(self.cov)))
        self.assertGreater(block.invariance(wrong,self.mean,self.cov)['covariance_relative'],.1)

    def test_subspace_matches_actual_projector_and_decoder_mass(self):
        import torch
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        from workflows.sbi.e2e_vdm_context_models import project,decode_density,block_mean
        basis,_ = block.constrained_basis(4)
        rng = np.random.default_rng(99)
        field = torch.as_tensor(rng.normal(size=(2,1,4,4,4)),dtype=torch.float64)
        expected = (field.numpy().reshape(2,64)@basis@basis.T).reshape(2,1,4,4,4)
        np.testing.assert_allclose(project(field).numpy(),expected,atol=1e-13)
        coarse = torch.ones((2,1,1,1,1),dtype=torch.float64)*1.7
        rho = decode_density(coarse,project(field))
        np.testing.assert_allclose(block_mean(rho).numpy(),coarse.numpy(),atol=1e-13)

    def test_global_subspace_update_preserves_complement(self):
        basis = np.array([[1.],[1.],[1.]])/np.sqrt(3)
        update = block.ConditionalUpdate.make(self.precision,self.mean,basis)
        state = np.arange(6,dtype=float).reshape(2,3)
        actual = update.step(state,np.random.default_rng(14))
        complement = np.eye(3)-basis@basis.T
        np.testing.assert_allclose(actual@complement,state@complement,atol=1e-14)
        self.assertLess(block.invariance(block.transition([update]),self.mean,self.cov)['covariance_relative'],1e-13)


if __name__ == '__main__':
    unittest.main()
