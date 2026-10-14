from pathlib import Path
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workflows/sbi'))
import e2e_exterior_cfm as cfm
import e2e_restricted_context as context
import e2e_gaussian_block_sampler as block


class ExteriorCFMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.data = cfm.GaussianTrainingData()
        cls.case = cls.data.heldout(1)[0]

    def model(self):
        torch.manual_seed(82)
        return cfm.TimeAffineCFM(8,8,self.data.context_dimensions,hidden=16)

    def test_context_contains_no_updated_block(self):
        for transform,indices in zip(self.data.transform,self.data.patches):
            np.testing.assert_allclose(transform[:,indices],0,atol=1e-14)

    def test_prior_joint_conditional_matches_posterior_precision(self):
        endpoint = (np.linalg.cholesky(self.data.exact_covariance),
                    np.concatenate([self.data.exact_response,np.zeros((8,8,1))],axis=-1))
        updates = cfm.kernels(self.data,endpoint,self.case)
        for u in updates:
            exact = context.conditional(self.case,u.indices,'full')
            np.testing.assert_allclose(u.gain,exact.gain,atol=2e-11)
            np.testing.assert_allclose(u.offset,exact.offset,atol=2e-11)
            np.testing.assert_allclose(u.covariance,exact.covariance,atol=2e-11)

    def test_exact_conditional_flow_converges(self):
        endpoint = cfm.heun_affine(cfm.exact_coefficients(self.data,512),8)
        rows = cfm.conditional_metrics(self.data,endpoint)
        self.assertLess(max(r['worst_relative_covariance_error'] for r in rows),.001)
        updates = cfm.kernels(self.data,endpoint,self.case)
        self.assertLess(block.invariance(block.transition(updates),self.case['mu'],self.case['sigma'])['covariance_relative'],.001)

    def test_actual_model_matches_affine_ode(self):
        model = self.model()
        with torch.no_grad():
            model.network[-1].weight.normal_(std=.01)
            model.network[-1].bias.normal_(std=.01)
        self.assertTrue(cfm.actual_model_check(model,self.data,32)['passed'])

    def test_training_uses_sampled_cfm_pairs_and_changes_parameters(self):
        model = self.model(); old = model.network[-1].weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(),lr=.001)
        generator = torch.Generator().manual_seed(81)
        loss = cfm.train_step(model,optimizer,self.data,generator,batch=32,time_groups=4)
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(torch.equal(old,model.network[-1].weight))

    def test_zero_velocity_is_independent_noise_negative_control(self):
        endpoint = cfm.heun_affine(cfm.learned_coefficients(self.model(),32),8)
        np.testing.assert_allclose(endpoint[0],np.broadcast_to(np.eye(8),(8,8,8)))
        np.testing.assert_array_equal(endpoint[1],0)
        self.assertFalse(all(r['component_accuracy_passed'] for r in cfm.conditional_metrics(self.data,endpoint)))


if __name__ == '__main__':
    unittest.main()
