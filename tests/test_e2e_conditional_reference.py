import copy
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from workflows.sbi.e2e_conditional_reference_math import (
    prior, posterior, problem, probes, oracle_moments, metrics, null_thresholds, qualify,
)
from workflows.sbi.e2e_conditional_reference import ExactModel, sample_cfm, train_step
from workflows.sbi.e2e_direct_vdm import ConditionalVDM, sample


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.sigma,_,_,_,cls.cases,_,cls.radius=problem(8,2)

    def test_prior(self):
        self.assertTrue(np.allclose(np.diag(self.sigma),1))
        self.assertGreater(np.linalg.eigvalsh(self.sigma).min(),0)

    def test_no_observations_returns_prior(self):
        case=posterior(self.sigma,np.zeros(512),np.ones(512),np.zeros(512))
        np.testing.assert_allclose(case["sigma"],self.sigma,atol=1e-12)
        np.testing.assert_equal(case["mu"],np.zeros(512))

    def test_observation_precision_reduces_covariance(self):
        self.assertGreater(np.linalg.eigvalsh(self.sigma-self.cases[0]["sigma"]).min(),-1e-10)

    def test_probes(self):
        q=probes(8)
        np.testing.assert_allclose(q.T@q,np.eye(16),atol=1e-12)

    def test_oracle_refines(self):
        for objective in ("vdm","cfm"):
            coarse=oracle_moments(self.cases[0],objective,32)
            fine=oracle_moments(self.cases[0],objective,1024)
            self.assertLess(fine["covariance_relative"],coarse["covariance_relative"])
            self.assertLess(fine["covariance_relative"],.05)

    def test_sampler_moments_match_independent_propagation(self):
        # Small eight-variable diagonal problem; inexpensive statistical check.
        case=dict(mu=np.linspace(-.5,.5,8),values=np.linspace(.2,.8,8),vectors=np.eye(8))
        count=8192
        for objective in ("vdm","cfm"):
            model=ExactModel(case,objective)
            condition=torch.zeros(count,3,2,2,2,dtype=torch.float64)
            generator=torch.Generator().manual_seed(103)
            fn=sample if objective=="vdm" else sample_cfm
            draws=fn(model,condition,64,generator).flatten(1).numpy()
            expected=oracle_moments(case,objective,64)
            np.testing.assert_allclose(draws.mean(0),expected["mean"],atol=.035)
            np.testing.assert_allclose(draws.var(0),expected["variance"],rtol=.07)

    def test_metrics_detect_broken_covariance(self):
        case=self.cases[0]
        q=probes(8)
        rng=np.random.default_rng(41)
        epsilon=rng.normal(size=(1024,512))
        exact=metrics(case["mu"]+epsilon@case["chol"].T,case,q,self.radius)
        independent=metrics(case["mu"]+epsilon*np.sqrt(np.diag(case["sigma"])),case,q,self.radius)
        self.assertLess(exact["covariance_relative"],independent["covariance_relative"])
        self.assertGreater(exact["octant_coverage"],independent["octant_coverage"])

    def test_training_replay(self):
        for objective in ("vdm","cfm"):
            torch.manual_seed(82)
            model=ConditionalVDM(3,8,1,False)
            optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
            rng=torch.Generator().manual_seed(801)
            x=torch.randn(2,1,4,4,4,generator=rng)
            c=torch.randn(2,3,4,4,4,generator=rng)
            train_step(model,optimizer,x,c,objective,rng)
            state=copy.deepcopy(model.state_dict())
            opt=copy.deepcopy(optimizer.state_dict())
            rs=rng.get_state()
            expected=train_step(model,optimizer,x,c,objective,rng)
            weights=copy.deepcopy(model.state_dict())
            model.load_state_dict(state); optimizer.load_state_dict(opt); rng.set_state(rs)
            actual=train_step(model,optimizer,x,c,objective,rng)
            self.assertEqual(expected,actual)
            for key in weights:
                torch.testing.assert_close(weights[key],model.state_dict()[key],rtol=0,atol=0)

    def test_condition_contains_no_truth(self):
        # Same observed vector implies same posterior even if unobserved y changes.
        case=self.cases[0]
        y=case["y"].copy(); y[case["mask"]==0]=1000
        other=posterior(self.sigma,case["mask"],case["std"],y)
        np.testing.assert_allclose(other["mu"],case["mu"])

    def test_config_bounds(self):
        root=Path(__file__).resolve().parents[1]
        c=json.loads((root/"configs/e2e_conditional_reference_v1.json").read_text())
        self.assertLessEqual(c["deadline_seconds"],6600)
        self.assertLessEqual(c["scratch_cap_gib"],20)
        self.assertEqual(len(c["seeds"]),2)
        self.assertTrue(all(n%2==0 for n in c["nfe"]))


if __name__=="__main__":
    unittest.main()
