import unittest
import numpy as np
import torch
from workflows.sbi.e2e_conditional_reference import Data, ExactModel
from workflows.sbi.e2e_reference_target_controls import Teacher, LearnedAffine, bridge, learning_rate, items, step
from workflows.sbi.e2e_conditional_reference import train_step
from workflows.sbi.e2e_direct_vdm import ConditionalVDM
import copy
from types import SimpleNamespace
from workflows.sbi.e2e_conditional_reference_math import problem


class TargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.data = Data(dict(grid=8,cases=4,draws=64,null_repeats=8), torch.device('cpu'))
        cls.teacher = Teacher(cls.data)

    def test_teacher_matches_independent_fixed_oracle(self):
        for case in range(4):
            c = self.data.conditions[case].expand(5,-1,-1,-1,-1)
            z = torch.randn(5,1,8,8,8)
            t = torch.tensor([0.,.1,.5,.9,1.])
            oracle = ExactModel(self.data.cases[case], 'cfm')
            expected = oracle(z.double(), oracle.schedule(t.double()), c.double())
            torch.testing.assert_close(self.teacher(z,t,c).double(), expected, atol=2e-5,rtol=2e-5)

    def test_affine_is_capable_but_not_initialized_at_truth(self):
        model = LearnedAffine(self.teacher)
        self.assertEqual(float(model.mean_map.abs().max()),0.)
        self.assertTrue(torch.equal(model.log_values,torch.zeros_like(model.log_values)))
        with torch.no_grad():
            model.mean_map.copy_(self.teacher.gain)
            model.log_values.copy_(self.teacher.values.log())
        c = torch.cat(self.data.conditions)
        z = torch.randn(4,1,8,8,8)
        t = torch.tensor([0.,.2,.7,1.])
        torch.testing.assert_close(model(z,model.schedule(t),c), self.teacher(z,t,c),atol=2e-5,rtol=2e-5)

    def test_affine_gradients_and_bridge_pairing(self):
        model = LearnedAffine(self.teacher)
        gen = torch.Generator().manual_seed(17)
        x,c = self.data.batch(32,None,gen)
        state=gen.get_state()
        z,t,v = bridge(x,gen)
        gen.set_state(state)
        zz,tt,vv=bridge(x,gen)
        self.assertTrue(torch.equal(z,zz) and torch.equal(t,tt) and torch.equal(v,vv))
        (model(z,model.schedule(t),c)-self.teacher(z,t,c)).square().mean().backward()
        self.assertGreater(float(model.mean_map.grad.abs().sum()),0.)
        self.assertGreater(float(model.log_values.grad.abs().sum()),0.)

    def test_assignment_and_schedule(self):
        names=[i['name'] for i in items()]
        self.assertEqual(len(names),30)
        self.assertEqual(len(set(names)),30)
        self.assertAlmostEqual(learning_rate('decay',0,16384),3e-4)
        self.assertAlmostEqual(learning_rate('decay',16384,16384),3e-6)
        self.assertEqual(learning_rate('teacher',16384,16384),3e-4)

    def test_stochastic_step_is_unchanged_from_parent(self):
        torch.manual_seed(918)
        model=ConditionalVDM(3,8,2,False)
        other=copy.deepcopy(model)
        a=torch.optim.Adam(model.parameters(),lr=3e-4)
        b=torch.optim.Adam(other.parameters(),lr=3e-4)
        ga=torch.Generator().manual_seed(145)
        gb=torch.Generator().manual_seed(145)
        x,c=self.data.batch(32,None,torch.Generator().manual_seed(911))
        first=train_step(model,a,x,c,'cfm',ga)
        second=step(other,b,self.teacher,x,c,gb,False)
        self.assertEqual(first,second)
        self.assertTrue(torch.equal(ga.get_state(),gb.get_state()))
        for p,q in zip(model.parameters(),other.parameters()):
            torch.testing.assert_close(p,q,rtol=0,atol=0)

    def test_teacher_against_joint_gaussian_conditioning(self):
        # Independent linear solve, not the eigenbasis/oracle implementation.
        _,chol,mask,_,cases,_,_=problem(4,2)
        data=SimpleNamespace(chol=torch.tensor(chol,dtype=torch.float32),
            mask=torch.tensor(mask,dtype=torch.float32),cases=cases,
            conditions=[torch.tensor(np.stack([c['y'],c['mask'],c['std']]),dtype=torch.float32)
                        .reshape(1,3,4,4,4) for c in cases])
        teacher=Teacher(data)
        rng=np.random.default_rng(412)
        for index,c in enumerate(data.cases):
            z=rng.normal(size=64)
            for t in [0.,.2,.8,1.]:
                cov_z=t*t*c['sigma']+(1-t)**2*np.eye(64)
                cov_vz=t*c['sigma']-(1-t)*np.eye(64)
                expected=c['mu']+cov_vz@np.linalg.solve(cov_z,z-t*c['mu'])
                actual=teacher(torch.tensor(z,dtype=torch.float32).reshape(1,1,4,4,4),
                               torch.tensor([t]),data.conditions[index]).flatten().numpy()
                np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-5)


if __name__=='__main__':
    unittest.main()
