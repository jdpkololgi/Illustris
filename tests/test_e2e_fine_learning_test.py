import unittest
import numpy as np
import torch

from workflows.sbi.e2e_fine_learning_test import coefficients, prediction_loss, training_time, gates
from workflows.sbi.e2e_wide_models import flow_matching_loss, diffusion_loss


class FineLearningTest(unittest.TestCase):
    def test_loss_and_gradient_match_original(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(.3))
            def forward(self, state, time, condition, wide_condition=None):
                return self.weight*state + time.view(-1,1,1,1,1)*condition[:, :1]
        target = torch.ones((1,1,4,4,4))*.2
        condition = torch.ones_like(target)*.1
        for method, original in [('cfm',flow_matching_loss),('diffusion',diffusion_loss)]:
            model = Toy()
            generator = torch.Generator().manual_seed(10)
            noise = torch.randn(target.shape, generator=generator)
            t = float(torch.rand(1,generator=generator)[0])
            loss = prediction_loss(model,target,noise,t,condition,None,method)
            loss.backward(); gradient = model.weight.grad.clone()
            model.zero_grad()
            reference = original(model,target,condition,torch.Generator().manual_seed(10))
            reference.backward()
            torch.testing.assert_close(loss,reference)
            torch.testing.assert_close(gradient,model.weight.grad)

    def test_balanced_schedules_cross_every_anchor(self):
        ratios = [.05,.2,1.,5.,20.]
        for method in ('cfm','diffusion'):
            for arm, period, levels in [('balanced_noise',15,ratios),('near_clean',6,ratios[:2])]:
                for anchor in range(3):
                    seen = []
                    for step in range(anchor,period,3):
                        t=training_time(method,arm,step,4,ratios)
                        alpha,sigma=coefficients(method,t);seen.append(sigma/alpha)
                    np.testing.assert_allclose(seen,levels)
            self.assertEqual(training_time(method,'uniform_time',0,4,ratios),
                             training_time(method,'uniform_time',0,4,ratios))

    def test_gate_rejects_noise_removal_by_destroying_signal(self):
        cfg={'gate':dict(near_clean_ratios=[.05,.2],max_abs_noise_amplitude=.2,
                        max_error_vs_parent=.25,signal_gain_min=.9,signal_gain_max=1.1)}
        old=[];new=[]
        for group in ('fit','transfer'):
            for phase in ('ph000','ph002','ph003'):
                for ratio in (.05,.2):
                    for rep in range(2):
                        row=dict(group=group,phase=phase,ratio=ratio)
                        old.append(dict(**row,metrics=dict(error_power=[1]*4)))
                        new.append(dict(**row,metrics=dict(error_power=[.1]*4,noise_amplitude=[.1]*4,gain=[1]*4)))
        self.assertTrue(gates(old,new,cfg)['fit']['passed'])
        new[0]['metrics']['gain']=[0]*4
        self.assertFalse(gates(old,new,cfg)['fit']['passed'])
        self.assertTrue(gates(old,new,cfg)['transfer']['passed'])


if __name__=='__main__':
    unittest.main()
