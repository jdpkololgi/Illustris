import copy
import unittest
import numpy as np
import torch
from workflows.sbi.e2e_frozen_controls import zero_first_moment, apply_step, panel, relative_rms
from workflows.sbi.e2e_wide_continue import equal_state


class FrozenControlsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.p=torch.nn.Parameter(torch.tensor([1.,2.],dtype=torch.float64))
        self.opt=torch.optim.AdamW([self.p],lr=.01,weight_decay=.01,amsgrad=True)
        apply_step(self.opt,[self.p],torch.tensor([2.,-3.]),None)

    def test_first_moment_only(self):
        before=copy.deepcopy(self.opt.state_dict());weights=self.p.detach().clone()
        zero_first_moment(self.opt);after=self.opt.state_dict()
        for key in before['state'][0]:
            if key=='exp_avg':self.assertEqual(float(after['state'][0][key].abs().sum()),0.)
            else:self.assertTrue(equal_state(before['state'][0][key],after['state'][0][key]))
        self.assertEqual(before['param_groups'],after['param_groups'])
        torch.testing.assert_close(self.p,weights)

    def test_uninitialized_rejected(self):
        with self.assertRaises(ValueError):zero_first_moment(torch.optim.AdamW([self.p]))

    def test_clipping_switch(self):
        state=copy.deepcopy(self.opt.state_dict());origin=self.p.detach().clone()
        a=apply_step(self.opt,[self.p],torch.tensor([3.,4.]),1.)
        clipped=self.p.detach().clone()
        self.opt.load_state_dict(copy.deepcopy(state));self.p.data.copy_(origin)
        b=apply_step(self.opt,[self.p],torch.tensor([3.,4.]),None)
        self.assertAlmostEqual(a['clipping_scale'],.2,places=6)
        self.assertEqual(b['clipping_scale'],1.)
        self.assertFalse(torch.equal(clipped,self.p))

    def test_restore_and_repeat(self):
        state=copy.deepcopy(self.opt.state_dict());origin=self.p.detach().clone()
        apply_step(self.opt,[self.p],torch.tensor([3.,4.]),1.);first=self.p.detach().clone()
        self.opt.load_state_dict(copy.deepcopy(state));self.p.data.copy_(origin)
        apply_step(self.opt,[self.p],torch.tensor([3.,4.]),1.)
        self.assertTrue(torch.equal(first,self.p))

    def test_nonfinite_rejected(self):
        with self.assertRaises(RuntimeError):apply_step(self.opt,[self.p],torch.tensor([float('nan'),0.]),None)

    def test_phase_balanced_selection(self):
        rows=[dict(phase=p,anchor_id=f'{p}{i}') for p in ('a','b','c') for i in range(3)]
        actual=panel(dict(train=rows,transfer=rows),['a','b','c'])
        self.assertEqual([r['anchor_id'] for r in actual],['a0','b0','c0']*2)
        with self.assertRaises(ValueError):panel(dict(train=rows,transfer=[]),['a'])

    def test_relative_rms(self):
        self.assertAlmostEqual(relative_rms(np.ones(8)*1.1,np.ones(8)),.1)
        self.assertEqual(relative_rms(np.zeros(8),np.zeros(8)),0.)

if __name__=='__main__':unittest.main()
