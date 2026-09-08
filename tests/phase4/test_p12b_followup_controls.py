import unittest
import torch
from workflows.sbi.p12b_representation_followup import clip_groups
from workflows.sbi.p12b_representation_diagnostics import norm


class FollowupControlTests(unittest.TestCase):
    def setup_groups(self):
        a=torch.nn.Parameter(torch.ones(2));b=torch.nn.Parameter(torch.ones(2))
        optimizer=torch.optim.AdamW([{"params":[a]},{"params":[b]}])
        a.grad=torch.tensor([6.,8.]);b.grad=torch.tensor([0.,10.])
        return a,b,optimizer

    def test_separate_clipping(self):
        a,b,o=self.setup_groups()
        self.assertEqual(clip_groups(o,"separate",5),[10.,10.])
        self.assertAlmostEqual(norm([a]),5.,places=5)
        self.assertAlmostEqual(norm([b]),5.,places=5)

    def test_global_clipping(self):
        a,b,o=self.setup_groups()
        self.assertEqual(clip_groups(o,"global",5),[10.,10.])
        self.assertAlmostEqual(norm([a,b]),5.,places=5)
        self.assertAlmostEqual(norm([a]),5./2**.5,places=5)

    def test_single_group_control_identical(self):
        a=torch.nn.Parameter(torch.ones(2));b=torch.nn.Parameter(torch.ones(2))
        a.grad=torch.tensor([6.,8.]);b.grad=a.grad.clone()
        x=torch.optim.AdamW([a]);y=torch.optim.AdamW([b])
        clip_groups(x,"global",5);clip_groups(y,"separate",5)
        torch.testing.assert_close(a.grad,b.grad,atol=0,rtol=0)

    def test_unknown_policy_rejected(self):
        _,_,o=self.setup_groups()
        with self.assertRaises(ValueError):clip_groups(o,"other",5)


if __name__=="__main__":unittest.main()
