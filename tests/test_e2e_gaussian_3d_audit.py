"""Post-run audit tests; no change to the frozen experiment."""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workflows/sbi'))
import e2e_gaussian_3d_cfm as g
import e2e_gaussian_3d_run as run


class MatchedAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_data_rng_is_independent_of_bridge_shape(self):
        data=g.JointGenerator(16,'cpu');pairs=[]
        for width in (8,16):
            data_rng=torch.Generator().manual_seed(83);bridge_rng=torch.Generator().manual_seed(84)
            arm=[]
            for _ in range(3):
                arm.append(data.batch(2,data_rng))
                torch.randn((2,1,width,width,width),generator=bridge_rng)
                torch.rand((),generator=bridge_rng)
            pairs.append(arm)
        for first,second in zip(*pairs):
            for a,b in zip(first,second):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_probe_gate_rejects_large_shift_and_never_promotes(self):
        samples=np.random.default_rng(22).normal(size=(10000,3))
        self.assertFalse(run.scores(samples,np.eye(3))['qualified'])
        self.assertEqual(run.scores(samples+2,np.eye(3))['accuracy_status'],'failed')
        self.assertEqual(run.scores(samples*.1,np.eye(3))['accuracy_status'],'failed')

    def test_exact_observation_contract(self):
        data=g.JointGenerator(8,'cpu');x,y=data.batch(4,torch.Generator().manual_seed(99))
        self.assertEqual(y.shape,(4,3,8,8,8))
        self.assertTrue(torch.isfinite(x).all())
        self.assertTrue(torch.equal(y[:,:1]*(1-y[:,1:2]),torch.zeros_like(x)))
        torch.testing.assert_close(y[:,1:2],data.mask.expand_as(x))
        torch.testing.assert_close(y[:,2:3],data.std.expand_as(x))


if __name__=='__main__':unittest.main()
