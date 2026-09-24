import unittest
import numpy as np
from workflows.sbi.e2e_stable_alpha import items,lr,choose
from workflows.sbi.e2e_reference_usecase import smooth,tidal,regions,distort,linear_summary


class StableAlphaTests(unittest.TestCase):
    def test_panel(self):
        p=items();self.assertEqual(len(p),16);self.assertEqual(len({i['name'] for i in p}),16)
        for rank in range(4):self.assertEqual({i['alpha'] for i in p[rank::4]},{.25,.3,.35,.4})
    def test_schedule(self):
        self.assertEqual(lr(65536),3e-4);self.assertAlmostEqual(lr(98304),3e-6)
    def test_incomplete_selection(self):
        with self.assertRaises(ValueError):choose([])
    def test_smoothing_identity_and_dc(self):
        x=np.random.default_rng(9).normal(size=(2,512));np.testing.assert_allclose(smooth(x,0),x,atol=1e-12)
        np.testing.assert_allclose(smooth(x,2).mean(1),x.mean(1),atol=1e-12)
    def test_tidal_trace(self):
        x=np.random.default_rng(2).normal(size=(2,512))
        np.testing.assert_allclose(np.trace(tidal(x,1),axis1=-2,axis2=-1).reshape(2,512),smooth(x,1),atol=1e-12)
    def test_regions(self):
        for w,q in regions().items():
            self.assertEqual(q.shape,(512,(8//w)**3));np.testing.assert_allclose(q.sum(0),1)
    def test_control_identity(self):
        x=np.random.default_rng(4).normal(size=(3,512));case={'mu':np.ones(512)}
        np.testing.assert_allclose(distort(x,case,np.zeros((8,8,8)),'identity'),x)
    def test_linear_reference(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(10000,2))
        r=linear_summary(x,dict(mu=np.zeros(2),sigma=np.eye(2)),np.eye(2))
        self.assertAlmostEqual(r['exact_reference_interval_coverage_mean'],.9,delta=.015)


if __name__=='__main__':unittest.main()
