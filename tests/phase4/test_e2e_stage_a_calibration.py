import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'workflows/sbi'))
import e2e_stage_a_calibration as a


class StageATests(unittest.TestCase):
    def test_phase_guard(self):
        for p in ('ph001','ph014','ph018','ph019','Loa'):
            with self.assertRaises(PermissionError):a.guard(p)
        for p in a.PHASES:a.guard(p)

    def test_shear_basis_norm_and_trace(self):
        x=np.random.default_rng(1).normal(size=(4,6));s=a.shear_coordinates(x)
        traceless=x.copy();traceless[:,[0,3,5]]-=x[:,[0,3,5]].mean(-1,keepdims=True)
        np.testing.assert_allclose((s*s).sum(-1),(traceless**2*[1,2,2,1,2,1]).sum(-1))
        x[:,[0,3,5]]+=4
        np.testing.assert_allclose(a.shear_coordinates(x),s,atol=1e-14)

    def test_fair_crps_direct(self):
        x=np.random.default_rng(2).normal(size=(32,2));y=np.array([.1,.2])
        direct=np.abs(x-y).mean()-np.abs(x[:,None]-x[None,:]).sum()/(2*32*31*2)
        self.assertAlmostEqual(a.scalar_scores(x,y)['crps'],direct)
        self.assertAlmostEqual(a.scalar_scores(x,y)['reference']['0.9'],29/33)

    def test_degenerate_and_empty(self):
        self.assertEqual(a.vector_scores(np.ones((32,2)),np.ones(2))['status'],'conditioned_or_degenerate')
        self.assertEqual(a.scalar_scores(np.empty((32,0)),np.empty(0))['status'],'empty_support')

    def test_covariance_not_ignored(self):
        rng=np.random.default_rng(3);x=np.sort(rng.normal(size=(32,2,3)),axis=-1)
        y=np.sort(rng.normal(size=(2,3)),axis=-1);q=a.eig_metrics(x,y)
        expected=(x[:,:,2]-x[:,:,0]).std(axis=0,ddof=1).mean()
        self.assertAlmostEqual(q['width']['sd'],expected)

    def test_group_denominators(self):
        m=a.scalar_scores(np.arange(32.)[:,None],np.array([16.]))
        rows=[dict(phase='ph012',cap='NGC',shell=0,support_stratum='interior',metrics={'x':m})]
        groups=a.aggregate(rows)
        self.assertEqual(groups[0]['metrics']['x']['cases'],1)
        self.assertEqual(groups[1]['metrics']['x']['split_deficit90_0p10'],'insufficient_cores')

    def test_reconstruction_fraction_and_brier(self):
        import e2e_stage_a_reconstruction as r
        eig=np.zeros((4,2,2,2,2,3));truth=np.zeros((2,2,2,2,3))
        eig[:2,...,2]=1
        result,fractions,actual=r.class_summary(eig,truth,.2)
        np.testing.assert_allclose(fractions.sum(-1),1)
        np.testing.assert_allclose(actual.sum(-1),1)
        self.assertAlmostEqual(result['brier'],.5)
        self.assertAlmostEqual(result['fair_brier'],1/3)


if __name__=='__main__':unittest.main()
