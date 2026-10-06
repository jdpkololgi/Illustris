import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'workflows/sbi'))
import e2e_trace_shear_diagnostic as d
import e2e_operator_split as split


class DiagnosticTests(unittest.TestCase):
    def test_width_variance_includes_covariance(self):
        rng=np.random.default_rng(29)
        x=rng.normal(size=(32,2,3))
        x[...,2]+=x[...,0]*2
        f=d.features(x)
        for i in range(2):
            cov=np.cov(x[:,i,:].T)
            self.assertAlmostEqual(f['width'][:,i].var(ddof=1),cov[0,0]+cov[2,2]-2*cov[0,2])
        np.testing.assert_allclose(f['width'],f['gap12']+f['gap23'])

    def test_trace_shift_preserves_gaps(self):
        x=np.random.default_rng(1).normal(size=(32,2,3))
        a,b=d.features(x),d.features(x+3)
        np.testing.assert_allclose(b['trace'],a['trace']+9)
        for name in ('gap12','gap23','width'):np.testing.assert_allclose(a[name],b[name],atol=1e-14)

    def test_compression_can_increase_extreme_classes(self):
        x=np.array([[-3.,1.,1.],[-1.,-1.,3.]])
        compressed=x.mean(-1,keepdims=True)+.1*(x-x.mean(-1,keepdims=True))
        np.testing.assert_equal((x>0).sum(-1),[2,1])
        np.testing.assert_equal((compressed>0).sum(-1),[0,3])

    def test_local_refinement_preserves_trace(self):
        fine=np.exp(np.random.default_rng(3).normal(scale=.1,size=(8,8,8)))
        coarse=split.b.op.mean_pool(fine,2)
        tc=split.patch_tensor(coarse,(2,2,2),1)
        tf=split.patch_tensor(fine,(4,4,4),2)
        avg=np.stack([split.b.op.mean_pool(tf[...,k],2) for k in range(6)],axis=-1)
        np.testing.assert_allclose((avg-tc)[...,[0,3,5]].sum(-1),0,atol=1e-12)


if __name__=='__main__':unittest.main()
