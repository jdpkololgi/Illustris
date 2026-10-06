import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'workflows/sbi'))
import e2e_stage_b_metrics as m


class StageBTests(unittest.TestCase):
    def test_probability_endpoints(self):
        np.testing.assert_array_equal(m.probability_bins([0,.2,.4,.6,.8,1]),[0,1,2,3,4,4])
        with self.assertRaises(ValueError):m.probability_bins([-1e-4])

    def test_brier_and_merging(self):
        p=np.array([[.5,.5,0,0],[.5,.5,0,0]])
        full=m.reliability(p,[0,0],['a','b'],draws=4)
        pieces=[m.reliability(p[i:i+1],[0],[str(i)],draws=4) for i in range(2)]
        a,b=m.merge_reliability([full]),m.merge_reliability(pieces)
        self.assertAlmostEqual(a['brier'],.5)
        self.assertAlmostEqual(a['fair_brier'],1/3)
        self.assertEqual(a['brier'],b['brier'])

    def test_empty_and_absent_class(self):
        r=m.reliability(np.empty((0,4)),np.array([],int),np.array([],str),draws=32)
        self.assertEqual(m.merge_reliability([r])['status'],'empty_support')
        f,n=m.region_fractions(np.zeros((32,2,4),int),np.array([[1,2,0,0],[0,0,0,0]]))
        np.testing.assert_array_equal(f[:,0,0],1)
        self.assertTrue(np.isnan(f[:,1]).all())

    def test_scalar_tied_exchangeability(self):
        z=np.array([0,0,0,1,1,1,2,3,3],float)/3
        results=[m.fraction_score(np.delete(z,i)[:,None],np.array([z[i]])) for i in range(len(z))]
        for p in m.LEVELS:
            self.assertAlmostEqual(np.mean([r['coverage'][str(p)] for r in results]),p)

    def test_joint_tied_exchangeability_and_permutation(self):
        z=np.array([[0,0],[0,0],[0,0],[1,0],[0,1],[1,1],[1,1],[2,1],[1,2]],float)
        results=[m.joint_fraction_score(np.delete(z,i,axis=0),z[i]) for i in range(len(z))]
        for p in m.LEVELS:
            self.assertAlmostEqual(np.mean([r['coverage'][str(p)] for r in results]),p)
        self.assertEqual(m.joint_fraction_score(z[1:],z[0])['coverage'],
                         m.joint_fraction_score(z[:0:-1],z[0])['coverage'])

    def test_fraction_crps_direct(self):
        rng=np.random.default_rng(2);x=rng.uniform(size=(32,2,4));y=rng.uniform(size=(2,4))
        q=m.fraction_score(x,y)
        direct=abs(x-y).mean(0)-abs(x[:,None]-x[None,:]).sum((0,1))/(2*32*31)
        np.testing.assert_allclose(q['crps'],direct,atol=1e-14)

    def test_addresses_and_phase_guard(self):
        import e2e_stage_b_prepare as p
        xyz=np.array([[0,0,0],[16*6.766,0,0],[32*6.766,0,0],[-1e-9,0,0]])
        inside,flat=p.cell_addresses(xyz,np.zeros(3))
        np.testing.assert_array_equal(inside,[True,True,False,False])
        np.testing.assert_array_equal(flat,[0,4096])
        for phase in ('ph001','ph018','ph019','ph014'):
            for train in (True,False):
                with self.assertRaises(PermissionError):p.guard(phase,train)
        with self.assertRaises(PermissionError):p.guard('ph012',True)
        with self.assertRaises(PermissionError):p.guard('ph000',False)

    def test_biased_and_underdispersed_fraction_sensitivity(self):
        rng=np.random.default_rng(10)
        z=rng.dirichlet([2,3,4,1],size=(33,100))
        oracle=m.fraction_score(z[1:],z[0])
        bad=m.fraction_score(np.broadcast_to([.97,.01,.01,.01],(32,100,4)),z[0])
        narrow=m.fraction_score(z[1:].mean(0)+.05*(z[1:]-z[1:].mean(0)),z[0])
        self.assertLess(np.mean(bad['coverage']['0.9']),.2)
        self.assertLess(np.mean(narrow['coverage']['0.9']),.3)
        self.assertGreater(np.mean(oracle['coverage']['0.9']),.8)


if __name__=='__main__':unittest.main()
