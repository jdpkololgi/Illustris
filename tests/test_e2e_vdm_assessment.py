import unittest
import numpy as np
import torch
from workflows.sbi.e2e_direct_vdm import LinearSchedule
from workflows.sbi.e2e_vdm_assessment_math import coupled_sample,pool2,tidal_features,calibration,regional_density


class GaussianOracle(torch.nn.Module):
    def __init__(self):super().__init__();self.schedule=LinearSchedule(learned=False)
    def forward(self,z,g,c):return self.schedule.coefficients(g)[1][:,None,None,None,None]*z


class AssessmentTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def test_batch_partition_replay(self):
        m=GaussianOracle();c=torch.zeros(3,1,8,8,8);seeds=[3,4,5]
        all=coupled_sample(m,c,10,seeds,40)
        parts=torch.cat([coupled_sample(m,c[i:i+1],10,[seeds[i]],40) for i in range(3)])
        torch.testing.assert_close(all,parts,atol=0,rtol=0)
        self.assertTrue(m.training)
    def test_resolution_strong_convergence(self):
        m=GaussianOracle();c=torch.zeros(8,1,8,8,8);ss=list(range(8))
        a,b,d=[coupled_sample(m,c,n,ss,400) for n in [25,100,400]]
        self.assertLess(float((b-d).square().mean()),float((a-d).square().mean()))
        self.assertGreater(float(d.var()),float(a.var()))
    def test_invalid_grid(self):
        with self.assertRaises(ValueError):coupled_sample(GaussianOracle(),torch.zeros(1,1,8,8,8),7,[1],100)
    def test_constant_tide_and_gaps(self):
        x=np.full((8,8,8),.6)
        for b in ['periodic','reflect']:
            f=tidal_features(x,b,core=4,stride=1)
            np.testing.assert_allclose(f[:,1:4],.2,atol=1e-6)
            np.testing.assert_allclose(f[:,4:],0,atol=1e-6)
        f=tidal_features(x,'zero',core=4,stride=1)
        np.testing.assert_allclose(f[:,1:4].sum(1),f[:,0],atol=1e-6)
    def test_plane_wave(self):
        x=np.broadcast_to(np.cos(2*np.pi*np.arange(8)/8)[:,None,None],(8,8,8))
        f=tidal_features(x,'periodic',core=8,stride=1)
        expected=np.sort(np.stack([x,np.zeros_like(x),np.zeros_like(x)],axis=-1),axis=-1).reshape(-1,3)
        np.testing.assert_allclose(f[:,1:4],expected,atol=1e-6)
    def test_pool_tensor_before_eigen(self):
        a=np.arange(8**3*6).reshape(8,8,8,6)
        out=pool2(a);self.assertEqual(out.shape,(4,4,4,6))
        np.testing.assert_allclose(out[0,0,0],a[:2,:2,:2].mean((0,1,2)))
    def test_crps_bruteforce(self):
        x=np.array([[1.,2.],[3.,4.],[5.,6.]]);y=np.array([2.,3.])
        c=calibration(x,y);expected=np.abs(x-y).mean(0)-.5*np.abs(x[:,None]-x[None]).mean((0,1))
        np.testing.assert_allclose(c['crps'],expected)
        self.assertTrue(np.all((c['rank']>0)&(c['rank']<1)))
    def test_oracle_coverage_and_underdispersion(self):
        rng=np.random.default_rng(81);y=rng.normal(size=3000);x=rng.normal(size=(128,3000))
        c=calibration(x,y);n=calibration(x*.1,y)
        self.assertLess(abs(c['covered_0.9'].mean()-.9),.04)
        self.assertLess(n['covered_0.9'].mean(),.2)
    def test_region_order(self):
        x=np.zeros((8,8,8));x[:4,:4,:4]=7
        np.testing.assert_allclose(regional_density(x),[8,1,1,1,1,1,1,1])


if __name__=='__main__':unittest.main()
