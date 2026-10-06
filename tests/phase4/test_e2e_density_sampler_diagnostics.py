import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'workflows/sbi'))
import e2e_density_only_reconstruction as r
import e2e_sampler_factor_diagnostic as d
import e2e_sampler_scale_diagnostic as scale


class ReconstructionTests(unittest.TestCase):
    def test_constant(self):
        rho=np.full((5,6,7),.4)
        for method in ('constant','linear','monotone'):
            np.testing.assert_allclose(r.reconstruct(rho,method),.4,atol=0,rtol=0)

    def test_mass_positive(self):
        rho=np.exp(np.random.default_rng(12).normal(size=(7,8,9))*3)
        for method in ('constant','linear','monotone'):
            fine=r.reconstruct(rho,method)
            self.assertGreater(fine.min(),0)
            np.testing.assert_allclose(r.s.b.op.mean_pool(fine,2),rho,rtol=1e-13,atol=1e-13)

    def test_linear_gradient(self):
        x,y,z=np.indices((6,7,8))
        rho=10.+.2*x+.3*y-.1*z
        fine=r.reconstruct(rho,'linear')
        a,b,c=np.indices((12,14,16))
        expected=10.+.2*(a/2-.25)+.3*(b/2-.25)-.1*(c/2-.25)
        np.testing.assert_allclose(fine[2:-2,2:-2,2:-2],expected[2:-2,2:-2,2:-2],atol=1e-14)

    def test_shear_rotation_metric(self):
        t=np.zeros((2,4,4,4,6));t[...,0]=1;t[...,3]=-1
        self.assertAlmostEqual(d.shear_rms(t),np.sqrt(2))
        t=np.zeros_like(t);t[...,1]=1
        self.assertAlmostEqual(d.shear_rms(t),np.sqrt(2))
        t[...,0]=t[...,3]=t[...,5]=3
        self.assertAlmostEqual(d.shear_rms(t),np.sqrt(2))

    def test_spectral_variance_and_mean_noise(self):
        x=np.arange(8)[:,None,None]
        wave=np.broadcast_to(np.cos(2*np.pi*x/8),(8,8,8)).copy()
        target=2*wave
        draws=target[None]+np.array([-1.,1.,-1.,1.])[:,None,None,None]*wave
        result=scale.bands(draws,target,1.,(0,1.,np.inf))[0]
        self.assertAlmostEqual(result['mean_error_power'],0,places=13)
        self.assertAlmostEqual(result['mean_power'],result['truth_power'])
        self.assertAlmostEqual(result['posterior_variance_power'],result['truth_power']/3)
        self.assertAlmostEqual(result['mean_error_minus_draw_noise'],-result['posterior_variance_power']/4)


if __name__=='__main__':unittest.main()
