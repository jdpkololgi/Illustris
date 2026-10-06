import unittest
import numpy as np
from workflows.sbi.e2e_cfm_mixture_futility import covariance, moments, metric, effect, energy, decision, LIMITS


class FutilityTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(32)
        self.x=self.rng.normal(size=(3,16,4));self.t=np.zeros((3,4))

    def test_identical_components(self):
        q=effect(self.x,self.x,self.t,metric(self.x),1.)
        self.assertTrue(all(abs(v)<1e-12 for v in q.values()))
        es=energy(self.x,self.x,self.t)
        np.testing.assert_allclose(es[:,0],es[:,2],atol=1e-14)

    def test_covariance_identity(self):
        y=self.x+2;m=moments(self.x,y)
        np.testing.assert_allclose(m['covariance_mix'],covariance(self.x)+np.ones((3,4,4)))
        np.testing.assert_allclose(m['between_debiased'],m['between'])

    def test_noise_correction(self):
        y=self.rng.normal(size=self.x.shape);m=moments(self.x,y)
        np.testing.assert_allclose(m['between']-m['between_debiased'],covariance(y-self.x)/64)

    def test_mean_and_variance_controls(self):
        w=metric(self.x)
        self.assertGreater(effect(self.x,self.x+2,self.t,w,1)['mean_shift_sd'],.1)
        self.assertGreater(effect(self.x,self.x*2,self.t,w,1)['covariance_change'],.1)

    def test_energy_direct_independent_pairs(self):
        x=self.x[:1,:4];y=x+.7;truth=self.t[:1]
        direct=sum(np.linalg.norm(v-truth[0]) for v in np.concatenate((x[0],y[0])))/8
        pair=0.
        for i in range(4):
            for j in range(4):
                if i==j:continue
                pair+=sum(np.linalg.norm(a[0,i]-b[0,j]) for a in (x,y) for b in (x,y))/4
        direct-=.5*pair/12
        self.assertAlmostEqual(direct,energy(x,y,truth)[0,2])

    def test_three_way_decision(self):
        g=dict(draws=8,valid=True,bands={k:[0,0,0] for k in LIMITS})
        self.assertEqual(decision([g]*4),'futile_on_screened_support')
        self.assertEqual(decision([dict(g,draws=4)]*4),'insufficient_evidence')
        b={k:[0,0,v*2] for k,v in LIMITS.items()}
        self.assertEqual(decision([dict(g,bands=b)]*4),'insufficient_evidence')
        b['mean_shift_sd']=[.2,.3,.4]
        self.assertEqual(decision([dict(g,bands=b)]*4),'material_effect_detected')


if __name__=='__main__':unittest.main()
