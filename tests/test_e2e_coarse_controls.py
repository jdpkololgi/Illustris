import unittest
import numpy as np
from workflows.sbi.e2e_coarse_controls import guard,covariance,gaussian_draws,regions,scored
from workflows.sbi.e2e_cfm_pilot_evaluate import regional
from workflows.sbi import e2e_coupled_operators as op
from workflows.sbi import e2e_coupled_views as views

class CoarseTests(unittest.TestCase):
    def test_phase_guard(self):
        for p in ('ph016','ph019','ph001','ph006'):
            with self.assertRaises(PermissionError):guard(p)
        guard('ph014')
        with self.assertRaises(PermissionError):guard('ph014',fit=True)
        guard('ph024',fit=True)

    def test_block_closure(self):
        rng=np.random.default_rng(1);wide=np.exp(rng.normal(size=(48,)*3)*.1)
        _,crop=views.crop_slices('ph012',(0,0,0));base=op.lift(wide[crop],4)
        residual=op.project(rng.normal(size=base.shape))
        np.testing.assert_allclose(regions(wide,'ph012'),regional(base+residual-1)[:10],atol=1e-14)

    def test_dense_posterior(self):
        shape=(2,2,2);n=8;power=np.linspace(.2,2,n).reshape(shape)
        response=np.linspace(0,1,n).reshape(shape);noise=np.ones(shape)*.4;data=np.ones(shape)*.3
        cov=np.column_stack([covariance(np.eye(n)[i].reshape(shape),power).ravel() for i in range(n)])
        exact=np.linalg.inv(np.linalg.inv(cov)+np.diag((response**2/noise).ravel()))
        mean=exact@(response*data/noise).ravel()
        draws,info=gaussian_draws(power,response,noise,data,1024,17)
        flat=draws.reshape(-1,n)
        self.assertLess(np.linalg.norm(np.cov(flat.T)-exact)/np.linalg.norm(exact),.13)
        self.assertLess(np.max(abs(flat.mean(0)-mean)/np.sqrt(np.diag(exact))),.12)
        self.assertLess(info['max_residual'],2e-7)

    def test_mask_no_information(self):
        p=np.ones((2,)*3)
        a,_=gaussian_draws(p,p*0,p,p*0,4,2)
        b,_=gaussian_draws(p,p*0,p,p*100,4,2)
        np.testing.assert_array_equal(a,b)

if __name__=='__main__':unittest.main()
