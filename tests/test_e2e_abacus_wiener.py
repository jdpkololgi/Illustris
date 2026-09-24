import unittest
import numpy as np
from workflows.sbi.e2e_abacus_wiener import covariance, draw_wiener


class WienerTest(unittest.TestCase):
    def test_covariance_operator_and_dense_posterior(self):
        shape=(2,2,2);n=8
        spectrum=np.arange(1,9,dtype=float).reshape(shape)/4
        response=np.arange(n,dtype=float).reshape(shape)/5
        noise=np.linspace(.5,2,n).reshape(shape)
        data=np.linspace(-1,1,n).reshape(shape)
        columns=[covariance(np.eye(n)[i].reshape(shape),spectrum).ravel() for i in range(n)]
        cov=np.stack(columns,axis=1)
        np.testing.assert_allclose(cov,cov.T,atol=1e-12)
        precision=np.linalg.inv(cov)+np.diag((response**2/noise).ravel())
        expected_cov=np.linalg.inv(precision)
        expected_mean=expected_cov@(response*data/noise).ravel()
        samples,info=draw_wiener(spectrum,response,noise,data,4096,21)
        flat=samples.reshape(-1,n)
        self.assertLess(info['max_relative_residual'],2e-7)
        self.assertLess(np.max(abs(flat.mean(0)-expected_mean)/np.sqrt(np.diag(expected_cov))),.07)
        self.assertLess(np.linalg.norm(np.cov(flat.T)-expected_cov)/np.linalg.norm(expected_cov),.09)

    def test_zero_observation_response_recovers_prior(self):
        shape=(2,2,2);spectrum=np.ones(shape)*2
        samples,_=draw_wiener(spectrum,np.zeros(shape),np.ones(shape),np.ones(shape)*999,1024,31)
        self.assertLess(abs(samples.mean()),.05)
        self.assertLess(abs(samples.var()-2),.1)

    def test_invalid_noise_rejected(self):
        x=np.ones((2,2,2))
        with self.assertRaises(ValueError):draw_wiener(x,x,x*0,x,2,1)


if __name__=='__main__':unittest.main()
