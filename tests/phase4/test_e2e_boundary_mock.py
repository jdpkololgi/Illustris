import unittest
import numpy as np
from workflows.sbi import e2e_boundary_mock as m
from workflows.sbi import e2e_coupled_operators as periodic
from workflows.sbi import e2e_footprint_tidal_operator as isolated
from workflows.sbi import e2e_boundary_models as model


class BoundaryMockTests(unittest.TestCase):
    def test_phase_guard(self):
        for phase in ('ph018','ph019','ph014','ph001'):
            with self.assertRaises(PermissionError): m.guard(phase)
        for phase in m.TRAIN+m.EVAL: m.guard(phase)

    def test_padding_and_pool(self):
        a=np.arange(64.).reshape(4,4,4)
        padded=m.crop(a,[-2,0,0],[8,4,4])
        np.testing.assert_array_equal(padded[2:6],a)
        self.assertEqual(padded[:2].sum()+padded[6:].sum(),0)
        np.testing.assert_array_equal(m.pool(padded,2)[1:3],m.pool(a,2))

    def test_core_refinement_geometry(self):
        rows=[dict(cap='NGC',center=[16,16,16])]
        raw=np.arange(64*32*32.).reshape(64,32,32)
        fine=m.pool(raw,2)
        np.testing.assert_allclose(m.core_extract(raw,rows,'NGC',[0,0,0],1),
            m.core_extract(fine,rows,'NGC',[0,0,0],2))

    def test_strict_periodic_dc(self):
        delta=np.ones((8,10,12))*.3
        np.testing.assert_allclose(periodic.tensor_from_delta(delta,1,dc=False),0,atol=1e-15)
        t=periodic.tensor_from_delta(delta,1,dc=True)
        np.testing.assert_allclose(t[...,[0,3,5]].sum(-1),delta,atol=1e-15)

    def test_two_source_linearity(self):
        rng=np.random.default_rng(1); a=rng.normal(size=(5,6,7)); b=rng.normal(size=a.shape)
        k=isolated.kernel(a.shape)
        np.testing.assert_allclose(isolated.convolve(a+b,k),
            isolated.convolve(a,k)+isolated.convolve(b,k),atol=2e-14)

    def test_harmonic_basis_and_oracle(self):
        for degree,k in ((2,5),(3,12)):
            basis=model.harmonic_basis(n=8,degree=degree)
            self.assertEqual(basis.shape,(512,6,k))
            np.testing.assert_allclose(basis[:,[0,3,5]].sum(1),0,atol=1e-14)
            truth=np.arange(k)/10
            field=np.einsum('nck,k->nc',basis,truth)
            mask=np.arange(512)%3!=0
            np.testing.assert_allclose(model.coefficients(field,basis,mask),truth,atol=1e-13)

    def test_training_fit_rejects_evaluation(self):
        x=np.arange(40.).reshape(20,2);y=np.ones((20,5))
        with self.assertRaises(PermissionError):model.fit(x,y,['ph016']*20)
        fitted=model.fit(x,y,['ph007']*10+['ph008']*10)
        np.testing.assert_allclose(model.predict(fitted,x),y,atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(fitted['covariance']).min(),-1e-20)

    def test_score_controls(self):
        truth=np.zeros((10,6));truth[:,[0,3,5]]=.3
        good=model.metrics(truth,truth,np.ones(10,bool))
        bad=model.metrics(np.zeros_like(truth),truth,np.ones(10,bool))
        self.assertEqual(good['class_flip'],0)
        self.assertEqual(bad['class_flip'],1)
        self.assertEqual(good['tensor_mse'],0)
        self.assertIsNone(model.metrics(truth,truth,np.zeros(10,bool)))

    def test_finite_draw_coverage_control(self):
        rng=np.random.default_rng(47)
        draws=rng.normal(size=(64,4096,6))*.1
        truth=rng.normal(size=(4096,6))*.1
        valid=np.ones(4096,bool)
        calibrated=model.distribution_metrics(draws,truth,valid)
        biased=model.distribution_metrics(draws+np.array([2,0,0,2,0,2]),truth,valid)
        np.testing.assert_allclose(calibrated['central90_coverage'],57/65,atol=.02)
        self.assertLess(max(biased['central90_coverage']),.01)


if __name__=='__main__': unittest.main()
