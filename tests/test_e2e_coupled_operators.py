import unittest
import numpy as np
from workflows.sbi import e2e_coupled_operators as op
from workflows.sbi.e2e_vdm_context_products import sample_averaged
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.abacus_tweb.p3a_build_canonical_fields import GridSpec


class CoupledOperatorsTests(unittest.TestCase):
    def test_positive_rectangular_roundtrip_and_mass(self):
        rng=np.random.default_rng(7)
        rho=np.exp(rng.normal(size=(16,12,8)))
        coarse,residual=op.encode(rho)
        rebuilt=op.decode(coarse,residual)
        np.testing.assert_allclose(rebuilt,rho,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(op.mean_pool(residual),0,atol=2e-15)
        arbitrary=rng.normal(size=rho.shape)*20
        sample=op.decode(coarse,arbitrary)
        np.testing.assert_allclose(op.mean_pool(sample),coarse,rtol=1e-12)
        self.assertTrue(np.all(sample>0))

    def test_rectangular_tensor_trace_and_plane_wave(self):
        shape=(16,12,8)
        x=np.arange(shape[0])[:,None,None]
        delta=np.broadcast_to(np.sin(2*np.pi*x/shape[0]),shape)+2.3
        tensor=op.tensor_from_delta(delta,3.,dc=False)
        np.testing.assert_allclose(tensor[...,0],delta-2.3,atol=2e-14)
        np.testing.assert_allclose(tensor[...,1:],0,atol=2e-14)
        random=np.random.default_rng(2).normal(size=shape)
        tensor=op.tensor_from_delta(random,3.)
        np.testing.assert_allclose(tensor[...,[0,3,5]].sum(-1),random,atol=2e-14)
        tensor=op.tensor_from_delta(np.full(shape,.6),3.)
        np.testing.assert_allclose(tensor[...,[0,3,5]],.2,atol=1e-14)
        np.testing.assert_allclose(tensor[...,[1,2,4]],0,atol=1e-14)

    def test_consistent_rectangular_matched_domain(self):
        delta=np.random.default_rng(3).normal(size=(16,12,8))+.4
        coarse=op.mean_pool(delta)
        tensor=op.consistent_tensor(delta,coarse,[(0,4),(0,3),(0,2)])
        truth=op.tensor_from_delta(delta,6.766)
        np.testing.assert_allclose(tensor,truth,atol=2e-14)

    def test_sparse_restriction_matches_existing_cubic_operator(self):
        field=np.random.default_rng(1).normal(size=(32,32,32))
        origin=[-23.5,156.3,281.4]
        grid=coord.grid_record(GridSpec(tuple(origin),(32,32,32),3.383,27.064))
        cfg=dict(raw_cell_mpc=3.383,coordinate_h=1.,box_offset_mpc_h=-1000.,box_mpc_h=2000.)
        starts=[np.arange(n)*2-12 for n in (8,6,4)]
        expected=sample_averaged(field,starts,2,{'origin_mpc':origin},cfg)
        actual=op.sample_averaged_local(field,starts,2,grid)
        np.testing.assert_allclose(actual,expected,atol=2e-14,rtol=2e-14)
        constant=op.sample_averaged_local(np.ones_like(field),starts,8,grid)
        np.testing.assert_allclose(constant,1,atol=2e-14)


if __name__=='__main__':
    unittest.main()
