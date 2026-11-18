import unittest
import numpy as np
from workflows.sbi.p12a_cic_response_basis import compress_lattice,expected_counts,validate_geometry
from workflows.sbi.cfm_cic_quadrature import contract_lattice


class RadialBasisTests(unittest.TestCase):
    def test_full_and_oof_are_exact_for_arbitrary_training_curves(self):
        factor=4;cell=2.;shape=np.array([4,5,3]);edges=np.arange(0.,101.,10.)
        rng=np.random.default_rng(43)
        for origin in (np.array([-4.,-3.,-2.]),np.array([6.,8.,0.]),np.array([95.,0.,0.])):
            axes=[origin[i]-cell+(np.arange((shape[i]+2)*factor)+.5)*cell/factor for i in range(3)]
            xyz=np.meshgrid(*axes,indexing='ij',sparse=True)
            radius=np.sqrt(sum(a*a for a in xyz));bins=np.searchsorted(edges,radius,side='right')-1
            inside=(radius>0)&(bins>=0)&(bins<len(edges)-1)
            intensity=rng.uniform(0.,2.,size=radius.shape)*inside
            weights=compress_lattice(intensity,bins,factor,cell,edges)
            centers=[origin[i]+(np.arange(shape[i])+.5)*cell for i in range(3)]
            centers=np.meshgrid(*centers,indexing='ij',sparse=True)
            center_bin=np.clip(np.searchsorted(edges,np.sqrt(sum(a*a for a in centers)),side='right')-1,0,len(edges)-2)
            for omitted in range(21):
                density=rng.uniform(0.,.003,len(edges)-1);valid=rng.uniform(size=len(density))>.15
                safe=np.clip(bins,0,len(density)-1)
                direct=contract_lattice(intensity*density[safe]*valid[safe],factor,cell)
                compressed=expected_counts(weights,center_bin,density,valid)
                np.testing.assert_allclose(compressed,direct,rtol=3e-14,atol=1e-16)

    def test_rejects_aliased_radial_support(self):
        with self.assertRaises(ValueError):validate_geometry([0.,1.,2.],1.)

    def test_negative_or_unknown_selection_rejected(self):
        with self.assertRaises(ValueError):expected_counts(np.ones((3,1)),np.array([0]),[-1.,2.],[True,True])
        with self.assertRaises(ValueError):expected_counts(np.ones((3,1)),np.array([-1]),[1.,2.],[True,True])


if __name__=='__main__':unittest.main()
