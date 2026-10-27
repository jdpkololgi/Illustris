import unittest
import healpy as hp
import numpy as np
from workflows.sbi.p12a_parent_response import sample_centres
from workflows.abacus_tweb import p3br_build_random_response as b

class ParentResponseTests(unittest.TestCase):
    def test_pointwise_formula_matches_registered_full_grid(self):
        component={'grid':{'origin_mpc':[-40.,-40.,-40.],'shape':[8,8,8],'cell_mpc':10.,'padding_mpc':0.}}
        spec=b.grid_from_component(component)
        selection={'cosmology':{'radius_grid_mpc':[0.,100.],'redshift_grid':[0.,1.]}}
        nside=8; pixels=np.arange(hp.nside2npix(nside));cap_support=pixels%5!=0
        angle=np.linspace(0,.3,len(pixels),dtype=np.float32)
        previous=getattr(b,'_SELECTION_CONTEXT',None);b._SELECTION_CONTEXT=selection
        try: pix,r,z=b.angular_block(spec,(slice(0,8),)*3,nside)
        finally:b._SELECTION_CONTEXT=previous
        binary=(z>=.1)&(z<.6)&~((z>=.585)&(z<.595))&cap_support[pix]
        radial=np.minimum.reduce([abs(r-q) for q in [10.,60.,58.5,59.5]])
        expected=np.minimum(angle[pix].astype(np.float64)*r,radial).astype(np.float32)*binary
        indices=np.stack(np.meshgrid(np.arange(8),np.arange(8),np.arange(8),indexing='ij'),axis=-1).reshape(-1,3)
        # Non-centred galaxy positions must still address the same grid voxels.
        xyz=np.array(spec.origin)+(indices+.21)*spec.cell_mpc
        d,s=sample_centres(xyz,component,cap_support,angle,selection)
        np.testing.assert_array_equal(s,binary.ravel())
        np.testing.assert_array_equal(d,expected.ravel())
        d,s=sample_centres(np.array([[1000.,0.,0.]]),component,cap_support,angle,selection)
        self.assertFalse(s[0]);self.assertTrue(np.isnan(d[0]))

if __name__=='__main__':unittest.main()
