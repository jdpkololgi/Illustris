import copy
import unittest
from unittest.mock import patch
import numpy as np
from astropy.cosmology import Planck18
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi.e2e_coupled_geometry import source_position
from workflows.abacus_tweb.p3a_build_canonical_fields import GridSpec


class CoordinateTests(unittest.TestCase):
    def test_pinned_table_roundtrip_and_units(self):
        z=np.linspace(.1,.6,301)
        r=coord.radius_mpc_h(z)
        np.testing.assert_allclose(coord.redshift(r),z,atol=2e-11,rtol=0)
        self.assertTrue(np.all(r>250))
        self.assertTrue(np.all(r<1600))
        self.assertAlmostEqual(float(coord.radius_mpc_h(0)),0.)
        with self.assertRaises(ValueError):
            coord.radius_mpc_h(-.1)
        # A distance table already in Mpc/h must NOT be multiplied by h again.
        sky=coord.sky_mpc_h([0.,90.,0.],[0.,0.,90.],[.2]*3)
        np.testing.assert_allclose(sky,np.eye(3)*coord.radius_mpc_h(.2),atol=1e-10)

    def test_radial_measure_is_preserved_including_h_cubed(self):
        zz=np.linspace(0,.8,4001)
        rr=Planck18.comoving_distance(zz).value
        selection={'cosmology':{'redshift_grid':zz,'radius_grid_mpc':rr}}
        z=np.linspace(.1013,.5987,205)
        i=np.searchsorted(zz,z,side='right')-1
        old_dv=np.interp(z,zz,rr)**2*np.diff(rr)[i]/np.diff(zz)[i]
        new_dv=coord.radius_mpc_h(z)**2*coord.table()[2](z,1)
        jac=coord.selection_volume_jacobian(z,selection)
        np.testing.assert_allclose(jac*new_dv,old_dv,rtol=5e-15)
        self.assertTrue(np.all((jac>3.) & (jac<3.4)))
        # The correction is not a constant h conversion.
        self.assertGreater(np.ptp(jac*.6766**3),.0001)

    def test_grid_and_native_mapping_fail_on_ambiguous_units(self):
        spec=GridSpec((100.,200.,300.),(32,32,32),3.383,27.064)
        grid=coord.grid_record(spec)
        p=source_position([8,16,24],grid)
        np.testing.assert_allclose(p,(np.array(spec.origin)+np.array([8,16,24])*3.383-1000)%2000)
        for key,value in [('distance_unit','Mpc'),('cell_mpc_h',5.),('coordinate_sha256','wrong'),('origin_mpc',[1,2,3])]:
            bad=copy.deepcopy(grid);bad[key]=value
            with self.assertRaises(ValueError):
                source_position([0,0,0],bad)

    def test_table_hash_is_checked(self):
        coord.table.cache_clear()
        with patch.object(c,'sha256',return_value='wrong'):
            with self.assertRaises(ValueError):
                coord.table()
        coord.table.cache_clear()


if __name__=='__main__':
    unittest.main()
