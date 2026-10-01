import unittest
import numpy as np
from workflows.sbi.e2e_loa_field_atlas import tile_keys,groups
from workflows.sbi.e2e_loa_field_tensor import at_galaxies


class AtlasTests(unittest.TestCase):
    def test_half_open_unique_ownership(self):
        grid=dict(origin_mpc_h=[0,0,0],cell_mpc_h=3.383)
        xyz=np.array([[0,0,0],[63.999,31.999,31.999],[64,32,32],[128,64,64]])*3.383
        np.testing.assert_array_equal(tile_keys(xyz,grid),[[0,0,0],[0,0,0],[1,1,1],[2,2,2]])

    def test_cap_groups_do_not_overlap(self):
        grid=dict(origin_mpc_h=[0,0,0],cell_mpc_h=3.383)
        props=dict(CAP=np.array([0,1,1,0]),xyz=np.array([[1,1,1],[2,2,2],[220,120,120],[3,3,3]]))
        out=groups(props,dict(grids=dict(NGC=grid,SGC=grid)))
        np.testing.assert_array_equal(np.sort(np.concatenate(list(out.values()))),np.arange(4))
        self.assertEqual(set(out),{'NGC_000_000_000','NGC_001_001_001','SGC_000_000_000'})

    def test_tensor_interpolation_before_eigensolve(self):
        field=np.zeros((64,48,48,6));field[...,0]=.1;field[...,3]=.3;field[...,5]=.5
        grid=dict(origin_mpc_h=[0,0,0],cell_mpc_h=3.383)
        xyz=np.array([[17.2,18.3,19.7],[40.7,22.1,20.9]])*6.766
        e=at_galaxies(field,xyz,grid,[64,48,48])
        np.testing.assert_allclose(e,[[.1,.3,.5],[.1,.3,.5]])
        np.testing.assert_array_equal((e>.2).sum(-1),[2,2])
        with self.assertRaises(ValueError):at_galaxies(field,np.array([[-1,0,0]]),grid,[64,48,48])


if __name__=='__main__':unittest.main()
