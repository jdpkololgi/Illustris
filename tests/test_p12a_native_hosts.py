import unittest
import numpy as np
from workflows.sbi.audit_p12a_lowz_native_hosts import check_mapping, native_filter_rows


class NativeHostTests(unittest.TestCase):
    def test_native_filter_uses_strict_corner_distance_and_mass(self):
        xyz=np.array([[-1000,-1000,-1000],[-990,-1000,-1000],[990,-1000,-1000],
                      [-995,-1000,-1000]],dtype='f4')
        original=xyz.copy()
        np.testing.assert_array_equal(native_filter_rows(xyz,[10.,10.,10.,9.],10.),[0])
        np.testing.assert_array_equal(native_filter_rows(xyz,[10.,10.,10.,9.],10.001),[0,1,2])
        np.testing.assert_array_equal(xyz,original)

    def test_filtered_namespace_keeps_satellite_members(self):
        rows=np.array([(0,1,10.),(0,0,10.),(1,1,30.)],dtype=[('HALO_INDEX','i8'),('CEN','i4'),('HALO_MASS','f8')])
        mass=np.array([5.,10.,20.,30.]);xyz=np.array([[0,0,0],[10,0,0],[20,0,0],[30,0,0.]])
        mapped,qa=check_mapping(rows,np.array([[10,0,0],[11,0,0],[30,0,0]]),mass,xyz,np.array([1,3]))
        np.testing.assert_array_equal(mapped,[1,1,3]);self.assertTrue(qa['passed'])
        _,wrong=check_mapping(rows,np.array([[10,0,0],[11,0,0],[30,0,0]]),mass,xyz,np.arange(4))
        self.assertFalse(wrong['passed'])

    def test_bad_position_or_index_rejected(self):
        rows=np.array([(0,1,10.),(4,0,10.)],dtype=[('HALO_INDEX','i8'),('CEN','i4'),('HALO_MASS','f8')])
        _,qa=check_mapping(rows,np.array([[.1,0,0],[0,0,0]]),np.array([10.]),np.zeros((1,3)),np.array([0]))
        self.assertFalse(qa['passed']);self.assertEqual(qa['out_of_range'],1)
        self.assertEqual(qa['central_position_mismatches'],1)

    def test_equal_mass_wrong_satellite_host_rejected(self):
        rows=np.array([(0,0,10.)],dtype=[('HALO_INDEX','i8'),('CEN','i4'),('HALO_MASS','f8')])
        _,qa=check_mapping(rows,np.array([[20.,0.,0.]]),np.array([10.]),np.zeros((1,3)),
                           np.array([0]),np.array([.2]))
        self.assertEqual(qa['mass_mismatches'],0)
        self.assertEqual(qa['satellite_radius_mismatches'],1)
        self.assertFalse(qa['passed'])


if __name__=='__main__':unittest.main()
