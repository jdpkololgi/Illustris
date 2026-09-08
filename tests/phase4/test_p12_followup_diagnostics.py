import unittest
import numpy as np
import torch
from workflows.sbi.p12f3_d2_support_geometry import edge_distance, native_support, summary
from workflows.sbi.p12b_representation_diagnostics import norm, permuted_indices


class FollowupTests(unittest.TestCase):
    def test_patch_faces(self):
        points=np.array([[0,2,2],[2,2,2],[4,2,2],[-1,2,2]])
        np.testing.assert_allclose(edge_distance(points,[-.5]*3,[4.5]*3),[.5,2.5,.5,-.5])

    def test_context_and_core_are_distinct(self):
        points=np.array([[24.,30.,30.]])
        self.assertEqual(float(edge_distance(points,[-.5]*3,[79.5]*3)[0]),24.5)
        self.assertEqual(float(edge_distance(points,[23.5]*3,[55.5]*3)[0]),.5)

    def test_permutation_preserves_strata(self):
        cap=np.repeat([0,1],20); shell=np.tile(np.repeat([0,1],10),2)
        p=permuted_indices(cap,shell,42)
        np.testing.assert_array_equal(np.sort(p),np.arange(40))
        np.testing.assert_array_equal(cap[p],cap)
        np.testing.assert_array_equal(shell[p],shell)
        np.testing.assert_array_equal(p,permuted_indices(cap,shell,42))
        self.assertTrue(np.any(p!=np.arange(40)))

    def test_gradient_group_norm(self):
        a=torch.nn.Parameter(torch.ones(2));a.grad=torch.tensor([3.,4.])
        b=torch.nn.Parameter(torch.ones(1))
        self.assertEqual(norm([a,b]),5.)
        self.assertEqual(norm([b]),0.)

    def test_native_support_radial_and_angular(self):
        # A synthetic all-sky map, with radius mapped monotonically to redshift.
        support=np.ones(12*256**2,dtype=bool)
        domain=np.full(len(support),2,dtype=np.int8)
        selection={"cosmology":{"radius_grid_mpc":[0.,1000.],"redshift_grid":[0.,1.]}}
        xyz=np.array([[50.,0,0],[200.,0,0],[590.,0,0],[700.,0,0]])
        mask,angular,radial,_,_=native_support(xyz,1,support,domain,selection)
        np.testing.assert_array_equal(mask,[False,True,False,False])
        self.assertTrue(angular.all())
        np.testing.assert_array_equal(mask,radial)
        other=native_support(xyz,0,support,domain,selection)[0]
        self.assertFalse(other.any())

    def test_summary(self):
        self.assertEqual(summary([]),{"n":0})
        self.assertEqual(summary([1,2,3])["quantiles"]["p50"],2)


if __name__=="__main__":
    unittest.main()
