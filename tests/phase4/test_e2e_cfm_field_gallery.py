import unittest
import numpy as np
from workflows.sbi import e2e_cfm_field_gallery as g


class GalleryTests(unittest.TestCase):
    def test_cic_crop_matches_full(self):
        rng=np.random.default_rng(41)
        xyz=rng.uniform(-2,42,(200,3))
        grid=dict(origin_mpc_h=[0,0,0],cell_mpc_h=2.)
        full,_=g.patch_counts(xyz,grid,np.zeros(3,dtype=int),(20,20,20))
        crop,audit=g.patch_counts(xyz,grid,np.array([4,6,8]),(8,8,8))
        np.testing.assert_allclose(crop,full[4:12,6:14,8:16],atol=1e-6)
        self.assertGreater(audit['lost_weight'],0)

    def test_overlay_uses_exact_slab(self):
        grid=dict(origin_mpc_h=[0,0,0],cell_mpc_h=3.383)
        xyz=np.array([[10,20,23*6.766],[11,21,25*6.766],[12,22,24*6.766]])
        points=g.overlay(xyz,grid,np.array([64,48,48]))
        np.testing.assert_array_equal(points,[[10,20],[12,22]])

    def test_all_channels_compared(self):
        a={k:np.zeros((12,2,2,2)) for k in ('joint','wide')}
        b={k:v.copy() for k,v in a.items()}
        self.assertTrue(all(r['passed'] for r in g.compare_channels(a,b)))
        b['wide'][11,0,0,0]=1
        rows=g.compare_channels(a,b)
        self.assertEqual(len(rows),24)
        self.assertEqual(sum(not r['passed'] for r in rows),1)


if __name__=='__main__': unittest.main()
