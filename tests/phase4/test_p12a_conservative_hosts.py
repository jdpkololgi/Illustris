import unittest
import numpy as np
from workflows.sbi.p12a_conservative_hosts import compatible_candidates, brute_candidates


class ConservativeHostsTest(unittest.TestCase):
    def test_all_ambiguities_and_member_intersection(self):
        xyz=np.array([[0.,0,0],[.3,0,0]])
        native=np.array([[.1,0,0],[-.1,0,0],[.4,0,0],[0,0,0]])
        masses=np.array([12.,12.,12.,13.]);radius=np.full(4,.25)
        found=dict(compatible_candidates(xyz,np.array([12.,12.]),[0,0],[7,7],native,masses,radius))
        np.testing.assert_array_equal(found[7],[0])
        expected=brute_candidates(xyz,np.array([12.,12.]),[0,0],native,masses,radius)
        np.testing.assert_array_equal(found[7],expected)

    def test_periodic_ambiguous_no_nearest(self):
        native=np.array([[999.9,0,0],[-999.9,0,0],[0.,0,0]])
        found=dict(compatible_candidates([[-1000.,0,0]],[20.],[0],[4],native,
                                         np.full(3,20.),np.full(3,.2)))
        np.testing.assert_array_equal(found[4],[0,1])

    def test_central_and_no_match(self):
        native=np.array([[0.,0,0],[.006,0,0]])
        found=dict(compatible_candidates([[0.,0,0],[3.,0,0]],[20.,20.],[1,0],[0,1],
                                         native,np.full(2,20.),np.full(2,1.)))
        np.testing.assert_array_equal(found[0],[0])
        self.assertEqual(len(found[1]),0)

    def test_randomized_against_full_scan(self):
        rng=np.random.default_rng(37)
        native=rng.uniform(-5,5,(400,3));masses=rng.integers(10,15,400).astype(float)
        radii=rng.uniform(.2,2,400)
        xyz=rng.uniform(-5,5,(80,3));mass=rng.integers(10,15,80).astype(float)
        groups=np.repeat(np.arange(40),2);central=np.zeros(80,bool)
        found=dict(compatible_candidates(xyz,mass,central,groups,native,masses,radii))
        for group in range(40):
            ids=groups==group
            np.testing.assert_array_equal(found[group],brute_candidates(
                xyz[ids],mass[ids],central[ids],native,masses,radii))

    def test_fold_diagnostic_rejects_single_linked_component(self):
        from workflows.sbi.p12a_host_fold_diagnostic import audit
        rng=np.random.default_rng(99)
        xyz=rng.uniform(10,2000,(200,3));points=np.column_stack((xyz,np.ones(200)))
        cap=np.arange(200)%2;shell=np.arange(200)%4
        result=audit(points,cap,shell,np.ones(200,bool),np.ones(200,bool),
                     np.zeros(200,int),{0:np.zeros(3),1:np.zeros(3)})
        self.assertFalse(result['passed'])
        self.assertEqual(result['host_linked_components'],1)
        for attempt in result['candidates']:
            self.assertTrue(attempt['gates']['no_conservative_host_fold_crossing'])
            self.assertFalse(attempt['gates']['five_nonempty_folds'])


if __name__=='__main__': unittest.main()
