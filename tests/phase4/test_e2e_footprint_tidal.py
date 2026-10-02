import unittest
import numpy as np
from numpy.polynomial.legendre import leggauss
from workflows.sbi.e2e_footprint_tidal_operator import cube_tensor,kernel,convolve,source_delta,eigensystem,probabilities


class FootprintTidalTests(unittest.TestCase):
    def test_self_term_and_trace(self):
        np.testing.assert_allclose(cube_tensor(0.,0.,0.),[1/3,0,0,1/3,0,1/3],atol=1e-15)
        for point in [(1.,0.,0.),(4.,-2.,3.),(.1,.2,-.3),(10.2,20.3,30.1)]:
            expected=float(max(abs(x) for x in point)<.5)
            self.assertAlmostEqual(sum(cube_tensor(*point)[[0,3,5]]),expected,places=13)

    def test_independent_volume_quadrature(self):
        g,w=leggauss(18);g=g/2;w=w/2
        xyz=np.stack(np.meshgrid(g,g,g,indexing='ij'),axis=-1)
        weight=w[:,None,None]*w[None,:,None]*w[None,None,:]
        for p in [(1.,0.,0.),(1.1,-2.3,.7),(4.,5.,-6.)]:
            r=np.array(p)-xyz;r2=np.sum(r*r,axis=-1)
            expected=[]
            for a,b in [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]:
                expected.append(np.sum(weight*((a==b)*r2-3*r[...,a]*r[...,b])/(4*np.pi*r2**2.5)))
            np.testing.assert_allclose(cube_tensor(*p),expected,rtol=1e-10,atol=1e-12)

    def test_fft_direct_and_padding(self):
        rng=np.random.default_rng(7);d=rng.normal(size=(4,3,5));shape=(3,4,2);offset=(1.2,-.3,4.1)
        actual=convolve(d,kernel(d.shape,shape,offset),shape)
        direct=np.zeros_like(actual)
        for t in np.ndindex(shape):
            for s in np.ndindex(d.shape):direct[t]+=d[s]*cube_tensor(*(np.array(t)-s+offset))
        np.testing.assert_allclose(actual,direct,atol=2e-14)
        np.testing.assert_allclose(actual,convolve(d,kernel(d.shape,shape,offset,7),shape),atol=2e-14)

    def test_no_periodic_images_and_poisson_trace(self):
        d=np.zeros((6,5,4));d[0,0,0]=1
        t=convolve(d,kernel(d.shape))
        np.testing.assert_allclose(t[-1,0,0],cube_tensor(5.,0.,0.),atol=1e-14)
        self.assertGreater(np.linalg.norm(t[1,0,0]),20*np.linalg.norm(t[-1,0,0]))
        np.testing.assert_allclose(t[...,[0,3,5]].sum(-1),d,atol=1e-13)

    def test_mask_and_drawwise_classes(self):
        r=np.full((3,3,3),2.);m=np.ones_like(r);m[0]=0;m[1]=.5
        a=source_delta(r,m,np.ones_like(r));r[0]=np.nan
        np.testing.assert_equal(source_delta(r,m,np.ones_like(r)),a)
        np.testing.assert_equal(a[:,0,0],[0,.5,1])
        tensor=np.array([[.3,0,0,.25,0,.1],[.3,0,0,.25,0,.21]])
        e=eigensystem(tensor);labels,p=probabilities(e[:,None,:],.2)
        np.testing.assert_equal(labels[:,0],[2,3]);np.testing.assert_equal(p[0],[0,0,.5,.5])

    def test_uniform_cube_centre_and_sign(self):
        d=np.ones((5,5,5));t=convolve(d,kernel(d.shape))
        np.testing.assert_allclose(t[2,2,2],[1/3,0,0,1/3,0,1/3],atol=1e-13)
        np.testing.assert_allclose(convolve(-d,kernel(d.shape)),-t,atol=1e-13)

    def test_invalid_mask_and_face_alignment(self):
        with self.assertRaises(ValueError):kernel((3,3,3),offset=(.5,0,0))
        with self.assertRaises(ValueError):source_delta(np.ones((2,2,2)),np.ones((2,2,2)),np.zeros((2,2,2)))

    def test_interior_exterior_split_and_harmonic_trace(self):
        rng=np.random.default_rng(19);d=rng.normal(size=(7,7,7));m=np.zeros_like(d)
        m[2:5,2:5,2:5]=1;k=kernel(d.shape)
        full=convolve(d,k);inside=convolve(m*d,k);outside=convolve((1-m)*d,k)
        np.testing.assert_allclose(full,inside+outside,atol=2e-14)
        np.testing.assert_allclose(outside[2:5,2:5,2:5][...,[0,3,5]].sum(-1),0,atol=2e-14)
        self.assertGreater(np.linalg.norm(outside[2:5,2:5,2:5]),.01)

if __name__=='__main__':unittest.main()
