import unittest
import numpy as np
import torch
from workflows.sbi.e2e_global_canvas import Geometry,integrate
from workflows.sbi.e2e_global_sampler import integrate as pair_integrate
from workflows.sbi.e2e_vdm_context_models import project,block_mean,decode_density

class Constant:
    schedule=staticmethod(lambda t:t)
    def __call__(self,x,t,c):return torch.ones_like(x)*c

class CanvasTests(unittest.TestCase):
    def test_exact_pair_replay(self):
        g=Geometry.make([[288,432,112],[352,432,112]],'coarse')
        self.assertEqual(g.shape,(56,48,48))
        z=torch.randn((1,1,*g.shape))
        ref=pair_integrate(Constant(),'coarse',[1.,3.],z,nfe=4)
        actual,_=integrate(Constant(),'coarse',[1.,3.],z,g,nfe=4)
        for i,r in enumerate(ref):self.assertTrue(torch.equal(r,actual[g.crop(i)]))
    def test_corner_reduction_and_holes(self):
        g=Geometry(((0,0,0),(2,2,2)),(6,6,6),(0,0,0),(4,4,4))
        z=torch.zeros((1,1,6,6,6));actual,w=integrate(Constant(),'coarse',[1.,3.],z,g,nfe=4)
        expected=torch.zeros_like(z);expected[g.crop(0)]+=1;expected[g.crop(1)]+=3
        self.assertTrue(torch.equal(actual,expected/w.clamp_min(1)))
        self.assertTrue(torch.equal(actual[w==0],z[w==0]))
    def test_fine_block_alignment(self):
        mid=[[288+64*i,432+32*j,112+32*k] for i in range(2) for j in range(2) for k in range(2)]
        g=Geometry.make(mid,'fine');self.assertEqual(g.shape,(96,64,64))
        z=project(torch.randn((1,1,*g.shape)))
        actual,_=integrate(Constant(),'fine',[1.]*8,z,g,nfe=4)
        self.assertLess(float(block_mean(actual).abs().max()),1e-6)
    def test_reject_bad_windows(self):
        for mid in ([[0,0,0],[0,0,0]],[[1,0,0]]):
            with self.assertRaises(ValueError):Geometry.make(mid,'fine')
    def test_global_decode_matches_independent_window_decodes(self):
        mid=[[288+64*i,432+32*j,112+32*k] for i in range(2) for j in range(2) for k in range(2)]
        cg,fg=(Geometry.make(mid,s) for s in ('coarse','fine'))
        z=torch.randn((1,1,*cg.shape),dtype=torch.float64)*.1
        u=project(torch.randn((1,1,*fg.shape),dtype=torch.float64))
        offset=np.asarray(fg.origin)//4-np.asarray(cg.origin)
        cut=(slice(None),slice(None),*(slice(int(a),int(a+n//4)) for a,n in zip(offset,fg.shape)))
        whole=decode_density(z[cut].exp(),u)
        for i in range(len(mid)):
            local=decode_density(z[cg.crop(i)][:,:,16:32,18:30,18:30].exp(),u[fg.crop(i)])
            self.assertTrue(torch.equal(local,whole[fg.crop(i)]))
if __name__=='__main__':unittest.main()
