import unittest
from types import SimpleNamespace
import numpy as np
import torch
from workflows.sbi.e2e_conditional_reference_math import problem
from workflows.sbi.e2e_reference_target_controls import Teacher, bridge
from workflows.sbi.e2e_spectral_preconditioning import Spectral, WhitenedTeacher, targets, items, sample
from workflows.sbi.e2e_spectral_absorption import absorbed_variance
from workflows.sbi.e2e_spectral_bridge_diagnostics import transformed_case


class SpectralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        _,chol,mask,_,cases,_,_=problem(4,2)
        cls.data=SimpleNamespace(n=4,chol=torch.tensor(chol,dtype=torch.float32),
            mask=torch.tensor(mask,dtype=torch.float32),cases=cases)
        cls.conditions=[torch.tensor(np.stack([c['y'],c['mask'],c['std']]),dtype=torch.float32)
                        .reshape(1,3,4,4,4) for c in cases]
        freq=np.fft.fftfreq(4)*4
        radius=sum(a*a for a in np.meshgrid(freq,freq,freq,indexing='ij'))
        cls.transform=Spectral(1/(1+radius))
        cls.teacher=Teacher(cls.data)
        cls.white=WhitenedTeacher(cls.data,cls.transform)

    def test_inverse_and_dc(self):
        x=torch.randn(5,1,4,4,4)
        torch.testing.assert_close(self.transform(self.transform(x),inverse=True),x,atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(self.transform(torch.ones_like(x)),torch.ones_like(x))

    def test_colored_bridge_pairing(self):
        x=torch.randn(5,1,4,4,4);c=self.conditions[0].expand(5,-1,-1,-1,-1)
        args=(self.transform,self.teacher,self.white)
        z,t,v=targets(x,c,torch.Generator().manual_seed(8),'physical',False,*args)
        u,tt,w=targets(x,c,torch.Generator().manual_seed(8),'coordinates',False,*args)
        torch.testing.assert_close(u,self.transform(z));torch.testing.assert_close(w,self.transform(v))
        torch.testing.assert_close(t,tt)
        zz,tt,vv=bridge(x,torch.Generator().manual_seed(8))
        torch.testing.assert_close(z,zz,rtol=0,atol=0);torch.testing.assert_close(v,vv,rtol=0,atol=0)

    def test_changed_teacher_independent_joint_solve(self):
        w=self.transform(torch.eye(64).reshape(64,1,4,4,4)).flatten(1).double().numpy()
        rng=np.random.default_rng(817)
        for i,c in enumerate(self.data.cases):
            cov=w@c['sigma']@w.T;mu=w@c['mu'];z=rng.normal(size=64)
            for t in [0.,.2,.8,1.]:
                expected=mu+(t*cov-(1-t)*np.eye(64))@np.linalg.solve(t*t*cov+(1-t)**2*np.eye(64),z-t*mu)
                actual=self.white(torch.tensor(z,dtype=torch.float32).reshape(1,1,4,4,4),torch.tensor([t]),self.conditions[i]).flatten().numpy()
                np.testing.assert_allclose(actual,expected,atol=3e-5,rtol=3e-5)

    def test_absorption_zero_and_positive(self):
        lam=np.array([.01,.1,1.])
        np.testing.assert_allclose(absorbed_variance(lam,np.zeros(3)),lam,rtol=1e-7)
        self.assertTrue(np.all(absorbed_variance(lam,np.full(3,.01))>lam))

    def test_balanced_design(self):
        panel=items();self.assertEqual(len(panel),32)
        self.assertEqual(len({i['name'] for i in panel}),32)
        for a in ['physical','weighted','coordinates','white_bridge']:
            self.assertEqual(sum(i['arm']==a for i in panel),8)

    def test_absorption_basis_roundtrip(self):
        original=self.data.cases[0]
        case,basis,w=transformed_case(original,self.transform.scale.numpy())
        np.testing.assert_allclose((basis*case['values'])@basis.T,original['sigma'],atol=1e-12)
        np.testing.assert_allclose(np.linalg.solve(w,case['mu']),original['mu'],atol=1e-12)

    def test_sampler_coordinate_equivariance(self):
        teacher=self.teacher;transform=self.transform
        class Exact(torch.nn.Module):
            def __init__(self,coordinates):
                super().__init__();self.coordinates=coordinates
            def schedule(self,t):return t
            def forward(self,z,t,c):
                if self.coordinates:return transform(teacher(transform(z,inverse=True),t,c))
                return teacher(z,t,c)
        c=self.conditions[0].expand(4,-1,-1,-1,-1)
        x=sample(Exact(False),c,128,torch.Generator().manual_seed(21),'physical',transform)
        y=sample(Exact(True),c,128,torch.Generator().manual_seed(21),'coordinates',transform)
        torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-5)


if __name__=='__main__':unittest.main()
