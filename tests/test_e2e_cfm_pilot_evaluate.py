import unittest
import numpy as np
import torch
from types import SimpleNamespace
from unittest.mock import patch
from workflows.sbi.e2e_coupled_benchmark_models import sample
from workflows.sbi.e2e_cfm_pilot_evaluate import development,draw_seed,probes,features,regional,matched_variogram
from workflows.sbi import e2e_coupled_operators as op
from workflows.sbi.e2e_vdm_context_metrics import calibration


class EvaluateTests(unittest.TestCase):
    def test_mixed_checkpoint_routing(self):
        from workflows.sbi.e2e_cfm_pilot_evaluate import load_models
        seen=[]
        class Fake:
            def __init__(self,*args):pass
            def cuda(self):return self
            def eval(self):return self
            def load_state_dict(self,state):pass
        def state(path,**kwargs):
            stage,seed=path.parent.name.split('_');step=int(path.stem.split('_')[1]);seen.append((stage,step))
            return dict(step=step,ema={},binding=dict(stage=stage,seed=int(seed),confirmation_access=False,normalizer='chart',model='hash'))
        with patch('workflows.sbi.e2e_cfm_pilot_evaluate.CoupledBackbone',Fake), patch('torch.load',side_effect=state), patch('workflows.sbi.e2e_cfm_pilot_evaluate.c.sha256',return_value='hash'):
            load_models(17,13312,26624)
        self.assertEqual(seen,[('coarse',13312),('fine',26624)])
    def test_replication_scope(self):
        with patch('workflows.sbi.e2e_cfm_pilot_evaluate.EVAL_PHASES',('ph014','ph015')):
            development('ph014');development('ph015')
            for phase in ('ph012','ph016','ph017','ph018','ph019','ph000'):
                with self.assertRaises(PermissionError):development(phase)
    def test_heun_linear_oracle_and_noise_pairing(self):
        class Linear(torch.nn.Module):
            stage='coarse'
            domain='wide'
            def schedule(self,t):return t
            def forward(self,z,t,condition):return .2*z
        model=Linear()
        cond=SimpleNamespace(joint=torch.zeros(2,1,4,4,4),region='wide',validate=lambda *args:None)
        seeds=[41,43]
        base=torch.cat([torch.randn((1,1,4,4,4),generator=torch.Generator().manual_seed(s)) for s in seeds])
        with patch.dict('workflows.sbi.e2e_coupled_benchmark_models.SHAPES',wide=(4,4,4)):
            a=sample(model,cond,'cfm',8,seeds);b=sample(model,cond,'cfm',16,seeds)
        expected=base*(1+.2/8+.5*(.2/8)**2)**8
        torch.testing.assert_close(a,expected,rtol=1e-6,atol=1e-6)
        exact=base*np.exp(.2)
        self.assertLess(float((b-exact).square().mean()),float((a-exact).square().mean()))
    def test_sealed_phases(self):
        for phase in ('ph014','ph019','ph000','ph004','ph001'):
            with self.assertRaises(PermissionError):development(phase)
        development('ph012');development('ph013')
    def test_addressing(self):
        a=draw_seed(17,'ph012','pair',0,'coarse')
        self.assertEqual(a,draw_seed(17,'ph012','pair',0,'coarse'))
        self.assertNotEqual(a,draw_seed(17,'ph012','pair',0,'fine'))
        self.assertNotEqual(a,draw_seed(29,'ph012','pair',0,'coarse'))
    def test_probes_annihilate_coarse_blocks(self):
        basis=probes()
        np.testing.assert_allclose(np.linalg.norm(basis.reshape(7,-1),axis=1),1)
        for w in basis:np.testing.assert_allclose(op.mean_pool(w),0,atol=1e-14)
        block=np.ones((16,12,12));field=op.lift(block,4)
        np.testing.assert_allclose(features(field),0,atol=1e-12)
        np.testing.assert_allclose(regional(field),1)
    def test_coverage_null_and_underdispersion(self):
        rng=np.random.default_rng(6);x=rng.normal(size=(32,4096));y=rng.normal(size=4096)
        exact=calibration(x,y);narrow=calibration(.3*x,y)
        self.assertLess(abs(exact['covered_0.9'].mean()-exact['attainable']['0.9']),.025)
        self.assertLess(narrow['covered_0.9'].mean(),.5)
    def test_dependence_control(self):
        rng=np.random.default_rng(22);good=[];bad=[]
        for _ in range(128):
            y=rng.normal(size=7);truth=np.r_[y,y]
            z=rng.normal(size=(32,7));paired=np.concatenate((z,z),axis=1)
            erased=rng.normal(size=(32,14))
            good.append(matched_variogram(paired,truth));bad.append(matched_variogram(erased,truth))
        self.assertAlmostEqual(np.mean(good),0)
        self.assertGreater(np.mean(bad),.5)


if __name__=='__main__':unittest.main()
