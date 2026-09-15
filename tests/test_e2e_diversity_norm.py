import copy
import math
import unittest
import numpy as np
import torch
from torch import nn
from workflows.sbi.e2e_diversity_norm import AffineChart, identity_norm, inverse_norm, fit_norm, field_for, overlap
from workflows.sbi.e2e_multinoise_models import coefficients
from workflows.sbi.e2e_diversity_norm_report import ranks, correlation, groups, summarize, with_metadata, finite_tree, paired_effect


class Toy(nn.Module):
    tau=.05
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.tensor(.4))
    def forward(self,x,t,c,w,context_present=None):
        return self.weight*x+.1*c[:,:1]+.2*w[:,:1]+t[:,None,None,None,None]


class DiversityNormTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41);torch.set_num_threads(1)
        self.norm=identity_norm();self.norm.update(target_mean=.3,target_std=1.7,
              fine_mean=[.2]*13,fine_std=[1.2]*13,wide_mean=[-.4]*12,wide_std=[.8]*12)
        self.x=torch.randn(5,1,4,4,4);self.c=torch.randn(5,13,4,4,4);self.w=torch.randn(5,12,4,4,4)
        self.t=torch.tensor([0.,.001,.03,.5,1.])

    def test_exact_identity_chart(self):
        raw=Toy();m=AffineChart(raw,identity_norm())
        self.assertTrue(torch.equal(raw(self.x,self.t,self.c,self.w),m(self.x,self.t,self.c,self.w)))

    def test_roundtrip_function_and_gradients(self):
        raw=Toy();m=AffineChart(AffineChart(raw,inverse_norm(self.norm)),self.norm)
        v0=raw(self.x,self.t,self.c,self.w);v1=m(self.x,self.t,self.c,self.w)
        torch.testing.assert_close(v0,v1,atol=1e-6,rtol=1e-5)
        g0=torch.autograd.grad(v0.square().mean(),raw.weight,retain_graph=True)[0]
        g1=torch.autograd.grad(v1.square().mean(),raw.weight)[0];torch.testing.assert_close(g0,g1)

    def test_physical_clean_output(self):
        raw=Toy();m=AffineChart(raw,self.norm);a,b,_=coefficients(self.t)
        s=self.norm['target_std'];mu=self.norm['target_mean'];q=(s*s*a*a+b*b).sqrt()
        tn=(2/math.pi)*torch.atan2(b,s*a).flatten();xn=(self.x-a*mu)/q
        vn=raw(xn,tn,(self.c-.2)/1.2,(self.w+.4)/.8)
        expected=s*((s*a/q)*xn-(b/q)*vn)+mu
        actual=a*self.x-b*m(self.x,self.t,self.c,self.w)
        torch.testing.assert_close(expected,actual,atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(actual[0],self.x[0]);self.assertTrue(torch.isfinite(actual).all())

    def test_physical_noise_unchanged(self):
        y=torch.randn_like(self.x);eps=torch.randn_like(y);a,b,_=coefficients(self.t)
        s=self.norm['target_std'];mu=self.norm['target_mean'];q=(s*s*a*a+b*b).sqrt()
        x=a*y+b*eps;xn=(x-a*mu)/q
        torch.testing.assert_close(xn,(s*a/q)*(y-mu)/s+(b/q)*eps)

    def test_each_field_exposed_to_every_noise_bin(self):
        for n in (3,6,9,15):
            ids=list(range(n));seen={}
            for step in range(18*(n//3)):
                field=field_for(step,n,ids);key=(field,(step//3)%6);seen[key]=seen.get(key,0)+1
            self.assertEqual(set(seen.values()),{1});self.assertEqual(len(seen),6*n)

    def test_train_normalization_global_not_per_field(self):
        rng=np.random.default_rng(9)
        items=[dict(target=rng.normal(i,2,(1,4,4,4)),coarse=rng.normal(i,3,(1,4,4,4)),
                    condition=rng.normal(i,2,(13,4,4,4)),wide=rng.normal(i,3,(12,4,4,4))) for i in (0,3)]
        norm=fit_norm(items);both=np.stack([v['target'] for v in items])
        self.assertAlmostEqual(norm['target_mean'],float(both.mean()))
        self.assertAlmostEqual(norm['target_std'],float(both.std()))
        self.assertEqual(norm['fine_mean'][1],0.);self.assertEqual(norm['fine_std'][1],1.)
        self.assertNotAlmostEqual(float(((items[0]['target']-norm['target_mean'])/norm['target_std']).mean()),0.)

    def test_periodic_overlap_and_phase(self):
        a=dict(phase='ph000',source_box_footprint=[[[1900,2000],[0,100]],[[0,100]],[[0,100]]])
        b=dict(phase='ph000',source_box_footprint=[[[50,150]],[[50,150]],[[50,150]]])
        self.assertTrue(overlap(a,b));b['phase']='ph002';self.assertFalse(overlap(a,b))

    def test_invalid_scaling(self):
        for val in (0.,-1.,float('nan')):
            norm=copy.deepcopy(self.norm);norm['target_std']=val
            with self.assertRaises(ValueError):AffineChart(Toy(),norm)

    def test_report_matched_groups(self):
        prep={'selection':{'train':[dict(anchor_id=str(i)) for i in range(15)],
                           'transfer':[dict(anchor_id='t'+str(i)) for i in range(12)]}}
        g=groups(prep,6)
        self.assertEqual(g['common_fit'],['0','1','2']);self.assertEqual(len(g['exposed']),6)
        self.assertEqual(len(g['unused']),9);self.assertEqual(len(g['transfer']),12)

    def test_report_correlations_with_ties(self):
        np.testing.assert_equal(ranks([2,1,2,3]),[1.5,0,1.5,3])
        self.assertAlmostEqual(correlation([1,2,3],[2,4,6]),1.)
        self.assertIsNone(correlation([1,1,1],[1,2,3]))

    def test_report_gate_uses_paired_parent(self):
        rows=[];parent={}
        for rep,error in enumerate([1.,2.]):
            r=dict(anchor_id='a',kind='noisy',ratio=.05,rep=rep,
                metrics=dict(noise_amplitude=[0,0,0,.1],error_power=[0,0,0,error],gain=[1,1,1,1]))
            rows.append(r);parent['a',.05,rep]=dict(metrics=dict(error_power=[0,0,0,10*error]))
        r=summarize(rows,['a'],parent)['0.05'];self.assertAlmostEqual(r['error_vs_parent'],.1)
        self.assertEqual(r['passed'],1)

    def test_report_field_relative_distortion(self):
        rows=[dict(anchor_id='a',kind='clean',ratio=.05,rms=.02,bias=0.,max_abs=.1,rms_over_injected=1.,
                   metrics=dict(error_power=[0,0,0,.01],gain=[1,1,1,1]))]
        out=with_metadata(rows,['a'],{},dict(a=dict(stats=dict(std=.4))))
        self.assertAlmostEqual(out['0.05']['clean_rms_over_field_std'],.05)

    def test_report_rejects_nonfinite_metrics(self):
        self.assertTrue(finite_tree({'a':[None,1.,'metadata']}))
        self.assertFalse(finite_tree({'a':[float('nan')]}))
        self.assertFalse(finite_tree({'a':[float('inf')]}))

    def test_report_pairing_not_row_order(self):
        before=[dict(anchor_id='a',kind='clean',ratio=.05,rms=2.),
                dict(anchor_id='a',kind='noisy',ratio=.05,rep=0,metrics=dict(noise_amplitude=[.4],error_power=[2.]))]
        after=[dict(anchor_id='a',kind='noisy',ratio=.05,rep=0,metrics=dict(noise_amplitude=[.1],error_power=[1.])),
               dict(anchor_id='a',kind='clean',ratio=.05,rms=1.)]
        effect=paired_effect(before,after,['a'])
        self.assertEqual(effect['clean_improved'],1)
        self.assertEqual(effect['median']['clean_rms_ratio'],.5)
        self.assertAlmostEqual(effect['median']['noise_left_change'],-.3)


if __name__=='__main__':unittest.main()
