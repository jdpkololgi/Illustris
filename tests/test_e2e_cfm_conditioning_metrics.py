import unittest
import numpy as np
from workflows.sbi import e2e_cfm_conditioning_metrics as m
from workflows.sbi.e2e_vdm_context_metrics import fair_energy,calibration
from workflows.sbi.e2e_stage_c_mechanism_metrics import moments
from workflows.sbi.e2e_cfm_conditioning_evaluation import fullcap_task,SETTINGS


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(818)
        self.index,self.counts=m.resamples()

    def test_whole_draw_addresses(self):
        np.testing.assert_array_equal(self.index,np.random.default_rng(6100617).integers(0,32,(4096,32)))
        np.testing.assert_array_equal(self.counts.sum(1),32)

    def test_energy_direct_resampling(self):
        x=self.rng.normal(size=(3,32,6));y=self.rng.normal(size=(3,6));k=m.energy_kernel(x,y)
        self.assertAlmostEqual(m.score_from_kernel(k),fair_energy(x.transpose(1,0,2),y).mean())
        got=m.score_from_kernel(k,self.counts[:4])
        for value,ix in zip(got,self.index[:4]):
            self.assertAlmostEqual(value,fair_energy(x[:,ix].transpose(1,0,2),y).mean())

    def test_brier_direct_resampling(self):
        x=self.rng.integers(0,4,(32,83));y=self.rng.integers(0,4,83);w=self.rng.integers(0,10,83)
        def direct(z):
            p=np.stack([(z==j).mean(0) for j in range(4)],-1)
            score=((p-np.eye(4)[y])**2-p*(1-p)/31).sum(-1)
            return np.average(score,weights=w)
        k=m.brier_kernel(x,y,w)
        self.assertAlmostEqual(m.score_from_kernel(k),direct(x))
        for value,ix in zip(m.score_from_kernel(k,self.counts[:5]),self.index[:5]):
            self.assertAlmostEqual(value,direct(x[ix]))

    def test_ratio_direct_resampling(self):
        x=self.rng.normal(size=(7,32,6));y=self.rng.normal(size=(7,6))
        actual=m.ratio_samples(x,y,self.counts[:4])
        for value,ix in zip(actual,self.index[:4]):
            self.assertAlmostEqual(value,moments(x[:,ix],y)['trace_adjusted_mse_over_variance'])

    def test_coverage_inclusive_ties(self):
        for discrete in (False,True):
            x=self.rng.normal(size=(32,53));y=self.rng.normal(size=53)
            if discrete:x=x.round();y=y.round()
            got=m.cell_curve_samples(x,y,self.counts[:6],chunk=9)
            for result,ix in zip(got,self.index[:6]):
                q=calibration(x[ix],y)
                np.testing.assert_array_equal(result,[q[f'covered_{p}'].mean() for p in m.LEVELS])

    def test_identical_proper_scores_pass(self):
        x=self.rng.normal(size=(3,32,6));y=self.rng.normal(size=(3,6));k=m.energy_kernel(x,y)
        self.assertEqual(m.deterioration(k,k,self.counts)['decision'],'pass')
        self.assertEqual(m.combine(['pass','inconclusive']),'inconclusive')
        self.assertEqual(m.combine(['fail','inconclusive']),'fail')

    def test_complete_frozen_grid(self):
        grid=[fullcap_task(i) for i in range(224)]
        self.assertEqual(len(set(grid)),224)
        self.assertEqual(grid[0],('ph012','NGC',0,'parent',0.))
        self.assertEqual(grid[40],('ph012','SGC',0,'control',0.))
        self.assertEqual(grid[-1],('ph013','SGC',28,'augmented',.2))
        for arm,strength in SETTINGS:
            self.assertEqual(sum(q[3:]==(arm,strength) for q in grid),32)
        for invalid in (-1,224):
            with self.assertRaises(ValueError):fullcap_task(invalid)


if __name__=='__main__':unittest.main()
