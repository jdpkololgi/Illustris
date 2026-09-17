import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np

from workflows.sbi.e2e_vdm_context_analysis import (density_spectra,calibration_summary,paired_refinement,
    paired_feature_widths,refinement_pass)
from workflows.sbi.e2e_vdm_context_report import pair_report,contrast,block_interval,contrast_decision,report


class AnalysisTests(unittest.TestCase):
    def test_power_of_mean_is_not_mean_power(self):
        rng=np.random.default_rng(2)
        truth=rng.normal(size=(48,48,48))
        draws=np.stack([truth+rng.normal(size=truth.shape) for _ in range(8)])
        result,power=density_spectra(draws,truth)
        np.testing.assert_allclose(result['mean_sample_power'],
            np.asarray(result['posterior_mean_power'])+result['posterior_residual_power_population'],rtol=1e-12)
        self.assertTrue(np.all(np.asarray(result['mean_sample_power'])>result['posterior_mean_power']))
        self.assertEqual(power.shape,(8,5))
        np.testing.assert_allclose(result['posterior_mean_power_mc_corrected'],
            np.asarray(result['posterior_mean_power'])-np.asarray(result['posterior_residual_power_unbiased'])/8)
        same=paired_refinement(draws,draws,power,power,bootstrap=8)
        self.assertEqual(same['mc95_power_bound'],0)
        self.assertEqual(same['mc95_width_bound'],0)

    def test_joint_calibration_masks_and_attainable_level(self):
        rng=np.random.default_rng(4)
        draws=rng.normal(size=(32,10,6))
        truth=rng.normal(size=(10,6))
        out=calibration_summary(draws,truth,np.ones(6),mask=np.arange(10)%2==0)
        self.assertEqual(out['probes'],5)
        self.assertEqual(out['attainable_coverage']['0.9'],29/33)
        self.assertTrue(np.isfinite(out['tidal_joint_energy']))
        empty=calibration_summary(draws,truth,np.ones(6),mask=np.zeros(10,bool))
        self.assertEqual(empty,dict(probes=0))

    def test_pair_score_and_spatial_block_caveat(self):
        rng=np.random.default_rng(19)
        a=dict(core_summary_draws=rng.normal(size=(32,6)).tolist(),core_summary_truth=[0]*6,phase='ph004',cap='NGC')
        b=dict(a,core_summary_draws=rng.normal(size=(32,6)).tolist())
        report=pair_report(a,b,np.ones(6))
        self.assertEqual(np.shape(report['posterior_cross_covariance']),(6,6))
        rows=[dict(task=dict(anchor=str(i)),phase='ph004',source_center_mpc_h=[0,0,0],primary=dict(score=i)) for i in range(3)]
        result=block_interval(rows,rows,'score',500,draws=8)
        self.assertIsNone(result['interval'])
        self.assertEqual(result['blocks'],1)

    def test_pooled_gain_and_component_coverage_do_not_cancel(self):
        def records(score,coverage):
            return [dict(task=dict(anchor=str(i)),phase='ph004',source_center_mpc_h=[i*600,0,0],
                spectra=dict(mean_sample_power=[1]*5,truth_power=[1]*5),
                primary=dict(density_crps=score,tidal_energy=score,density_rmse=1,
                    density_coverage90=.9,tidal_coverage90=float(np.mean(coverage)),
                    tidal_coverage90_components=coverage,attainable90=.9)) for i in range(3)]
        bad=contrast(records(1,[.5,.5,.5]),records(.8,[.7,1.,1.]),'tidal_energy')
        self.assertAlmostEqual(bad['coverage_gap_after'],.4/3)
        self.assertNotAlmostEqual(bad['coverage_gap_after'],0.)
        rows=[]
        for (seed,phase),gain in zip([(s,p) for s in (0,1) for p in ('ph004','ph005')],[.05,.1,.15,.2]):
            result=contrast(records(1,[.9]*3),records(1-gain,[.9]*3),'density_crps')
            rows.append(dict(result,replica=seed,phase=phase))
        self.assertTrue(contrast_decision(rows,'density_crps')['passed'])
        rows[0]['passed']=False
        self.assertFalse(contrast_decision(rows,'density_crps')['passed'])
        with self.assertRaises(ValueError):
            contrast_decision(rows[:3],'density_crps')

    def test_tidal_refinement_can_veto_density_only_success(self):
        x=np.random.default_rng(9).normal(size=(8,12,5))
        same=paired_feature_widths(x,x,bootstrap=8)
        self.assertEqual(same['attainable_width_coverage'],7/9)
        np.testing.assert_array_equal(same['relative_width_change'],np.zeros(5))
        good=dict(max_relative_power_change=0.,relative_width_change=0.,mc95_power_bound=0.,mc95_width_bound=0.)
        self.assertTrue(refinement_pass(good,good,same,same))
        bad=paired_feature_widths(x,x*1.2,bootstrap=8)
        self.assertFalse(refinement_pass(good,good,same,bad))
        self.assertFalse(refinement_pass(good,good,same,same,dict(early=good,late=dict(good,max_relative_power_change=.03))))

    def test_confirmation_verification_precedes_target_reader(self):
        with patch('workflows.sbi.e2e_vdm_context_report.require_compute'), \
             patch('workflows.sbi.e2e_vdm_context_report.output_root',return_value=Path('/tmp')), \
             patch('workflows.sbi.e2e_vdm_context_report.verify_launch'), \
             patch('workflows.sbi.e2e_vdm_context_control.verify_models_frozen',side_effect=ValueError('tampered')), \
             patch('workflows.sbi.e2e_vdm_context_report.Products') as products:
            with self.assertRaisesRegex(ValueError,'tampered'):
                report(Path('/tmp'))
            products.assert_not_called()


if __name__=='__main__':
    unittest.main()
