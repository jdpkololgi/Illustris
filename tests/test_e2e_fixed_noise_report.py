import unittest
import numpy as np
from workflows.sbi.e2e_fixed_noise_report import summarize
from workflows.sbi.e2e_fixed_noise_response import decomposition
from workflows.sbi.e2e_wide_denoising_audit import Bands


class FixedNoiseReportTests(unittest.TestCase):
    def fixture(self):
        arms=['vp_parent','vp_fresh','residual_unet','residual_highres']
        metrics={k:[1.]*4 for k in ('gain','error_power','error_over_truth_power','noise_amplitude','power_ratio','noise_correlated_error_power')}
        probes=[dict(group=g,phase=p,ratio=r,metrics=metrics,velocity_mse=1.)
                for g in ('fit','transfer') for p in ('ph000','ph002','ph003')
                for r in (.05,.2,1.,5.,20.) for _ in range(2)]
        results=[dict(arm=a,ratio=r,parameters=100,elapsed_seconds=1.,gate={},
                      curve=[dict(update=u,probes=probes,summary={}) for u in (0,128,256,512)],
                      training=[dict(gradient_norm_before_clip=2.,loss=1.)]*512)
                 for a in arms for r in (.05,.2)]
        return dict(complete=True,registration=dict(config=dict(arms=arms,clip=1.),
                    panel=dict(gate=dict(near_clean_ratios=[.05,.2],max_abs_noise_amplitude=.2,
                                         max_error_vs_parent=.25,signal_gain_min=.9,signal_gain_max=1.1))),
                    checkpoints={str(i):'test' for i in range(24)},fields=[{}]*72,elapsed_seconds=1.,
                    training_ready=False,calibration_pass=None,baseline=probes,results=results,
                    references=[dict(label=k,probes=probes,summary={},gate={},noiseless={}) for k in ('identity','lowpass')])

    def test_count_clipping_and_phase_curves(self):
        s=summarize(self.fixture())
        self.assertEqual(s['counts'],dict(updates=4096,probes=2100,reconstructions=72,checkpoints=24))
        self.assertEqual(len(s['branches']),8)
        self.assertEqual(s['branches']['residual_unet/0.05']['clipped_fraction'],1.)
        self.assertEqual(len(s['branches']['residual_unet/0.05']['phase_curves']),24)

    def test_incomplete_rejected(self):
        data=self.fixture();data['fields'].pop()
        with self.assertRaises(ValueError):
            summarize(data)

    def test_antithetic_error_power_closure(self):
        rng=np.random.default_rng(42);bias=rng.normal(size=(8,8,8));noise=rng.normal(size=(8,8,8))
        bands=Bands(8,3.383,[0,.32,.64,1.,np.inf])
        result=decomposition(bands,bias+noise,bias-noise,bias)
        np.testing.assert_allclose(result['even_clean_correlation'],np.ones(4))
        np.testing.assert_allclose(result['even_minus_clean_error_power'],np.zeros(4),atol=1e-30)


if __name__=='__main__':
    unittest.main()
