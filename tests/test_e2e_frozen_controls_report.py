import itertools
import unittest
from workflows.sbi.e2e_frozen_controls_report import optimizer_summary, sampler_summary


class FrozenReportTests(unittest.TestCase):
    def test_factorial_controls_matched(self):
        rows=[]
        for seed,anchor,q,mom,clip,variant in itertools.product((0,1),range(3),(.005,.01,.05),
                ('retained','zero_first'),(True,False),('denoising_only','identity_strong','projected_identity')):
            scale=1 if variant=='denoising_only' else 2
            before=[dict(sigma=s,clean_mse=3.,noisy_mse=4.) for s in (.005,.01,.05)]
            # Factorial-dependent control levels expose incorrect cross-cell pairing.
            baseline=1+seed+int(clip)+(mom=='retained')
            after=[dict(sigma=s,clean_mse=baseline/scale,noisy_mse=baseline*scale) for s in (.005,.01,.05)]
            rows.append(dict(replica=seed,anchor_id=anchor,gradient_sigma=q,momentum=mom,clipping=clip,
                variant=variant,before=before,after=after,clipping_scale=1.,raw_primary_dot_aux=0.,
                incremental_primary_dot_vs_control=0.,actual_primary_dot_displacement=-1.))
        report=optimizer_summary(rows)
        self.assertEqual(len(report),24)
        for r in report:
            self.assertEqual(r['median_clean_ratio'],1 if r['variant']=='denoising_only' else .5)
            self.assertEqual(r['median_noisy_ratio'],1 if r['variant']=='denoising_only' else 2)
            self.assertEqual(r['joint_nonworse'],27 if r['variant']=='denoising_only' else 0)
        with self.assertRaises(ValueError):optimizer_summary(rows[:-1])

    def test_sampler_requires_full_panel(self):
        with self.assertRaises(ValueError):sampler_summary([])

if __name__=='__main__':unittest.main()
