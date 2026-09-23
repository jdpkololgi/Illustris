import unittest
from workflows.sbi.e2e_reference_controls_report import group, sampler_differences


class ReportTests(unittest.TestCase):
    def test_seed_and_case_grouping(self):
        rows=[]
        for seed in [17,29]:
            for case in [0,1]:
                for nfe in [128,256]:
                    rows.append(dict(branch='teacher',fixed=None,seed=seed,case=case,nfe=nfe,
                        name=f'teacher{seed}',mean_rms=.08,covariance_relative=.07,
                        variance_ratio=1.,octant_coverage=.9,power_ratio=[1.,1.3],
                        passed=True,power_pass=False,risk={'exact_target_mse':.01}))
        g=group(rows,'branch')['teacher_amortised']
        self.assertEqual(g['cells'],4)
        self.assertEqual(g['original_pass'],4)
        self.assertEqual(g['joint_pass'],0)
        self.assertEqual(g['mean_frozen_exact_target_mse'],.01)
        self.assertEqual(len(sampler_differences(rows)),4)
        self.assertEqual(max(r['max_absolute_power_difference'] for r in sampler_differences(rows)),0.)


if __name__=='__main__':unittest.main()
