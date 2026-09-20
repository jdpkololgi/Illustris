import copy
import unittest

from workflows.sbi.e2e_conditional_reference_curve_report import aggregate, cfm_loss_floor, highest_shell_null, summarize


class CurveReportTests(unittest.TestCase):
    def fixture(self):
        items = [dict(name=f'{o}_seed{s}_{r}', objective=o,
                      fixed=None if r == 'amortised' else int(r[-1]))
                 for o in ('vdm', 'cfm') for s in (17, 29)
                 for r in ('amortised', 'fixed0', 'fixed1')]
        ext = dict(checkpoints=[4096,8192,16384,32768,65536], updates=65536,
                   nfe_by_objective=dict(vdm=[512,1024], cfm=[128,256]), precision_draws=2048)
        manifest = dict(sources={'a':'b'}, items=items, extension=ext,
                        base=dict(cases=4, fixed_cases=[0,1], objectives=['vdm','cfm'], draws=512))
        def row(i,u,c,n,d):
            return dict(fit=i['name'], objective=i['objective'], update=u, case=c, nfe=n,
                        draws=d, mean_rms=.05, covariance_relative=.1, variance_ratio=1.,
                        octant_coverage=.9, power_ratio=[1.,1.], passed=True)
        rows = [row(i,u,c,n,512) for i in items for u in ext['checkpoints']
                for c in (range(4) if i['fixed'] is None else [i['fixed']])
                for n in ext['nfe_by_objective'][i['objective']]]
        precise = [row(i,65536,c,max(ext['nfe_by_objective'][i['objective']]),2048)
                   for i in items for c in (range(4) if i['fixed'] is None else [i['fixed']])]
        return dict(sources={'a':'b'}, evaluations=rows, precision=precise), manifest

    def test_complete_panel(self):
        done, manifest = self.fixture()
        out = summarize(done,manifest)
        self.assertEqual(out['curve_ensembles'],240)
        self.assertEqual(out['precision_ensembles'],24)
        for group in out['groups'].values():
            self.assertTrue(group['reproducible_matched_four_gate'])
            self.assertEqual(group['precision_matched']['cells'],4)

    def test_one_failed_cell_not_hidden_by_average(self):
        done, manifest = self.fixture()
        target = next(r for r in done['evaluations'] if r['fit']=='cfm_seed29_fixed1'
                      and r['update']==32768 and r['nfe']==256)
        target['passed']=False
        out=summarize(done,manifest)
        self.assertFalse(out['groups']['cfm_fixed']['reproducible_matched_four_gate'])

    def test_transfer_cases_separate_and_power_separate(self):
        done, manifest = self.fixture()
        target = next(r for r in done['precision'] if r['fit']=='vdm_seed17_amortised' and r['case']==3)
        target['passed']=False
        target = next(r for r in done['precision'] if r['fit']=='cfm_seed17_fixed0')
        target['power_ratio']=[1.,1.2]
        out=summarize(done,manifest)
        self.assertTrue(out['groups']['vdm_amortised']['reproducible_matched_four_gate'])
        self.assertFalse(out['groups']['vdm_amortised']['final_all_cases_four_gate'])
        self.assertTrue(out['groups']['cfm_fixed']['reproducible_matched_four_gate'])
        self.assertFalse(out['groups']['cfm_fixed']['final_matched_power_gate'])

    def test_fail_closed(self):
        for kind in ('missing','duplicate','precision_nfe','sources'):
            done, manifest=self.fixture()
            if kind=='missing': done['evaluations'].pop()
            if kind=='duplicate': done['evaluations'].append(copy.deepcopy(done['evaluations'][0]))
            if kind=='precision_nfe': done['precision'][0]['nfe']=32
            if kind=='sources': done['sources']={'a':'wrong'}
            with self.assertRaises(ValueError): summarize(done,manifest)
        with self.assertRaises(ValueError): aggregate([])

    def test_power_null_against_isotropic_chisquare(self):
        import numpy as np
        from scipy.stats import chi2
        from workflows.sbi.e2e_conditional_reference_math import prior
        _, _, radius = prior(4)
        out=highest_shell_null(np.eye(64),radius,128)
        df=127*out['modes']
        self.assertAlmostEqual(out['standard_deviation'],(2/df)**.5,places=12)
        np.testing.assert_allclose(out['central_99_interval'],chi2.ppf([.005,.995],df)/df,atol=.006)

    def test_cfm_risk_floor(self):
        import math
        self.assertAlmostEqual(cfm_loss_floor([1.,1.]),math.pi/2)
        self.assertAlmostEqual(cfm_loss_floor([.01,.25,1.,4.]),math.pi/2*(.1+.5+1+2)/4)
        with self.assertRaises(ValueError): cfm_loss_floor([0.,1.])


if __name__ == '__main__':
    unittest.main()
