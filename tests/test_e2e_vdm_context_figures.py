from pathlib import Path
from unittest.mock import patch
import unittest
import numpy as np

from tests.context_test_support import safe_temporary_directory
from workflows.sbi.e2e_durable import publish_json
from workflows.sbi.e2e_vdm_context_report import render_report
from workflows.sbi.e2e_vdm_context_data import read_json
from workflows.sbi.e2e_field_build_products import sha256


class FigureTests(unittest.TestCase):
    def test_complete_fixed_result_renders_bound_artifacts(self):
        fields=np.random.default_rng(3).normal(size=(8,48,48,48))*.1
        class Data:
            def raw_targets(self,anchor):
                return dict(rho=1+fields[0])
        rows=[]
        records=[]
        for arm in 'ABCD':
            for seed in (0,1):
                for step in (5120,10240,20480):
                    rows.append(dict(arm=arm,replica=seed,checkpoint=step,phase='ph004',
                        density_crps=.1,tidal_energy=.2,density_coverage90=.85,tidal_coverage90=.84,attainable90=59/65))
            records.append(dict(task=dict(arm=arm,replica=0,anchor='field',purpose='main',steps=250,checkpoint=20480),
                spectra=dict(truth_power=[1]*5,mean_sample_power=[.8]*5,posterior_mean_power=[.6]*5,
                             correlation_posterior_mean=[.7]*5)))
        result=dict(progression=rows,central_draws=33280,coarse_draws=7872,
            contrast_decisions={'H1_diversity':dict(relative_primary_gain=.12,passed=True)},limitations=['test only'])
        with safe_temporary_directory() as tmp:
            root=Path(tmp)
            (root/'analysis').mkdir()
            publish_json(root/'MANIFEST.json',{})
            publish_json(root/'DRAW_LEDGER.json',dict(panels=dict(refinement=['field'])))
            publish_json(root/'analysis/RESULTS.json',result)
            with patch('workflows.sbi.e2e_vdm_context_report.ensemble',return_value=(fields,{})):
                render_report(root,result,records,Data())
            proof=read_json(root/'analysis/FIGURES.json')
            self.assertEqual(proof['inputs_sha256'],sha256(root/'analysis/RESULTS.json'))
            for name,digest in proof['files'].items():
                self.assertEqual(sha256(root/'analysis'/name),digest)
            self.assertIn('12.00%',(root/'analysis/REPORT.md').read_text())


if __name__=='__main__':
    unittest.main()
