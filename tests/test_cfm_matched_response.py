import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import fitsio
import h5py
import numpy as np

from workflows.sbi import build_cfm_matched_response as build
from workflows.sbi.prepare_cfm_lowz_inputs import sha256


class MatchedResponseTests(unittest.TestCase):
    def test_full_writer_and_immutable_links(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); phase='ph001'
            plan=dict(output_root=str(root),context_redshift=[0.,.6],excluded_sentinel_redshift=[.585,.595],
                      source_campaign={'sha256':'source'})
            policy=dict(training=[phase],development_and_calibration=['ph006'],final_test=['ph018','ph019'])
            raw=root/'raw.h5'
            with h5py.File(raw,'w') as f:
                f.attrs.update(phase=phase,cap='SGC',origin_mpc_h=[-20.,-20.,-20.],cell_mpc_h=40.)
                f['counts']=np.zeros((4,4,4),dtype='f4')
                f['support_random']=np.ones((4,4,4),dtype='u1')
                f['angular_response']=np.ones((4,4,4),dtype='f4')
            receipt=root/'responses_full'/phase/'RESPONSE_PREPARED.json'
            receipt.parent.mkdir(parents=True)
            receipt.write_text(json.dumps(dict(phase=phase,label_free=True,outputs=[dict(path=str(raw),sha256=sha256(raw),bytes=raw.stat().st_size)])))
            selection=root/'point_selection_candidates_v1/full.json';selection.parent.mkdir()
            selection.write_text(json.dumps(dict(schema='lowz-point-selection-candidate-v1',density_unit='(Mpc/h)^-3',
                fit_phases=[phase],omitted_phase=None,bin_edges=[0.,.6],density=[[.001],[.001]],valid=[[True],[True]])))
            observed=root/'observed.fits'
            rows=np.zeros(2,dtype=[('RA','f8'),('DEC','f8'),('Z','f8')]); rows['Z']=[.01,-.01]
            fitsio.write(observed,rows)
            angular=root/'angular.npz';n=12*256**2
            np.savez(angular,support=np.ones(n,bool),domain=np.zeros(n,'u1'),angular_response=np.ones(n,'f4'))
            def source(root,ph,relative,*args):
                return (angular if relative.startswith('angular/') else observed), dict(test=True)
            with patch.dict('os.environ', {'SLURM_JOB_ID':'synthetic-test'}),patch.object(build,'require_compute'),patch.object(build.lowz,'load_plan',return_value=(plan,policy,{'root':str(root)})),patch.object(build.lowz,'source_payload',side_effect=source):
                build.build(phase,['SGC'],1)
            marker=root/'responses_matched_cic_v1'/phase/'SGC_COMPLETE.json'
            record=json.loads(marker.read_text())
            self.assertEqual(record['count_source_audit']['supported_source_rows'],1)
            self.assertFalse(record['model_inputs_ready'])
            with h5py.File(record['output']['path']) as f:
                self.assertAlmostEqual(float(f['counts'][:].sum()),1.,places=6)
                self.assertIsInstance(f.get('support_random',getlink=True),h5py.ExternalLink)
                self.assertTrue(np.isfinite(f['expected_counts_random'][:]).all())
            self.assertEqual(sha256(raw),record['external_source']['sha256'])


if __name__=='__main__':unittest.main()
