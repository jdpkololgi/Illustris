import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from workflows.sbi import e2e_wide_continue as c
from workflows.sbi.e2e_wide_evaluation_report import summarize


class ContinuationTests(unittest.TestCase):
    def test_state_equality_catches_optimizer_rng_changes(self):
        state={'tensor':torch.tensor([1.,2.]),'rng':np.array([3,4]),'history':[{'step':192}]}
        self.assertTrue(c.equal_state(state,copy.deepcopy(state)))
        for key in ('tensor','rng'):
            other=copy.deepcopy(state);other[key][0]+=1
            self.assertFalse(c.equal_state(state,other))

    def test_invalid_expansion_rejected_before_compute_or_writes(self):
        for start,stop in [(191,384),(192,385),(384,384)]:
            with self.assertRaises(ValueError):
                c.advance(None,None,None,{'step':start},None,None,None,stop,None)

    def test_contract_rejects_holdout_and_autoextension(self):
        spec=json.loads(c.CONTRACT_FILE.read_text())
        for key in ('heldout_access_authorized','automatic_extension'):
            changed={**spec,key:True}
            with patch.object(Path,'read_text',return_value=json.dumps(changed)):
                with self.assertRaises(ValueError):
                    c.contract({})

    def test_report_uses_registered_checkpoint_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            losses=[];refinements=[]
            for method in ('cfm','diffusion'):
                (root/f'a_{method}.json').write_text(json.dumps({'ensemble':{'observed':{'mean_pointwise_draw_std':[1,1,1]}}}))
                refinements.append({'method':method,'anchor_id':'a','masks':{'observed':{'eigen_rmse':[0,0,0]}}})
                for stage in ('coarse','fine'):
                    for phase in ('ph000','ph002','ph003'):
                        losses.append({'method':method,'stage':stage,'phase':phase,'losses':{'288':3.,'336':2.,'384':1.}})
            result=summarize({'registration':{'loss_checkpoints':[288,336,384]},'fixed_loss_probes':losses,
                'sampler_refinement':refinements,'summary':{},'by_phase':{},'by_shell_support':{}},root)
            for value in result['optimization'].values():
                self.assertEqual(value['means_288_336_384'],[3.,2.,1.])
                self.assertEqual(value['late_relative_improvement'],.5)


if __name__=='__main__':
    unittest.main()
