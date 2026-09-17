"""Transactional/coherence checks without a scientific GPU run."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from workflows.sbi.e2e_durable import publish_json
from workflows.sbi.e2e_vdm_context_models import FineCondition,block_mean
from workflows.sbi.e2e_vdm_context_sample import SharedParents,with_coarse,case
from workflows.sbi.e2e_vdm_context_tasks import draw_tasks
from workflows.sbi.e2e_vdm_context_sample import selected_tasks


class Observations:
    targets=False
    chart=dict(coarse=dict(mean=0.,std=1.),fine=dict(mean=0.,std=1.),residual=dict(std=.2))
    rows={'domain':{},'child':dict(core_offset_raw=[32,0,0])}
    def condition(self,anchor,device='cpu'):
        return FineCondition(torch.zeros(1,24,48,48,48,device=device),
                             torch.zeros(1,12,48,48,48,device=device),torch.zeros(1,3,device=device))
    def raw_targets(self,*args,**kwargs):
        raise AssertionError('normal sampler opened truth')


class SamplingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.task=dict(arm='D',replica=0,checkpoint=20480,anchor='domain',domain='domain',
                      start=0,count=16,purpose='joint',steps=250,coarse_mode='sampled',task_id='test')

    def test_shared_cache_request_order_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            publish_json(root/'MANIFEST.json',dict(test=True))
            parent=SharedParents(root,Observations(),None,'model-hash','cpu')
            def draw(model,condition,steps,seeds):
                return torch.stack([torch.full((1,48,48,48),(s%100)/100) for s in seeds])
            with patch('workflows.sbi.e2e_vdm_context_sample.coupled_sample',side_effect=draw) as mock:
                first=parent.sampled(self.task,[2,0])
                second=parent.sampled(dict(self.task,anchor='child'),[0,2])
                self.assertEqual(mock.call_count,1)
                torch.testing.assert_close(first,second.flip(0),atol=0,rtol=0)
                self.assertEqual(len(list((root/'parents').rglob('*.json'))),1)
                filename=next((root/'parents').rglob('*.npz'))
                with filename.open('ab') as stream:
                    stream.write(b'corruption')
                with self.assertRaises(ValueError):
                    parent.sampled(self.task,[0])

    def test_oracle_is_explicit_and_source_checked(self):
        parent=SharedParents(Path('/tmp'),Observations(),None,'model','cpu')
        with self.assertRaises(PermissionError):
            parent.get(dict(self.task,coarse_mode='oracle_diagnostic'),[0])
        bad=Observations()
        bad.targets=True
        with self.assertRaises(PermissionError):
            SharedParents(Path('/tmp'),bad,None,'model','cpu')

    def test_conditioned_mass_and_resumable_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            publish_json(root/'MANIFEST.json',dict(test=True))
            observations=Observations()
            coarse=torch.ones(8,1,48,48,48,dtype=torch.float64)*2
            cond=with_coarse(observations.condition('child'),coarse,observations.chart,[32,0,0],'sampled')
            torch.testing.assert_close(cond.coarse_local,torch.full((8,1,48,48,48),np.log(2)),atol=1e-6,rtol=0)
            class Parents:
                sha='coarse-hash'
                def get(self,task,ids):
                    return coarse[:len(ids)]
            model=torch.nn.Linear(1,1)
            with patch('workflows.sbi.e2e_vdm_context_sample.coupled_sample',return_value=torch.zeros(8,1,48,48,48)) as mock:
                case(root,self.task,model,'fine-hash',observations,Parents(),limit=8)
                first=root/'draws/test/000000.json'
                original=first.read_bytes()
                self.assertFalse((root/'draws/test/COMPLETE.json').exists())
                case(root,self.task,model,'fine-hash',observations,Parents())
                self.assertEqual(mock.call_count,2)
                self.assertEqual(first.read_bytes(),original)
                self.assertTrue((root/'draws/test/COMPLETE.json').exists())
                with np.load(next((root/'draws/test').glob('000000-*.npz'))) as saved:
                    np.testing.assert_array_equal(saved['delta'],1)


if __name__=='__main__':
    unittest.main()
