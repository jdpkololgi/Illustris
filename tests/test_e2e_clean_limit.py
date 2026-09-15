import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from workflows.sbi import e2e_durable as durable
from workflows.sbi import e2e_clean_limit as experiment
from workflows.sbi.e2e_multinoise_test import exposure
from workflows.sbi.e2e_clean_limit_launch import command, snapshot_paths


class DurableTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.model=torch.nn.Linear(2,1)
        self.optimizer=torch.optim.AdamW(self.model.parameters())
        self.generator=torch.Generator().manual_seed(17)
        self.binding={'unit_test':True}

    def tearDown(self):
        self.temp.cleanup()

    def save(self,step=0):
        return durable.save(self.root,model=self.model,optimizer=self.optimizer,generator=self.generator,
            binding=self.binding,stage='fine',method='diffusion',step=step,history=[])

    def test_save_load_and_orphan_ignored(self):
        receipt=self.save()
        (self.root/'checkpoint_999999_incomplete').mkdir()
        state,loaded=durable.load(self.root,self.binding)
        self.assertEqual(receipt,loaded);self.assertEqual(state['step'],0)
        for key,value in self.model.state_dict().items():self.assertTrue(torch.equal(value,state['model'][key]))

    def test_hash_corruption_fails_closed(self):
        receipt=self.save();path=self.root/receipt['path']
        with path.open('ab') as stream:stream.write(b'corrupt')
        with self.assertRaises(ValueError):durable.load(self.root,self.binding)

    def test_binding_and_pointer_mismatch(self):
        self.save()
        with self.assertRaises(ValueError):durable.load(self.root,{'wrong':True})
        pointer=json.loads((self.root/'LATEST.json').read_text());pointer['step']=1
        durable.publish_json(self.root/'LATEST.json',pointer,replace=True)
        with self.assertRaises(ValueError):durable.load(self.root,self.binding)

    def test_immutable_json_and_single_writer(self):
        durable.publish_json(self.root/'receipt.json',{'a':1})
        with self.assertRaises(FileExistsError):durable.publish_json(self.root/'receipt.json',{'a':2})
        with durable.single_writer(self.root):
            with self.assertRaises(BlockingIOError):
                with durable.single_writer(self.root):pass

    def test_resume_optimizer_and_rng(self):
        def update():
            self.optimizer.zero_grad();x=torch.randn(3,2,generator=self.generator)
            self.model(x).square().mean().backward();self.optimizer.step()
        update();self.save();update()
        expected=copy.deepcopy(self.model.state_dict())
        state,_=durable.load(self.root,self.binding)
        experiment.restore(state,self.model,self.optimizer,self.generator);update()
        for k,v in expected.items():self.assertTrue(torch.equal(v,self.model.state_dict()[k]))


class DesignTests(unittest.TestCase):
    def setUp(self):
        self.spec=experiment.read_spec()
        self.cfg=json.loads(experiment.d.CONFIG.read_text());self.cfg.update(self.cfg['replicates'][0])

    def test_control_schedule_unchanged(self):
        for i in range(3072,3300):
            self.assertEqual(experiment.schedule(i,self.cfg,self.spec,'control'),exposure(i,self.cfg))

    def test_only_registered_bin_changes(self):
        count=0;zero=0;minimum=1.;maximum=0.;fields=set()
        ids=list(range(15))
        for i in range(24576,36864):
            before=exposure(i,self.cfg);after=experiment.schedule(i,self.cfg,self.spec,'near_zero')
            if before[1] is None or (i//3)%6!=0:self.assertEqual(before,after)
            else:
                count+=1;self.assertTrue(0<=after[1]<=.05);zero+=after[1]==0
                if after[1]>0:minimum=min(minimum,after[1]);maximum=max(maximum,after[1])
                fields.add(experiment.d.field_for(i,15,ids))
        self.assertGreater(count,1500);self.assertGreater(zero,20)
        self.assertLess(minimum,2e-5);self.assertGreater(maximum,.049);self.assertEqual(len(fields),15)

    def test_slurm_dependencies_and_limits(self):
        for name in ('optimization','coverage','analysis'):
            args=command(Path('/tmp/example'),name,'12345')
            self.assertIn('--dependency=afterok:12345',args);self.assertIn('--no-requeue',args)
            self.assertIn('--qos='+('debug' if name=='analysis' else 'shared'),args)
            self.assertIn('--signal=USR1@180',args)
            self.assertNotIn('--qos=interactive',args)

    def test_snapshot_excludes_heldout_and_unrelated_artifacts(self):
        names=['workflows/sbi/example.py','configs/example.json','configs/example_ph001.json',
               'docs/evidence/ph001/receipt.json','data/target.npy','docs/e2e_clean_limit_20260915.md']
        self.assertEqual(snapshot_paths(names),[names[0],names[1],names[-1]])

    def test_shift_predictors_leave_phase_out(self):
        metadata=[];rows=[]
        for phase in range(3):
            for i in range(4):
                anchor=f'{phase}-{i}'
                stats={k:float(i+phase) for k in ('mean','std','skew','excess_kurtosis','q01','q99',
                    'redshift','observed_fraction','mean_galaxy_count','support_mean','angular_response_mean')}
                stats['std']=1.
                metadata.append(dict(anchor_id=anchor,stats=stats))
                rows.append(dict(anchor_id=anchor,phase=str(phase),kind='clean',ratio=.05,rms=float(i+phase)))
        prep=dict(metadata=metadata,selection=dict(transfer=metadata))
        result=experiment.shift_predictors(rows,prep)
        self.assertEqual(len(result),5);self.assertEqual(len(result['observation']['per_field']),12)
        self.assertTrue(np.isfinite(result['combined']['mse']))


if __name__=='__main__':unittest.main()
