from pathlib import Path
import tempfile
import unittest
import torch
from workflows.sbi import e2e_coupled_gpu_benchmark as bench
from workflows.sbi import e2e_coupled_benchmark_models as models
from workflows.sbi import e2e_vdm_context_models as legacy


class CoupledGPUBenchmarkTests(unittest.TestCase):
    def test_transfer_tree_preserves_condition_and_duplicates_shared_references(self):
        one=torch.ones(1,1,4,4,4)
        condition=models.FieldCondition(one,one,torch.zeros(1,3),'joint',one,one,'training_truth')
        original=[(one,condition),(one,condition)]
        calls=[]
        def copy(value):
            calls.append(value.numel()*value.element_size())
            return value.clone()
        copied=bench.map_tensors(original,copy)
        self.assertEqual(len(calls),12)
        self.assertIsInstance(copied[0][1],models.FieldCondition)
        self.assertEqual(copied[0][1].coarse_source,'training_truth')
        self.assertTrue(torch.equal(copied[0][0],one))
        self.assertNotEqual(copied[0][0].data_ptr(),copied[1][0].data_ptr())
        self.assertNotEqual(copied[0][1].joint.data_ptr(),copied[0][1].wide.data_ptr())
        self.assertIsNone(bench.map_tensors(None,copy))

    def test_seven_factors_and_two_parent_exposure(self):
        cfg=bench.config(); counts={}
        for name in bench.CASES:
            model,opt,rng,presentations,kind=bench.make_case(name,cfg,'meta')
            expected=2 if name in ('D_fine','I_fine') else 1
            self.assertEqual(len(presentations),expected)
            self.assertEqual(kind,'cfm' if name.startswith('CFM') else 'vdm')
            counts[name]=sum(p.numel() for p in model.parameters())
            for target,condition in presentations:
                if isinstance(model,models.CoupledBackbone):
                    condition.validate(target,model.stage,model.domain,True)
                    bench.inference_condition(condition).validate(target,model.stage,model.domain,False)
                elif isinstance(model,legacy.ContextVDM):
                    condition.validate(target,'D',True)
                    bench.inference_condition(condition).validate(target,'D',False)
                else: self.assertEqual(condition.shape,(2,12,48,48,48))
        self.assertEqual(counts['I_fine'],counts['J_fine'])
        self.assertEqual(counts['J_fine'],counts['CFM_fine'])
        self.assertEqual(counts['IJ_coarse'],counts['CFM_coarse'])

    def test_inference_removes_synthetic_training_parent(self):
        one=torch.ones(2,1,4,4,4)
        c=legacy.FineCondition(torch.ones(2,24,4,4,4),torch.ones(2,12,4,4,4),
                               torch.ones(2,3),one,one,'training_truth')
        actual=bench.inference_condition(c,True)
        self.assertEqual(actual.coarse_source,'fixed_mean')
        self.assertEqual(actual.local.shape[0],1)
        self.assertEqual(int(actual.coarse_wide.count_nonzero()),0)
        self.assertEqual(int(actual.coarse_local.count_nonzero()),0)
        self.assertEqual(int(c.coarse_local.count_nonzero()),128)

    def test_checkpoint_publication_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'TECHNICAL_INITIAL.pt'
            state={'model':{'a':torch.tensor([1.])},'step':2}
            result=bench.atomic_state(path,state)
            self.assertEqual(result['bytes'],path.stat().st_size)
            actual=torch.load(path,weights_only=True)
            self.assertTrue(bench.equal_state(actual,state))
            with self.assertRaises(FileExistsError): bench.atomic_state(path,{'step':99})
            self.assertTrue(bench.equal_state(torch.load(path,weights_only=True),state))


if __name__=='__main__': unittest.main()
