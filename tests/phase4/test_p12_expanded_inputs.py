import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import numpy as np
from workflows.sbi import p12_prepare_base_response_dataset as data
from workflows.sbi.p12_export_unet_summaries import terminal_export_checkpoint
from workflows.sbi.p12_train_base_response_fmpe import spatial_group_keys
from workflows.sbi.p12a_prepare_response_inputs import visible_roles
from workflows.abacus_tweb.p10_training_contract import sha256


class ExpandedInputTests(unittest.TestCase):
    def test_spatial_ids_include_phase_and_cap(self):
        rows={'phase_index':np.array([15,16,15]), 'cap':np.array([0,0,0]), 'superblock_id':np.array([4,4,4])}
        a=spatial_group_keys(rows,np.array([0])); b=spatial_group_keys(rows,np.array([1])); c=spatial_group_keys(rows,np.array([2]))
        self.assertEqual(np.intersect1d(a,b).size,0)
        self.assertEqual(np.intersect1d(a,c).size,1)

    def test_pipeline_roots_roles_and_reserved_refusal(self):
        names=['TRAIN_PHASES','VALIDATION_PHASES','FULL_CONTRACT','CONTRACT_ROOT','SUMMARY_ROOT',
               'RANDOM_RESPONSE_CONTRACT','RANDOM_RESPONSE_CACHE','PIPELINE_CONFIG','PIPELINE_CONFIG_SHA256','RESERVED_PHASES']
        previous={name:getattr(data,name) for name in names}
        try:
            with TemporaryDirectory() as tmp:
                root=Path(tmp); marker=root/'TRAINING_LOADER_READY.json'
                roles={'training':['ph000','ph007'],'validation_and_selection':['ph006','ph012'],
                       'sealed_blind_test':['ph001'],'reserved_confirmation':['ph014']}
                marker.write_text(json.dumps({'pass':True,'roles':roles}))
                config={'full_contract':str(root),'full_contract_sha256':sha256(marker),
                        'crossfit_contracts':str(root/'folds'),'summaries':str(root/'summaries'),
                        'response_contract':str(root/'response'),'response_cache':str(root/'cache'),
                        'calibration_folds':[0,1],'evaluation_folds':[2,3,4]}
                path=root/'config.json';path.write_text(json.dumps(config)); data.configure_pipeline(path)
                self.assertEqual(data.phase_contract_root('ph012'),root)
                self.assertEqual(data.phase_contract_root('ph007'),root/'folds/omit_ph007')
                with self.assertRaises(PermissionError): data.load_phase_sample('ph014',10,0)
                self.assertEqual(visible_roles(root)[1],roles['training']+roles['validation_and_selection'])
                config['full_contract_sha256']='wrong';path.write_text(json.dumps(config))
                with self.assertRaises(ValueError): data.configure_pipeline(path)
        finally:
            for name,value in previous.items(): setattr(data,name,value)

    def test_terminal_export_requires_completed_fixed_budget_and_bound_inputs(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'transforms/field').mkdir(parents=True)
            paths=['TRAINING_LOADER_READY.json','transforms/target_scaler.json','transforms/field/field_transform.json']
            for name in paths: (root/name).write_text('{}')
            loader=SimpleNamespace(root=root, training_phases=('ph000','ph007'),validation_phases=('ph024',),target_scaler={},field_normalization={})
            state={'schema_version':'p10-arm-a-checkpoint-v1','model':'unet','frozen_arguments':{'epochs':2,'disable_early_stopping':True},
                   'global_step':6,'resume':{'epoch':2,'cursor':3},'source_contract':{'a':'hash'},'model_state':{}}
            completion={'epochs_completed':2,'global_steps':6,'training_cores_per_epoch':3,'training_phases':['ph000','ph007'],'source_contract':{'a':'hash'}}
            for key,name in zip(['training_ready_marker_sha256','target_scaler_sha256','field_transform_sha256'],paths):completion[key]=sha256(root/name)
            result=terminal_export_checkpoint(state,completion,loader,2)
            self.assertEqual(result['selection_policy'],'terminal_fixed_budget')
            state['resume']['cursor']=2
            with self.assertRaises(RuntimeError):terminal_export_checkpoint(state,completion,loader,2)
            state['resume']['cursor']=3;(root/paths[1]).write_text('{"changed":true}')
            with self.assertRaises(RuntimeError):terminal_export_checkpoint(state,completion,loader,2)

class ParentResponseInputTests(unittest.TestCase):
    def test_parent_response_reorders_by_identity_and_rejects_corruption(self):
        from unittest.mock import patch
        with TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'folds/omit_ph000/phases/ph000').mkdir(parents=True)
            record=root/'folds/omit_ph000/phases/ph000/phase_contract.json'
            record.write_text(json.dumps({'inputs':{'p1_manifest_sha256':'p1'}}))
            distance=root/'distance.npy';support=root/'support.npy'
            np.save(distance,np.array([10.,np.nan,30.],dtype=np.float32));np.save(support,np.array([True,False,True]))
            marker=root/'parent.json';marker.write_text(json.dumps({'phase':'ph000','pass':True,'rows':3,
                'p1_manifest_sha256':'p1','arrays':{'distance':str(distance),'support':str(support)},
                'array_sha256':{'distance':sha256(distance),'support':sha256(support)}}))
            (root/'adapter_inventory.json').write_text(json.dumps({'phases':{'ph000':{
                'parent_response':str(marker),'parent_response_sha256':sha256(marker)}}}))
            with patch.object(data,'RANDOM_RESPONSE_CONTRACT',root), patch.object(data,'CONTRACT_ROOT',root/'folds'), patch.object(data,'PIPELINE_CONFIG',None):
                d,s,_=data.random_response_for_phase('ph000',np.array([2,0]),'parent-order')
                np.testing.assert_array_equal(d,[30.,10.]);self.assertTrue(s.all())
                np.save(distance,np.array([99.,np.nan,30.],dtype=np.float32))
                with self.assertRaises(RuntimeError):data.random_response_for_phase('ph000',np.array([2,0]),'parent-order')


if __name__ == '__main__':unittest.main()
