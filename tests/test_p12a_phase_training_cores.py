import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from workflows.sbi.p12a_phase_training_cores import core_arrays
from workflows.abacus_tweb.p10_training_contract import P10PhaseBalancedLoader


class PhaseTrainingCoresTest(unittest.TestCase):
    def test_exact_core_ownership_without_spatial_folds(self):
        size=64/.6766
        points=np.array([[1.,1,1,0],[size,1,1,0],[size+.1,1,1,0],[2.,1,1,0]])
        cap=np.array([0,0,0,1]);shell=np.array([0,1,1,0])
        active=np.ones(4,bool);eligible=np.array([1,1,0,1],bool)
        grid=dict(origin_mpc=[0,0,0],cell_mpc=5.,shape=[100,100,100])
        cores,assignment=core_arrays(points,cap,shell,active,active,eligible,np.arange(4),{'SGC':grid,'NGC':grid})
        self.assertEqual(len(cores['core_id']),3)
        self.assertEqual(cores['active_count'].sum(),3)
        self.assertTrue(np.all(cores['fold']==-1))
        self.assertNotEqual(assignment['core_id'][0],assignment['core_id'][1])
        self.assertEqual(assignment['core_id'][1],assignment['core_id'][2])
        np.testing.assert_array_equal(assignment['periodic_duplicate_image'],~eligible)

    def test_candidate_is_not_public_readiness(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            record=dict(roles=dict(training=['ph000'],validation_and_selection=['ph006'],sealed_blind_test=['ph018']),**{'pass':True})
            (root/'TRAINING_LOADER_CANDIDATE.json').write_text(json.dumps(record))
            for phase in ('ph000','ph006'):
                path=root/'phases'/phase;path.mkdir(parents=True)
                (path/'phase_contract.json').write_text('{}')
            with self.assertRaises(RuntimeError):P10PhaseBalancedLoader(root)
            with P10PhaseBalancedLoader(root,allow_preparation=True) as loader:
                self.assertEqual(loader.training_phases,('ph000',))

    def test_matched_field_dispatch(self):
        loader=P10PhaseBalancedLoader.__new__(P10PhaseBalancedLoader)
        loader.root=Path('/unused');loader.manifest={'field_adapter':'matched_cic'};loader._field={}
        with patch('workflows.abacus_tweb.p12a_matched_field_adapter.MatchedFieldPatchAdapter') as adapter:
            self.assertIs(loader.field_adapter('ph001'),adapter.return_value)
            adapter.assert_called_once()


if __name__=='__main__':unittest.main()
