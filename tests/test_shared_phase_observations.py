import copy
import unittest
from pathlib import Path
from workflows.sbi.prepare_shared_phase_observations import validate_parent_record

class ParentReuseTests(unittest.TestCase):
    def test_targets_wrong_phase_and_wrong_sources_rejected(self):
        sources={'cutsky':Path('/tmp/cut'),'forfa':Path('/tmp/fa')}
        record={'phase':'ph004','target_truth_present':False,'parity':{'row_identity_verified':True},
                'cutsky':{'path':'/tmp/cut'},'forfa':{'path':'/tmp/fa'}}
        validate_parent_record(record,'ph004',sources,['TARGETID','RA','DEC','Z_COSMO'])
        for column in ('LAMBDA1','CWEB','TWEB_STATUS'):
            with self.assertRaises(ValueError):validate_parent_record(record,'ph004',sources,[column])
        with self.assertRaises(ValueError):validate_parent_record(record,'ph005',sources,[])
        bad=copy.deepcopy(record);bad['cutsky']['path']='/tmp/wrong'
        with self.assertRaises(ValueError):validate_parent_record(bad,'ph004',sources,[])
        bad=copy.deepcopy(record);bad['target_truth_present']=True
        with self.assertRaises(ValueError):validate_parent_record(bad,'ph004',sources,[])

if __name__=='__main__':unittest.main()
