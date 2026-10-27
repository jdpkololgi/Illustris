import copy
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workflows/abacus_tweb'))
from galaxy_label_equivalence import validate_equivalence
from galaxy_position_validation_status import ambiguity_status,independent_status

class EquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.record=dict(candidate_search_exhaustive=True,candidate_source_rows=[10,11],
            candidate_positions=[[100.1,100.1,100.1],[100.102,100.1,100.1]],
            query_position=[100.1001,100.1,100.1],candidate_eigenvalues=[[-1,0,1],[-1,0,1]],query_eigenvalues=[-1,0,1])
    def test_distinct_positions_identical_targets(self):
        self.assertTrue(validate_equivalence(self.record,.005))
    def test_same_labels_different_cell_rejected(self):
        r=copy.deepcopy(self.record);r['query_position']=[.97655,0,0];r['candidate_positions']=[[.97654,0,0],[.97657,0,0]]
        with self.assertRaises(ValueError):validate_equivalence(r,.005)
    def test_differing_labels_rejected(self):
        r=copy.deepcopy(self.record);r['candidate_eigenvalues'][1][0]=-.99
        with self.assertRaises(ValueError):validate_equivalence(r,.005)
    def test_missing_or_incomplete_candidates_rejected(self):
        for key,value in [('candidate_source_rows',[10,10]),('candidate_search_exhaustive',False),('candidate_positions',[]),('query_eigenvalues',[float('nan'),0,1])]:
            r=copy.deepcopy(self.record);r[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_equivalence(r,.005)
    def test_distant_same_cell_rejected(self):
        r=copy.deepcopy(self.record);r['candidate_positions'][1][0]+=.01
        with self.assertRaises(ValueError):validate_equivalence(r,.005)
    def test_identity_flag_does_not_promote_independent_code(self):
        ids=np.array([10,20,30]);unique=(np.array([10]),np.array([3],dtype='u1'))
        np.testing.assert_array_equal(independent_status(ids,unique),[3,0,0])
        np.testing.assert_array_equal(ambiguity_status(ids,np.array([20])),[0,1,0])

if __name__=='__main__':unittest.main()

class JoinFlagTests(unittest.TestCase):
    def test_join_preserves_ambiguity_without_masking_labels(self):
        from p10_build_observed_truth import output_dtype,join_successful_chunk,LSS_COLUMNS
        parent=np.zeros(1,dtype=output_dtype(True,True));parent['TARGETID']=1
        parent['RA']=10;parent['DEC']=20;parent['Z_COSMO']=.2;parent['RES']=0;parent['CEN']=1
        parent['HAS_LABEL']=parent['TWEB_STATUS']=1;parent['ID_AMBIG']=1
        parent['LAMBDA1']=-1;parent['LAMBDA2']=0;parent['LAMBDA3']=1;parent['CWEB']=1
        lss=np.zeros(1,dtype=[(n,'f8') for n in LSS_COLUMNS]);lss['TARGETID']=1
        lss['RA']=10;lss['DEC']=20;lss['Z_not4clus']=.2
        out,_=join_successful_chunk(lss,parent)
        self.assertEqual(out['ID_AMBIG'].tolist(),[1]);self.assertEqual(out['INDEP_POS'].tolist(),[0]);self.assertEqual(out['HAS_LABEL'].tolist(),[1])

class CertificateTests(unittest.TestCase):
    def fixture(self):
        from galaxy_label_equivalence import RULE_VERSION
        record=dict(original_row=5,population='unresolved_central',slab=33,
                    candidate_search_exhaustive=True,candidate_source_rows=[10,11],
                    candidate_positions=[[100.1]*3,[100.102,100.1,100.1]],
                    query_position=[100.1]*3,candidate_eigenvalues=[[-1,0,1],[-1,0,1]],query_eigenvalues=[-1,0,1])
        checks=[dict(slab=slab,population=pop,rows=1,accepted=0 if j==3 else 1,
                     pass_gate=j!=3,pass_label_gate=True,label_equivalent_ambiguous=int(j==3))
                for j,(slab,pop) in enumerate([(0,'resolved_central'),(0,'unresolved_central'),(33,'resolved_central'),(33,'unresolved_central')])]
        info=dict(cubic={str(k):dict(rows=1,identity_matches=1,within_tolerance=1) for k in range(2)},
                  native_checks=checks,field_gate=True,ambiguity_rule_version=RULE_VERSION,label_equivalence_records=[record])
        data=dict(original_row=np.arange(6),reference_kind=np.array(['cubic_galaxy','cubic_galaxy','resolved_central','unresolved_central','resolved_central','uncovered']),
                  reconstructed_native=np.full((6,3),100.1),independent_native=np.full((6,3),100.1),
                  label_equivalent_ambiguity=np.array([False]*5+[True]),CEN=np.array([0,1,1,1,1,1]),
                  RES=np.array([1,1,1,0,1,0]),FILE_NUM=np.array([10,10,0,0,33,33]),BOX_INDEX=np.array([0,0,-1,-1,-1,-1]))
        return info,data
    def test_certificate_retains_unique_only_codes(self):
        from certify_galaxy_position_scope import validate_phase_evidence
        info,data=self.fixture();rows,codes,ambiguous,counts=validate_phase_evidence(info,data,.005)
        self.assertEqual(rows.tolist(),[0,1,2,3,4]);self.assertEqual(ambiguous.tolist(),[5]);self.assertEqual(counts,{1:2,2:2,3:1})
    def test_missing_equivalence_record_fails(self):
        from certify_galaxy_position_scope import validate_phase_evidence
        info,data=self.fixture();info['label_equivalence_records']=[]
        with self.assertRaises(ValueError):validate_phase_evidence(info,data,.005)
    def test_wrong_source_coordinate_fails(self):
        from certify_galaxy_position_scope import validate_phase_evidence
        info,data=self.fixture();data['reconstructed_native'][5,0]+=.00001
        with self.assertRaises(ValueError):validate_phase_evidence(info,data,.005)
