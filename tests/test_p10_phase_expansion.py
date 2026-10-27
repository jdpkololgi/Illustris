"""Role isolation and native-density compatibility for the expanded campaign."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'workflows/abacus_tweb'))
from p10_phase_assets import load_registry, validate_registry, RegistryError
from p10_expansion_preflight import density_contract

class ExpansionTests(unittest.TestCase):
    def test_original_registry_unchanged(self):
        self.assertEqual(len(load_registry(ROOT/'configs/p10_phase_registry_v1.json')['phases']),7)
    def test_expanded_roles(self):
        r=load_registry(ROOT/'configs/p10_phase_registry_v2.json')
        c=r['model_phase_contract']
        groups=[set(c[k]) for k in ('training','validation_and_selection','sealed_blind_test','reserved_confirmation')]
        self.assertEqual([len(g) for g in groups],[15,3,1,6])
        self.assertEqual(len(set.union(*groups)),25)
        for p in groups[2]|groups[3]:self.assertEqual(r['phases'][p]['role'],'sealed_blind')
    def test_reserved_phase_cannot_enter_training(self):
        r=load_registry(ROOT/'configs/p10_phase_registry_v2.json')
        r['model_phase_contract']['training'].append('ph018')
        with self.assertRaises(RegistryError):validate_registry(r)
    def test_reserved_access_cannot_drift(self):
        r=load_registry(ROOT/'configs/p10_phase_registry_v2.json')
        r['phases']['ph014']['truth_access']='development'
        with self.assertRaises(RegistryError):validate_registry(r)
    def test_wrong_density_rejected(self):
        good=dict(phase='ph007',ngrid=2048,box_mpc_h=2000.,dtype='float32',processed_files=136,
                  subsample_fraction=.1,relative_count_error=1e-7,particle_count=33022530364,
                  outputs=[dict(path='/native/counts_tsc_2048.npy',sha256='a'*64)],**{'pass':True})
        density_contract(good,'ph007')
        for key,value in [('phase','ph008'),('processed_files',68),('ngrid',256),('relative_count_error',float('nan')),('subsample_fraction',.03)]:
            bad=copy.deepcopy(good);bad[key]=value
            with self.assertRaises(ValueError):density_contract(bad,'ph007')

if __name__=='__main__':unittest.main()
