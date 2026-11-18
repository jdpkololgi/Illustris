import copy
import unittest
from workflows.sbi.prepare_shared_phase_sources import validate_native_manifest

class NativeReuseTests(unittest.TestCase):
    def fixture(self):
        return dict(phase='ph004',preflight_only=False,
            build=dict(processed_file_count=136,max_files_per_directory=None,ngrid=2048,
                       boxsize_mpc_h=2000.,dtype='float32',particle_count=100000,
                       relative_count_error=1e-7),
            target_contract=dict(mass_assignment='TSC',redshift=.2,cosmology='c000',
                                 particle_subsamples={'total_fraction':.1}))

    def test_full_native_contract_required(self):
        value=self.fixture();validate_native_manifest(value,'ph004')
        for key,replacement in [('processed_file_count',135),('ngrid',1024),
                                ('max_files_per_directory',1),('relative_count_error',.01),
                                ('dtype','float64')]:
            bad=copy.deepcopy(value);bad['build'][key]=replacement
            with self.assertRaises(ValueError):validate_native_manifest(bad,'ph004')
        with self.assertRaises(ValueError):validate_native_manifest(value,'ph005')
        value['preflight_only']=True
        with self.assertRaises(ValueError):validate_native_manifest(value,'ph004')

    def test_particle_fraction_and_assignment_required(self):
        value=self.fixture();value['target_contract']['particle_subsamples']['total_fraction']=.03
        with self.assertRaises(ValueError):validate_native_manifest(value,'ph004')
        value=self.fixture();value['target_contract']['mass_assignment']='CIC'
        with self.assertRaises(ValueError):validate_native_manifest(value,'ph004')

if __name__=='__main__':unittest.main()
