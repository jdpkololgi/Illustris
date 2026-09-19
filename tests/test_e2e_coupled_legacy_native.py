import unittest
from workflows.sbi import e2e_coupled_legacy_native as legacy


class LegacyNativeTests(unittest.TestCase):
    def test_legacy_import_cannot_bypass_fresh_phase_zero_build(self):
        with self.assertRaises(PermissionError): legacy.validate_legacy_record('ph000',{}, {},0)
        with self.assertRaises(PermissionError): legacy.legacy_paths('ph001')
        with self.assertRaises(PermissionError): legacy.legacy_paths('ph007')

    def test_adoption_requires_exact_count_and_contract(self):
        manifest=dict(phase='ph002',build=dict(processed_file_count=136,ngrid=2048,
            boxsize_mpc_h=2000.,particle_count=100),
            target_contract=dict(redshift=.2,cosmology='c000',mass_assignment='TSC',particle_subsamples=dict(total_fraction=.1)))
        self.assertEqual(legacy.validate_legacy_record('ph002',manifest,{'phase':'ph002'},100),manifest['build'])
        with self.assertRaises(ValueError):
            legacy.validate_legacy_record('ph002',manifest,{'phase':'ph002'},101)
        manifest['build']['processed_file_count']=135
        with self.assertRaises(ValueError):
            legacy.validate_legacy_record('ph002',manifest,{'phase':'ph002'},100)

    def test_assignment_is_read_from_the_actual_legacy_target_contract(self):
        manifest=dict(phase='ph002',build=dict(processed_file_count=136,ngrid=2048,
            boxsize_mpc_h=2000.,particle_count=100),target_contract=dict(redshift=.2,
            cosmology='c000',mass_assignment='CIC',particle_subsamples=dict(total_fraction=.1)))
        with self.assertRaises(ValueError):
            legacy.validate_legacy_record('ph002',manifest,{'phase':'ph002'},100)


if __name__=='__main__': unittest.main()
