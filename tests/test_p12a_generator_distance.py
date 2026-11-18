import copy
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from workflows.sbi.prepare_p12a_generator_distance import check_ini, EXPECTED_INI, COMMIT, CLASS_VERSION, CLONE_ADDITIONS, NBODYKIT_COMMIT, CLASSYLSS_TAG, CLASSYLSS_PATCHES
from workflows.sbi.audit_p12a_lowz_native_hosts import qualified_filter_radius


class GeneratorDistanceTests(unittest.TestCase):
    def test_exact_generator_parameters_not_effective_mass_substitute(self):
        text='\n'.join(f'{k} = {v}' for k,v in EXPECTED_INI.items())
        self.assertEqual(check_ini(text),EXPECTED_INI)
        with self.assertRaises(ValueError):
            check_ini(text.replace('omega_ncdm = 0.0006442','m_ncdm = 0.06'))
        with self.assertRaises(ValueError):
            check_ini(text+'\nh = 0.6736')

    def test_scalar_rounding_can_change_an_entire_index_suffix(self):
        import numpy as np
        from workflows.sbi.audit_p12a_lowz_native_hosts import native_filter_rows
        positions=np.array([[-999.,-1000.,-1000.],[-990.,-1000.,-1000.],
                            [-998.,-1000.,-1000.]],dtype='f4')
        np.testing.assert_array_equal(native_filter_rows(positions,[10.,10.,10.],10.),[0,2])
        np.testing.assert_array_equal(native_filter_rows(positions,[10.,10.,10.],10.001),[0,1,2])

    def test_certificate_rejects_fit_wrong_parameters_and_changed_source(self):
        record=dict(schema='p12a-generator-distance-v1',parameters=EXPECTED_INI,
            generator_commit=COMMIT,class_version=CLASS_VERSION,catalogue_data_used=False,
            clone_additions=CLONE_ADDITIONS,nbodykit_commit=NBODYKIT_COMMIT,
            classylss_tag=CLASSYLSS_TAG,classylss_patches=list(CLASSYLSS_PATCHES),
            normalized_ini=dict(path='normalized_ini',sha256='same'),
            fitted_radius=False,measurement=dict(redshift=.16,h=.6736,omega_ncdm=.0006442,
                Omega_Lambda=0.,Omega_fld=.7,distance_mpc_h=461.,conformal_difference_mpc_h=461.),solver_selection='test',
            sources=[dict(path='source',sha256='same')],helper=dict(path='helper',sha256='same'),
            binary=dict(path='binary',sha256='same'),preparer=dict(path='preparer',sha256='same'))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'certificate.json';p.write_text(json.dumps(record))
            with patch('workflows.sbi.audit_p12a_lowz_native_hosts.sha256',return_value='same'):
                self.assertEqual(qualified_filter_radius(p)[0],461.)
                for field,value in [('fitted_radius',True),('catalogue_data_used',True),
                                    ('parameters',{}),('generator_commit','wrong'),('clone_additions',{})]:
                    bad=copy.deepcopy(record);bad[field]=value;p.write_text(json.dumps(bad))
                    with self.assertRaises(ValueError):qualified_filter_radius(p)
            p.write_text(json.dumps(record))
            with patch('workflows.sbi.audit_p12a_lowz_native_hosts.sha256',return_value='changed'):
                with self.assertRaises(ValueError):qualified_filter_radius(p)


if __name__=='__main__':unittest.main()
