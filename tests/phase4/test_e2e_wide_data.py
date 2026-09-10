"""Tiny array checks only: no production payload reads or normalization fit."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from workflows.sbi import e2e_wide_data as data
from workflows.sbi.e2e_field_regenerate_spectral import interpolate
from workflows.sbi.e2e_field_wide_coarse import local_coords


class WideDataTests(unittest.TestCase):
    def test_quintic_matches_builder_and_preserves_constant(self):
        rng = np.random.default_rng(17)
        coarse = rng.normal(size=(12,12,12)).astype('f4')
        expected = interpolate(coarse, local_coords({'span_fine_cells':48, 'factor':4},8), degree=5)
        np.testing.assert_allclose(data.coarse_to_fine(coarse,8), expected, atol=1e-6)
        np.testing.assert_allclose(data.coarse_to_fine(np.ones_like(coarse)[None],8), 1)

    def test_forbidden_phase_paths_before_open(self):
        for phase in ('ph001','ph004','ph005','ph006'):
            with self.assertRaises(PermissionError):
                data.WideResearchDataset('/nonexistent/'+phase)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'ph001').mkdir()
            (root/'alias').symlink_to(root/'ph001')
            with self.assertRaises(PermissionError):
                data.guarded_path(root/'alias')

    def test_compute_guard_before_payload_access(self):
        dataset = object.__new__(data.WideResearchDataset)
        with patch.object(data, 'require_compute', side_effect=RuntimeError('compute required')):
            with self.assertRaises(RuntimeError):
                dataset.raw(0)
            with self.assertRaises(RuntimeError):
                dataset.verify_payloads()
            with self.assertRaises(RuntimeError):
                data.fit_normalization(dataset, '/unused')

    def dataset(self):
        dataset = object.__new__(data.WideResearchDataset)
        dataset.normalization = {'targets': {'coarse': {'mean':2.,'std':3.},
                                             'fine': {'mean':-1.,'std':.5}},
                                 'coarse': {}, 'fine': {}}
        for level, names in [('coarse',data.COARSE_CHANNELS),('fine',data.LOCAL_CHANNELS)]:
            dataset.normalization[level] = {name:{'mean':1.,'std':2.}
                                             for name in names if name not in data.IDENTITY}
        return dataset

    def test_target_transform_is_invertible_not_per_parent_demean(self):
        dataset = self.dataset()
        x = np.array([1.,4.,7.])
        for level in ('coarse','fine'):
            np.testing.assert_allclose(dataset.inverse_target(dataset.normalize_targets(x,level),level),x)
        self.assertNotEqual(dataset.normalize_targets(x).mean(),0)

    def test_mask_channels_preserve_zero_missing_distinction(self):
        dataset = self.dataset()
        coarse = np.zeros((len(data.COARSE_CHANNELS),2,2,2),dtype='f4')
        i = data.COARSE_CHANNELS.index('geometry_valid_fraction')
        coarse[i,0] = 1
        item = {'coarse_condition':coarse, 'fine_condition':np.zeros((len(data.FINE_CHANNELS),2,2,2),dtype='f4')}
        result = dataset._normalize(item)
        np.testing.assert_array_equal(result['coarse_condition'][i,0],1)
        np.testing.assert_array_equal(result['coarse_condition'][i,1],0)

    def test_inference_uses_observation_only_branch(self):
        dataset = self.dataset()
        item = {'coarse_condition':np.zeros((12,2,2,2),dtype='f4'),
                'fine_condition':np.zeros((13,2,2,2),dtype='f4')}
        with patch.object(dataset,'_read',return_value=item) as read:
            result = dataset.inference_conditions(0)
        read.assert_called_once_with(0,include_targets=False)
        self.assertNotIn('coarse_target',result)
        np.testing.assert_array_equal(result['fine_condition'][-1],0)

    def test_negative_counts_rejected(self):
        with self.assertRaises(ValueError):
            data.transformed('counts',np.array([-1.]))

    def test_normalization_binds_dependency_sources(self):
        with patch.object(data, 'sha256', side_effect=lambda path: Path(path).name):
            original = data.normalization_source_hashes()
        self.assertIn('e2e_field_dataset.py', original)
        self.assertIn('e2e_field_regenerate_spectral.py', original)
        self.assertIn('e2e_field_wide_coarse.py', original)
        with patch.object(data, 'sha256', side_effect=lambda path:
                          'changed' if Path(path).name == 'e2e_field_dataset.py' else Path(path).name):
            modified = data.normalization_source_hashes()
        self.assertNotEqual(data._digest(original), data._digest(modified))


if __name__ == '__main__':
    unittest.main()
