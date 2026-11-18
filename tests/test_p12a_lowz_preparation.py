import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import healpy as hp
import numpy as np

from workflows.abacus_tweb.p10_build_phase_index import shell_and_masks
from workflows.abacus_tweb.p10_redshift_contract import (
    domain_from, require_matching_domains, shell_names, validate_domain,
)
from workflows.abacus_tweb.p3a_build_canonical_fields import GridSpec, field_block
from workflows.sbi.p12a_parent_response import sample_centres
from workflows.sbi.prepare_p12a_lowz_inputs import load_plan, make_registry, make_schemas
from workflows.sbi.prepare_p12a_lowz_inputs import prepare_phase

REPO = Path(__file__).resolve().parents[1]
PLAN = json.loads((REPO / 'configs/p12a_lowz_preparation_20261029.json').read_text())
DOMAIN = domain_from(PLAN)


class LowzPreparationTests(unittest.TestCase):
    def test_campaign_phase_guards_before_any_payload_io(self):
        plan, source, _ = load_plan()
        for phase, campaign in [('ph000', False), ('ph018', True), ('ph019', True), ('ph099', True)]:
            with patch('workflows.sbi.prepare_p12a_lowz_inputs.read') as read:
                with self.assertRaises(PermissionError):
                    prepare_phase(phase, plan, source, Path('/unused'), 0, campaign=campaign)
                read.assert_not_called()

    def test_membership_includes_lowz_positive_context_and_excludes_sentinel(self):
        z = np.array([-.01, 0, .005, .01, .099, .15, .25, .55, .585, .595, .60, np.nan])
        shell, active, context = shell_and_masks(z, np.ones(len(z), bool), DOMAIN)
        self.assertEqual(np.flatnonzero(active).tolist(), [3, 4, 5, 6])
        self.assertEqual(np.flatnonzero(context).tolist(), [2, 3, 4, 5, 6, 7, 9])
        self.assertEqual(shell[3:7].tolist(), [0, 0, 0, 1])

    def test_legacy_range_is_unchanged(self):
        _, active, context = shell_and_masks(np.array([.01, .1, .15, .55]), np.ones(4, bool))
        self.assertEqual(active.tolist(), [False, False, True, False])
        self.assertEqual(context.tolist(), [False, True, True, True])

    def test_invalid_targets_not_promoted_by_range(self):
        _, active, context = shell_and_masks(np.array([.02]), np.array([False]), DOMAIN)
        self.assertFalse(active[0]); self.assertTrue(context[0])

    def test_domain_validation_and_dependency_mismatch(self):
        self.assertEqual(shell_names(DOMAIN)[0], '0p01_0p25')
        with self.assertRaises(ValueError): require_matching_domains(PLAN, {})
        for key, value in [('science', [0, .55]), ('context', [.1, .6]),
                           ('shells', [[.01, .2], [.25, .35], [.35, .45], [.45, .55]]),
                           ('diagnostic_edges', [.01, .1, .05, .55])]:
            bad = copy.deepcopy(DOMAIN); bad[key] = value
            with self.assertRaises(ValueError): validate_domain(bad)

    def test_schemas_preserve_cell_and_physics_but_change_range(self):
        p3, p4 = make_schemas(PLAN)
        self.assertEqual(p3['grid']['cell_mpc'], 5.)
        self.assertEqual(p4['core_size_probe']['default_mpc_h'], 64.)
        self.assertEqual(p3['radial_support']['z_context'], [0., .6])
        self.assertEqual(require_matching_domains(p3, p4), DOMAIN)
        self.assertIsNone(p3['selection']['ntilde_path'])

    def test_plan_and_registry_preserve_sealed_phases(self):
        plan, source, policy = load_plan()
        registry = make_registry(plan, source)
        self.assertEqual(registry['model_phase_contract']['sealed_blind_test'], ['ph018', 'ph019'])
        self.assertEqual(len(policy['training']), 20)
        self.assertEqual(domain_from(registry), DOMAIN)
        self.assertNotEqual(registry['path_templates']['phase_output'], source['source_root'] + '/{phase}')

    def test_response_uses_registered_floor_and_origin_is_unsupported(self):
        component = {'grid': {'origin_mpc': [-.5, -.5, -.5], 'shape': [7, 1, 1],
                              'cell_mpc': 1., 'padding_mpc': 0.}}
        selection = {'redshift_contract': DOMAIN,
                     'cosmology': {'radius_grid_mpc': [0., 100.], 'redshift_grid': [0., 1.]}}
        support = np.ones(hp.nside2npix(1), bool)
        angle = np.ones(len(support), np.float32)
        points = np.array([[0., 0., 0.], [2., 0., 0.], [5., 0., 0.]])
        distances, valid = sample_centres(points, component, support, angle, selection)
        self.assertEqual(valid.tolist(), [False, True, True])
        np.testing.assert_array_equal(distances, [0., 2., 5.])
        old = copy.deepcopy(selection); del old['redshift_contract']
        self.assertFalse(sample_centres(points, component, support, angle, old)[1].any())
        bad = copy.deepcopy(selection); bad['cosmology']['redshift_grid'] = [.1, .6]
        with self.assertRaises(ValueError): sample_centres(points, component, support, angle, bad)

    def test_deferred_fields_never_use_old_selection_and_mask_observer(self):
        p3, _ = make_schemas(PLAN)
        p3['angular_support']['nside'] = 1
        spec = GridSpec((-.5, -.5, -.5), (3, 3, 3), 1., 0.)
        counts = np.ones(spec.shape, np.float32)
        block = field_block(spec, (slice(0, 3),) * 3, counts,
                            np.ones(12, bool), 1, np.array([0., 1.]), np.array([0., 100.]),
                            None, p3)
        for key in ('ntilde_mpc3', 'expected_counts', 'log_count_ratio'):
            self.assertFalse(block[key].any())
        self.assertEqual(block['exposure_binary'][0, 0, 0], 0)
        for key in ('los_x', 'los_y', 'los_z'):
            self.assertEqual(block[key][0, 0, 0], 0)
        self.assertTrue(all(np.isfinite(value).all() for value in block.values()))


if __name__ == '__main__':
    unittest.main()
