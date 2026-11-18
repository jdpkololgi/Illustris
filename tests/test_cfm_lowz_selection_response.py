"""Synthetic low-z selection/response guards; no catalogue or target payload IO."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from workflows.sbi import cfm_lowz_selection_response as lowz


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.plan, self.policy, _ = lowz.load_plan()
        self.edges = lowz.bin_edges(self.plan)
        self.records = {}
        for i, phase in enumerate(self.policy['training']):
            self.records[phase] = dict(phase=phase, label_free=True, plan_sha256='synthetic',
                bin_edges=self.edges.tolist(), counts=np.full((2, 120), 100. + i).tolist(),
                volume=np.full((2, 120), 10.).tolist())

    def test_contract_and_positive_context(self):
        self.assertEqual(self.plan['science_redshift'], [.01, .55])
        values = [float('nan'), -1, 0, .001, .01, .5849, .585, .59, .595, .599, .6]
        self.assertEqual(lowz.radial_mask(values, self.plan).tolist(),
                         [False, False, False, True, True, True, False, False, True, True, False])

    def test_exact_training_set(self):
        fitted = lowz.fit_statistics(self.records, self.plan, self.policy)
        self.assertEqual(fitted['fit_phases'], self.policy['training'])
        self.assertFalse(fitted['model_inputs_ready'])
        np.testing.assert_allclose(fitted['caps']['NGC']['density'][2:110], 10.95)

    def test_development_never_in_fit(self):
        records = copy.deepcopy(self.records)
        records['ph006'] = records.pop('ph000')
        with self.assertRaisesRegex(ValueError, 'exactly'):
            lowz.fit_statistics(records, self.plan, self.policy)

    def test_sealed_guard_before_payload_io(self):
        for phase in ('ph018', 'ph019'):
            with patch.object(lowz, 'require_compute'), patch.object(lowz, 'source_payload') as source:
                with self.assertRaisesRegex(ValueError, 'outside'):
                    lowz.statistics(lowz.DEFAULT_PLAN, phase)
                source.assert_not_called()

    def test_omitted_phase_removed(self):
        records = {p: r for p, r in self.records.items() if p != 'ph000'}
        fitted = lowz.fit_statistics(records, self.plan, self.policy, 'ph000')
        self.assertNotIn('ph000', fitted['fit_phases'])
        np.testing.assert_allclose(fitted['caps']['NGC']['density'][2:110], 11.)
        with self.assertRaises(ValueError):
            lowz.fit_statistics(self.records, self.plan, self.policy, 'ph000')
        with self.assertRaises(ValueError):
            lowz.selected_phases(self.policy, 'ph006')

    def test_missing_science_volume_rejected(self):
        for row in self.records.values():
            row['volume'][1][2] = 0.
        with self.assertRaisesRegex(ValueError, 'science bins'):
            lowz.fit_statistics(self.records, self.plan, self.policy)

    def test_corrupt_statistics_rejected(self):
        for field, bad in (('counts', -1.), ('volume', float('nan'))):
            records = copy.deepcopy(self.records)
            records['ph000'][field][0][8] = bad
            with self.assertRaises(ValueError):
                lowz.fit_statistics(records, self.plan, self.policy)

    def test_units_and_no_extrapolation(self):
        fitted = lowz.fit_statistics(self.records, self.plan, self.policy)
        z = np.array([-.01, 0, .002, .01, .1, .585, .59, .595, .6, 1., np.nan])
        density, valid = lowz.density_at(z, fitted, 'NGC')
        self.assertEqual(valid.tolist(), [False, False, True, True, True, False, False, True, False, False, False])
        self.assertTrue((density[~valid] == 0).all())
        np.testing.assert_allclose(density[valid], 10.95)
        fitted['density_unit'] = 'Mpc^-3'
        with self.assertRaisesRegex(ValueError, 'units'):
            lowz.density_at(z, fitted, 'NGC')

    def test_empty_near_origin_bin_explicitly_invalid(self):
        for row in self.records.values():
            row['counts'][0][0] = 0
        fitted = lowz.fit_statistics(self.records, self.plan, self.policy)
        density, valid = lowz.density_at(np.array([.001, .01]), fitted, 'SGC')
        self.assertEqual(valid.tolist(), [False, True])
        self.assertEqual(density[0], 0.)

    def test_volume_scaled_with_density_once(self):
        fitted = lowz.fit_statistics(self.records, self.plan, self.policy)
        for name in lowz.CAPS:
            cap = fitted['caps'][name]
            valid = np.asarray(cap['valid'])
            np.testing.assert_allclose(np.asarray(cap['density'])[valid] * np.asarray(cap['effective_volume'])[valid],
                                       np.asarray(cap['pooled_counts'])[valid])

    def test_origin_response_is_finite_and_masked(self):
        from workflows.abacus_tweb.p3a_build_canonical_fields import GridSpec
        n = 12 * 256**2
        angular = dict(support=np.ones(n, dtype=bool), domain=np.full(n, 2, dtype=np.uint8),
                       angular_response=np.ones(n, dtype=np.float32))
        spec = GridSpec((-1.5, -1.5, -1.5), (3, 3, 3), 1., 0.)
        selection = lowz.fit_statistics(self.records, self.plan, self.policy)
        result, _ = lowz.response_chunk(spec, (slice(0, 3),)*3, angular, 'NGC', self.plan, selection)
        for value in result.values():
            self.assertTrue(np.isfinite(value).all())
        origin = (1, 1, 1)
        self.assertEqual(result['los_valid'][origin], 0)
        self.assertEqual(result['support_random'][origin], 0)
        self.assertEqual(result['expected_counts_random'][origin], 0)
        for name in ('los_x', 'los_y', 'los_z'):
            self.assertEqual(result[name][origin], 0)
        np.testing.assert_allclose(result['expected_counts_random'],
                                   result['ntilde_h3_mpc3'] * result['effective_volume'], rtol=1e-6)

    def test_small_statistics_and_response_writer(self):
        """Exercise FITS -> CIC -> statistics and HDF5 serialization on 27 cells."""
        import fitsio
        import h5py
        from workflows.abacus_tweb.p3a_build_canonical_fields import GridSpec
        from workflows.sbi import e2e_coupled_coordinates as coord
        from workflows.abacus_tweb import p3br_build_random_response as randoms
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = copy.deepcopy(self.plan)
            plan['output_root'] = str(root / 'output')
            plan_path = root / 'plan.json'
            plan_path.write_text(json.dumps(plan))
            rows = np.array([(0., 0., .03), (180., 0., .03)],
                            dtype=[('RA', 'f8'), ('DEC', 'f8'), ('Z', 'f8')])
            observed = root / 'observed.fits'
            fitsio.write(observed, rows)
            angular = root / 'angular.npz'
            n = 12 * 256**2
            np.savez_compressed(angular, support=np.ones(n, dtype=bool),
                domain=np.full(n, 2, dtype=np.uint8), angular_response=np.ones(n, dtype=np.float32))
            def payload(_, phase, relative, *args):
                return (angular if relative.startswith('angular') else observed), {'phase': phase}
            selection = lowz.fit_statistics(self.records, plan, self.policy)
            selection.update(plan_sha256=lowz.sha256(plan_path), code_bindings=lowz.code_bindings())
            selection_path = root / 'selection.json'
            selection_path.write_text(json.dumps(selection))
            with patch.object(lowz, 'require_compute'), \
                 patch.object(lowz, 'load_plan', return_value=(plan, self.policy, {'root': str(root)})), \
                 patch.object(lowz, 'source_payload', side_effect=payload), \
                 patch.object(lowz, 'support_grid', return_value=GridSpec((80., -5., -5.), (3, 3, 3), 3.383, 0.)), \
                 patch.object(coord, 'sky_mpc_h', return_value=np.array([[85., 0., 0.], [85., 0., 0.]])), \
                 patch.object(randoms, 'galactic_cap', return_value=np.array([0, 1])), \
                 patch.object(randoms, 'angular_boundary_distance', return_value=np.ones(n)):
                stats = lowz.statistics(plan_path, 'ph000')
                self.assertEqual(np.asarray(stats['counts']).shape, (2, 120))
                response = lowz.statistics(plan_path, 'ph006', selection_path=selection_path)
                self.assertFalse(response['model_inputs_ready'])
                self.assertEqual(len(response['outputs']), 2)
                for item in response['outputs']:
                    with h5py.File(item['path']) as data:
                        self.assertIn('selection_valid', data)
                        self.assertIn('distance_to_support_boundary', data)
                        self.assertAlmostEqual(float(data['counts'][:].sum()), 1., places=5)
                with self.assertRaises(FileExistsError):
                    lowz.statistics(plan_path, 'ph000')


if __name__ == '__main__':
    unittest.main()
