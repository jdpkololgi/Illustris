"""Small campaign orchestration/geometry receipt tests without real payloads."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from workflows.sbi import prepare_p12a_lowz_inputs as raw
from workflows.sbi import prepare_p12a_lowz_geometry_response as geometry
from workflows.sbi.run_cfm_lowz_response_preparation import stages
from workflows.sbi.audit_cfm_lowz_survey import clipped_chunks, grouped_closure
from workflows.sbi import audit_cfm_lowz_survey as audit
from workflows.sbi import cfm_lowz_selection_response as cfm


class CampaignTests(unittest.TestCase):
    def test_closure_replays_written_response_and_rejects_phase_leakage(self):
        plan, _, _ = cfm.load_plan()
        policy = dict(training=['ph000', 'ph001'], development_and_calibration=['ph006'])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = dict(plan, output_root=tmp)
            edges = cfm.bin_edges(plan)
            rows = {p: dict(phase=p, label_free=True, plan_sha256='test', bin_edges=edges.tolist(),
                counts=np.full((2, 120), 100.).tolist(), volume=np.full((2, 120), 10.).tolist(),
                code_bindings=cfm.code_bindings()) for p in policy['training']}
            full = cfm.fit_statistics(rows, plan, policy)
            for omitted in (None, 'ph000', 'ph001'):
                selection = cfm.fit_statistics({p:r for p,r in rows.items() if p != omitted},
                                               plan, policy, omitted)
                path = root / ('full' if omitted is None else 'oof_'+omitted)
                path.mkdir()
                (path/'SELECTION.json').write_text(json.dumps(selection))
            for phase in policy['training']+policy['development_and_calibration']:
                row = copy.deepcopy(rows['ph000']); row['phase'] = phase
                path = root / 'statistics_r1' / phase
                path.mkdir(parents=True)
                (path/'STATISTICS.json').write_text(json.dumps(row))
                response = dict(counts=row['counts'], volume=row['volume'], diagnostics={})
                for i, cap in enumerate(cfm.CAPS):
                    response['diagnostics'][cap] = dict(expected_counts=(
                        np.array(row['volume'])[i]*np.array(full['caps'][cap]['density'])).tolist())
                path = root / 'responses_full' / phase
                path.mkdir(parents=True)
                (path/'RESPONSE_PREPARED.json').write_text(json.dumps(response))
            with patch.object(audit, 'require_compute'), patch.object(cfm, 'load_plan', return_value=(plan, policy, {})):
                audit.closure_audit()
                result = json.loads((root/'INDEPENDENT_CLOSURE_DIAGNOSTICS.json').read_text())
                self.assertTrue(result['response_arithmetic_pass'])
                self.assertEqual(result['records']['ph006']['role'], 'development')
                path = root/'oof_ph000/SELECTION.json'
                bad = json.loads(path.read_text()); bad['fit_phases'] = ['ph000']
                path.write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError, 'leakage'):
                    audit.closure_audit()

    def test_clipped_volume_chunks_cover_sphere_centres_once(self):
        from workflows.abacus_tweb.p3a_build_canonical_fields import GridSpec
        spec = GridSpec((-100., -100., -100.), (200, 200, 200), 1., 0.)
        counts = np.zeros((200, 200, 200), dtype=np.uint8)
        for slices in clipped_chunks(spec, 20.):
            counts[slices] += 1
        self.assertEqual(counts.max(), 1)
        self.assertTrue((counts[80:120, 80:120, 80:120] == 1).all())
        self.assertEqual(int(counts.sum()), 40**3)

    def test_grouped_closure_is_not_pooled_calibration(self):
        result = grouped_closure(np.array([2., 8.]), np.array([3., 6.]),
                                 np.array([.01, .02, .03]), [.01, .02, .03])
        self.assertEqual([r['expected_over_observed_minus_one'] for r in result], [.5, -.25])

    def test_exact_phase_and_omission_scope(self):
        _, _, policy = raw.load_plan()
        phases, omissions = stages(policy)
        self.assertEqual(len(phases), 23)
        self.assertEqual(omissions, [None] + policy['training'])
        self.assertFalse(set(phases) & set(policy['final_test']))
        for phase in ['ph018', 'ph019', 'ph099']:
            with patch.object(raw, 'read') as reader:
                with self.assertRaises(PermissionError):
                    geometry.prepare(phase, {}, policy, {}, 0)
                reader.assert_not_called()

    def test_geometry_writer_is_separate_from_selection(self):
        plan, _, policy = raw.load_plan()
        with tempfile.TemporaryDirectory() as tmp:
            plan = copy.deepcopy(plan)
            plan['output_root'] = tmp
            root = Path(tmp) / 'ph001'
            (root / 'p1_canonical').mkdir(parents=True)
            (root / 'p3_fields').mkdir()
            points = root / 'p1_canonical/points.npy'
            np.save(points, np.array([[2., 0., 0., 0.], [-2., 0., 0., 1.], [-2., 0., 0., 1.]]))
            index = root / 'p1_canonical/canonical_index.npz'
            np.savez(index, context=np.array([True, True, False]))
            p1 = root / 'p1_canonical/manifest.json'
            p1.write_text(json.dumps({'points': str(points)}))
            angular = root / 'angular.npz'
            np.savez(angular, support=np.ones(12, bool), domain=np.arange(12) % 4)
            manifest = root / 'p3_fields/field_manifest.json'
            manifest.write_text(json.dumps(dict(points=str(points), p1_manifest_sha256=raw.sha256(p1),
                canonical_index=str(index), canonical_index_sha256=raw.sha256(index),
                redshift_contract=raw.domain_from(plan), components={'SGC': {}, 'NGC': {}},
                angular_support=dict(source=str(angular), sha256=raw.sha256(angular)))))
            (root / 'RAW_PREPARATION_COMPLETE.json').write_text(json.dumps(dict(
                phase='ph001', pass_raw_preparation=True, fields_sha256=raw.sha256(manifest))))
            contract = dict(redshift_contract=raw.domain_from(plan),
                cosmology=dict(radius_grid_mpc=[0., 100.], redshift_grid=[0., 1.]))
            with patch.object(geometry.randoms, 'load_map', return_value=dict(
                    support=np.ones(12, bool), domain=np.arange(12) % 4)), \
                 patch.object(geometry.randoms, 'angular_boundary_distance', return_value=np.ones(12)), \
                 patch.object(geometry, 'sample_centres', side_effect=lambda points, *args:
                              (np.ones(len(points)), np.ones(len(points), bool))):
                result = geometry.prepare('ph001', plan, policy, contract, time.monotonic()+10)
                self.assertEqual(result['support_rows'], 2)
                self.assertTrue(result['geometry_features_ready'])
                self.assertFalse(result['model_inputs_ready'])
                self.assertFalse(result['selection_fitted'])
                self.assertEqual(result['excluded_context_rows'], 1)
                self.assertEqual(sum(result['diagnostics']['observed_rows']), 2)
                self.assertFalse(np.load(result['arrays']['support'])[-1])
                self.assertNotIn('selection_sha256', result)
                for key, path in result['arrays'].items():
                    self.assertEqual(raw.sha256(path), result['array_sha256'][key])
                with self.assertRaises(FileExistsError):
                    geometry.prepare('ph001', plan, policy, contract, time.monotonic()+10)


if __name__ == '__main__':
    unittest.main()
