import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from workflows.sbi import prepare_lowz_point_selection as selection


class PointCandidateTests(unittest.TestCase):
    def setUp(self):
        self.edges = np.round(np.linspace(0., .6, 121), 12)
        self.records = {p: dict(phase=p, label_free=True, density_unit='Mpc^-3', bin_edges=self.edges.tolist(),
                              counts=np.full((2, 120), 100.).tolist(), volume=np.full((2, 120), 10.).tolist())
                        for p in ('ph000', 'ph001', 'ph002')}

    def test_exact_training_and_omitted_phase(self):
        training = list(self.records)
        full = selection.fit(self.records, training, self.edges, 'Mpc^-3')
        np.testing.assert_allclose(full['density'], 10.)
        subset = {p: row for p, row in self.records.items() if p != 'ph001'}
        out = selection.fit(subset, training, self.edges, 'Mpc^-3', 'ph001')
        self.assertNotIn('ph001', out['fit_phases'])
        self.assertFalse(out['model_inputs_ready'])
        with self.assertRaises(ValueError):
            selection.fit(self.records, training, self.edges, 'Mpc^-3', 'ph001')
        with self.assertRaises(ValueError):
            selection.fit(subset, training, self.edges, 'Mpc^-3', 'ph006')

    def test_units_and_development_contamination_rejected(self):
        with self.assertRaises(ValueError):
            selection.fit(self.records, list(self.records), self.edges, '(Mpc/h)^-3')
        extra = dict(self.records, ph006=self.records['ph000'])
        with self.assertRaises(ValueError):
            selection.fit(extra, list(self.records), self.edges, 'Mpc^-3')

    def test_analytic_volume_includes_origin_without_direction(self):
        np.testing.assert_allclose(selection.shell_volumes(4*np.pi, [0., 1., 2.]), 4*np.pi/3*np.array([1., 7.]))
        with self.assertRaises(ValueError): selection.shell_volumes(1., [1., 0.])

    def test_real_fits_point_statistics_read_only_observed_columns(self):
        import fitsio
        import healpy as hp
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, angular = root/'observed.fits', root/'angular.npz'
            rows = np.array([(0., 0., .02, 999.), (180., 0., .025, 888.),
                             (0., 0., -.02, 777.), (0., 0., .59, 666.)],
                            dtype=[('RA', 'f8'), ('DEC', 'f8'), ('Z', 'f8'), ('FAKE_TRUTH', 'f8')])
            fitsio.write(path, rows, clobber=False)
            n = hp.nside2npix(256)
            domains = np.zeros(n, np.uint8); domains[n//2:] = 2
            np.savez(angular, support=np.ones(n, bool), domain=domains, angular_response=np.ones(n))
            plan = dict(source_campaign={'sha256': 'synthetic'})
            policy = dict(training=['ph000'], development_and_calibration=[])
            source = dict(root=str(root))
            with patch.object(selection.cfm, 'source_payload', side_effect=[(path, {}), (angular, {})]), \
                    patch('fitsio.read', wraps=fitsio.read) as read:
                result = selection.statistics('cfm', 'ph000', plan, policy, source, root)
            self.assertEqual(read.call_args.kwargs['columns'], ['RA', 'DEC', 'Z'])
            self.assertEqual(result['context_rows'], 2)
            self.assertEqual(result['excluded_context_rows'], 2)
            self.assertEqual(np.sum(result['counts'])+np.sum(result['unsupported_point_counts']), 2)
            self.assertEqual(result['density_unit'], '(Mpc/h)^-3')


if __name__ == '__main__':
    unittest.main()
