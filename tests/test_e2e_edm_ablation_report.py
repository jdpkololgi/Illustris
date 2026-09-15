import copy
import unittest
from pathlib import Path
from unittest.mock import patch

from workflows.sbi.e2e_edm_ablation_report import summarize, verify_artifacts


class EDMAblationReportTests(unittest.TestCase):
    def fixture(self):
        metric = dict(noise_amplitude=[.1]*4, error_power=[1.]*4, gain=[1.]*4)
        probes = [dict(group=g, phase=p, ratio=r, metrics=metric)
                  for g in ('fit', 'transfer') for p in ('ph000', 'ph002', 'ph003')
                  for r in (.05, .2, 1., 5., 20.) for _ in range(2)]
        branches = [dict(arm=dict(name=f'arm{i}', clip=10. if i else 1.),
                         parameter_count=100, initial_max_abs=0., gate={},
                         curve=[dict(update=u, summary={}, probes=probes) for u in (0, 128, 256, 512)],
                         training=[dict(gradient_norm_before_clip=5., sigma=.2, loss=1.,
                                        gradient_groups=dict(base=5., film=0., receptive=0.))]*512)
                    for i in range(6)]
        fields = [dict(label=label, group=g, anchor_id=f'{g}{i}', sample_sha256='test',
                       metrics=dict(power_ratio=[1.]*4), density_below_minus_one_fraction=0.)
                  for label in ['parent']+[f'arm{i}' for i in range(6)]
                  for g in ('fit', 'transfer') for i in range(3)]
        return dict(complete=True, results=branches, fields=fields, registration={},
                    checkpoints={str(i): 'test' for i in range(18)}, elapsed_seconds=1.,
                    training_ready=False, calibration_pass=None)

    def test_counts_and_arm_specific_clipping(self):
        result = summarize(self.fixture())
        self.assertEqual(result['counts'], dict(updates=3072, probes=1440, draws=42, checkpoints=18))
        self.assertEqual(result['branches']['arm0']['clipping_fraction'], 1.)
        self.assertEqual(result['branches']['arm1']['clipping_fraction'], 0.)
        self.assertEqual(result['branches']['arm1']['sigma_histogram']['counts'], [0, 0, 512, 0, 0, 0])
        self.assertEqual(len(result['branches']['arm1']['phase_curves']), 48)

    def test_rejects_incomplete_or_missing_phase_pairs(self):
        data = self.fixture(); data['fields'].pop()
        with self.assertRaises(ValueError):
            summarize(data)
        data = copy.deepcopy(self.fixture())
        data['results'][0]['curve'][0]['probes'][0]['phase'] = 'wrong'
        with self.assertRaises(ValueError):
            summarize(data)

    def test_artifact_and_noise_pairing_verification(self):
        branch = dict(training=[dict(anchor_id='a', noise_seed=11)],
                      curve=[dict(probes=[dict(seed=22)])])
        data = dict(registration=dict(source_sha256={'test.py': 'ok'}),
                    checkpoints={'arm/128': 'ok'},
                    fields=[dict(method='diffusion', label='arm', anchor_id='a', sample_sha256='ok')],
                    results=[copy.deepcopy(branch), copy.deepcopy(branch)])
        with patch('workflows.sbi.e2e_edm_ablation_report.p.sha256', return_value='ok'):
            self.assertTrue(all(verify_artifacts(data, Path('/tmp')).values()))
            data['results'][1]['training'][0]['noise_seed'] = 33
            with self.assertRaisesRegex(ValueError, 'unpaired'):
                verify_artifacts(data, Path('/tmp'))
            data['results'][1]['training'][0]['noise_seed'] = 11
            data['results'][1]['curve'][0]['probes'][0]['seed'] = 11
            with self.assertRaisesRegex(ValueError, 'overlap'):
                verify_artifacts(data, Path('/tmp'))
        with patch('workflows.sbi.e2e_edm_ablation_report.p.sha256', return_value='wrong'):
            with self.assertRaisesRegex(ValueError, 'source drift'):
                verify_artifacts(data, Path('/tmp'))


if __name__ == '__main__':
    unittest.main()
