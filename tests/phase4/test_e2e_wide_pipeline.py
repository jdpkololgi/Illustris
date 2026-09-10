import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from workflows.sbi import e2e_wide_pipeline as p
from workflows.sbi.e2e_wide_models import ConditionalFieldNet


class WidePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_contract_and_output_guards(self):
        c = p.read_config()
        self.assertEqual(c['variant'], 'wide384_f4')
        self.assertEqual(c['sampling']['cfm_steps']*2, c['sampling']['diffusion_steps'])
        self.assertFalse(c['training_ready'])
        for path in ('/tmp/escape.json', c['output_root'], c['output_root']+'/ph004/n.json'):
            with self.assertRaises((ValueError, PermissionError)):
                p.output_path(c, path)
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp)/'config.json'
            c['training']['maximum_updates'] = 193
            config.write_text(json.dumps(c))
            with self.assertRaises(ValueError):
                p.read_config(config)

    def test_batch_order_and_random_addresses(self):
        sequence = [p.example_index(i, 7, 42) for i in range(21)]
        self.assertEqual(sequence, [p.example_index(i, 7, 42) for i in range(21)])
        for i in range(3):
            self.assertEqual(sorted(sequence[i*7:(i+1)*7]), list(range(7)))
        addresses = [p.seed_for(42, 'anchor', sample, stage)
                     for sample in ('a', 'b') for stage in ('coarse', 'fine')]
        self.assertEqual(len(set(addresses)), 4)

    def test_checkpoint_resume_matches_uninterrupted_for_both_methods(self):
        for method in ('cfm', 'diffusion'):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                torch.manual_seed(31)
                model = ConditionalFieldNet(2, base_channels=2, levels=2, wide_condition_channels=2)
                opt = torch.optim.AdamW(model.parameters(), lr=.001)
                gen = torch.Generator().manual_seed(82)
                target = torch.randn(1,1,8,8,8)
                cond = torch.randn(1,2,8,8,8)
                wide = torch.randn_like(cond)
                p.train_update(model, opt, target, cond, method, gen, 1., wide)
                checkpoint = Path(tmp)/'step_1.pt'
                p.save_checkpoint(checkpoint, model=model, optimizer=opt, generator=gen,
                                  binding={'fixture': True}, stage='fine', method=method,
                                  step=1, history=[])
                expected = p.train_update(model, opt, target, cond, method, gen, 1., wide)
                expected_state = copy.deepcopy(model.state_dict())
                resumed = ConditionalFieldNet(2, base_channels=2, levels=2, wide_condition_channels=2)
                resumed_opt = torch.optim.AdamW(resumed.parameters(), lr=.001)
                state = p.load_checkpoint(checkpoint, {'fixture': True}, 'fine', method)
                resumed.load_state_dict(state['model'])
                resumed_opt.load_state_dict(state['optimizer'])
                restored_gen = torch.Generator()
                p.restore_rng(state['rng'], restored_gen)
                actual = p.train_update(resumed, resumed_opt, target, cond, method, restored_gen, 1., wide)
                self.assertEqual(expected, actual)
                for k, v in resumed.state_dict().items():
                    self.assertTrue(torch.equal(v, expected_state[k]), k)
                with self.assertRaises(ValueError):
                    p.load_checkpoint(checkpoint, {'fixture': False}, 'fine', method)
                with self.assertRaises(ValueError):
                    p.load_checkpoint(checkpoint, {'fixture': True}, 'coarse', method)

    def test_shared_generated_coarse_replaces_placeholder_and_wide_reaches_fine(self):
        class Dataset:
            def inverse_target(self, x, name):
                return x*2+3 if name == 'coarse' else x*4-1
            def normalize_targets(self, x, name):
                return (x-3)/2
        coarse_model, fine_model = object(), object()
        item = {'anchor_id': 'ph000_fixture', 'coarse_condition': np.ones((2,16,16,16), np.float32),
                'fine_condition': np.zeros((3,8,8,8), np.float32)}
        calls = []
        def sampled(model, condition, **kwargs):
            calls.append((model, condition.clone(), kwargs))
            return torch.full((1,1,*condition.shape[2:]), 5. if model is coarse_model else 7.)
        c = p.read_config()
        c['fine_side'] = 8
        with patch.object(p, 'sample_field', side_effect=sampled):
            coarse, fine, seeds = p.generate_pair(c, Dataset(), item, coarse_model, fine_model,
                                                  'cfm', 'draw01', torch.device('cpu'))
        np.testing.assert_allclose(coarse, 13.)
        np.testing.assert_allclose(fine, 27.)
        np.testing.assert_allclose(calls[1][1][0,-1], 5.)
        self.assertTrue(torch.equal(calls[1][2]['wide_condition'], calls[0][1]))
        self.assertNotEqual(seeds['coarse'], seeds['fine'])
        self.assertTrue(np.all(item['fine_condition'][-1] == 0))

    def test_physics_constant_dc_and_trace(self):
        coarse = np.full((16,)*3, .6)
        fine = np.full((8,)*3, -.12)
        result = p.reconstruct(coarse, fine, core_side=4)
        np.testing.assert_allclose(result['delta_local96'], .48, atol=1e-7)
        np.testing.assert_allclose(result['eigen_core'], .16, atol=1e-7)
        self.assertLess(result['trace_max_abs'], 1e-7)
        coarse = np.random.default_rng(42).normal(size=(16,)*3)*.1
        fine = np.random.default_rng(43).normal(size=(8,)*3)*.1
        result = p.reconstruct(coarse, fine, core_side=4)
        self.assertLess(result['trace_max_abs'], 1e-7)
        self.assertTrue(np.all(np.diff(result['eigen_core'], axis=-1) >= 0))

    def test_real_array_work_refuses_login(self):
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaises(RuntimeError):
                p.runtime()
            with self.assertRaises(RuntimeError):
                p.reconstruct(np.zeros((96,)*3), np.zeros((96,)*3))


if __name__ == '__main__':
    unittest.main()
