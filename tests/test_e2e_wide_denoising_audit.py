"""Small synthetic checks; no data payloads or GPU needed."""
import math
import unittest

import numpy as np
import torch

from workflows.sbi.e2e_wide_denoising_audit import Bands, bridge, clean_estimate
from workflows.sbi.e2e_wide_models import sample_field
from workflows.sbi.e2e_wide_denoising_report import summarize


class DenoisingAuditTests(unittest.TestCase):
    def test_matched_ratio_and_exact_recovery(self):
        gen = torch.Generator().manual_seed(4)
        truth = torch.randn((1, 1, 8, 8, 8), generator=gen)
        noise = torch.randn(truth.shape, generator=gen)
        for method in ('cfm', 'diffusion'):
            for ratio in (.05, .2, 1., 5., 20.):
                t, alpha, sigma = bridge(method, ratio)
                self.assertAlmostEqual(sigma / alpha, ratio)
                state = alpha * truth + sigma * noise
                velocity = truth - noise if method == 'cfm' else alpha * noise - sigma * truth
                clean = clean_estimate(method, state, velocity, t)
                torch.testing.assert_close(clean, truth, atol=1e-6, rtol=1e-6)
                wrong = clean_estimate(method, state, velocity + .3, t)
                error = (.3 * sigma) * (1 if method == 'cfm' else -1)
                torch.testing.assert_close(wrong - truth, torch.full_like(truth, error), atol=1e-6, rtol=1e-5)

    def test_bands_scale_dc_and_signed_cross(self):
        bands = Bands(8, 1., [0, 1., 2., 3., np.inf])
        truth = np.random.default_rng(3).normal(size=(8, 8, 8))
        metric = bands.compare(2 * truth + 5, truth)
        np.testing.assert_allclose(metric['power_ratio'], 4, atol=1e-12)
        np.testing.assert_allclose(metric['gain'], 2, atol=1e-12)
        np.testing.assert_allclose(metric['correlation'], 1, atol=1e-12)
        parts = bands.decomposition(truth, -.5 * truth)
        np.testing.assert_allclose(parts['cross'], -np.array(parts['coarse']))
        np.testing.assert_allclose(parts['total'], .25 * np.array(parts['coarse']))
        f = bands.fft(truth)
        centered = (truth - np.sum(truth * bands.window) / bands.sum_window) * bands.window
        self.assertAlmostEqual(sum(bands.cross(f, f)), np.sum(centered**2) / np.sum(bands.window**2))

    def test_noise_leakage(self):
        bands = Bands(8, 1., [0, 1., 2., 3., np.inf])
        rng = np.random.default_rng(2)
        truth, noise = rng.normal(size=(2, 8, 8, 8))
        metric = bands.compare(truth + .2 * noise, truth, noise)
        np.testing.assert_allclose(metric['noise_leakage_coefficient'], .2, atol=1e-12)
        np.testing.assert_allclose(metric['noise_correlated_error_power'], metric['error_power'], atol=1e-12)

    def test_actual_samplers_with_analytic_oracles(self):
        truth = torch.randn((1, 1, 8, 8, 8), generator=torch.Generator().manual_seed(3))
        initial = torch.randn(truth.shape, generator=torch.Generator().manual_seed(4))

        class Oracle(torch.nn.Module):
            def __init__(self, method):
                super().__init__()
                self.method = method

            def forward(self, state, time, condition):
                if self.method == 'cfm':
                    return truth - initial
                angle = float(time[0]) * math.pi / 2
                return (math.cos(angle) * state - truth) / math.sin(angle)

        for method in ('cfm', 'diffusion'):
            result = sample_field(Oracle(method), torch.zeros_like(truth), method, 4,
                                  torch.Generator().manual_seed(4))
            torch.testing.assert_close(result, truth, atol=1e-6, rtol=1e-6)

    def test_report_pairs_same_anchor_method_and_draw(self):
        data = dict(complete=True, registration={}, elapsed_seconds=1., probe_summary={},
                    probes=[], components=[], true_coarse_diagnostics=[], sampler_trajectories=[])
        power = np.array([1., 2., 3., 4.])
        for phase in ('ph000', 'ph002', 'ph003'):
            for method in ('cfm', 'diffusion'):
                for step in (192, 384):
                    for draw in range(4):
                        data['components'].append(dict(anchor_id=phase, phase=phase, method=method,
                            step=step, draw=draw, truth={'total': power}, prediction={
                            'total': 2*power, 'coarse': .4*power, 'fine': 1.4*power, 'cross': .2*power}))
                    for stage in ('coarse', 'fine'):
                        for ratio in (.05, .2, 1., 5., 20.):
                            metrics = {k: power for k in ('truth_power', 'power_ratio', 'gain', 'error_power',
                                       'error_over_truth_power', 'noise_leakage_over_input_sigma',
                                       'velocity_error_band_fraction')}
                            data['probes'].append(dict(anchor_id=phase, phase=phase, method=method,
                                 stage=stage, step=step, ratio=ratio, metrics=metrics))
                data['true_coarse_diagnostics'].append(dict(anchor_id=phase, phase=phase, method=method,
                    components={'total': 1.5*power}, density_below_minus_one_fraction=.01, sample_sha256='test'))
                for stage in ('coarse', 'fine'):
                    data['sampler_trajectories'].append(dict(anchor_id=phase, method=method, stage=stage,
                         call=0, time=1., predicted_clean_power=1.7*power))
        result = summarize(data)
        np.testing.assert_allclose(result['components']['all/cfm/384']['power_ratio'], 2)
        np.testing.assert_allclose(result['true_coarse']['all/diffusion']['oracle_over_actual_power'], .75)
        np.testing.assert_allclose(result['trajectory']['cfm/fine'][0]['clean_power_ratio'], 1.7)


if __name__ == '__main__':
    unittest.main()
