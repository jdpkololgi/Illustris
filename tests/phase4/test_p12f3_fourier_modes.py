import unittest
import copy
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from workflows.sbi.p12f3_fourier_modes import (
    ConditionalFourierVelocityUNet,
    build_fourier_layout,
    empty_whitening_accumulator,
    equal_band_flow_loss,
    finalize_whitening,
    hermitian_max_error,
    lowpass_exact,
    pack_fourier_components,
    rectified_flow_pair,
    spectral_lowpass_reference,
    unpack_fourier_components,
    update_whitening_accumulator,
    whiten_components,
)
from workflows.sbi.p12f3_train_fourier_lowmode_flow import load_config


REPO_ROOT = Path(__file__).resolve().parents[2]


class P12F3FourierModeTests(unittest.TestCase):
    def setUp(self):
        self.edges = (0.0, 0.19, 0.38)
        self.layout = build_fourier_layout(
            (8, 10, 12), voxel_mpc_h=5.0, band_edges_h_mpc=self.edges
        )

    def test_exact_real_roundtrip_and_hermitian_projection(self):
        generator = torch.Generator().manual_seed(4)
        field = torch.randn((2, 1, 8, 10, 12), generator=generator)
        vector = pack_fourier_components(field, self.layout)
        reconstructed = unpack_fourier_components(vector, self.layout)
        expected = spectral_lowpass_reference(field, self.layout)
        self.assertLess(float(torch.max(torch.abs(reconstructed - expected))), 2e-6)
        self.assertLess(hermitian_max_error(vector, self.layout), 2e-6)
        self.assertTrue(torch.isreal(reconstructed).all())

    def test_layout_has_one_representative_and_no_dc(self):
        self.assertEqual(len(np.unique(self.layout.representative_flat)), self.layout.modes)
        self.assertNotIn(0, self.layout.representative_flat.tolist())
        nonself = ~self.layout.self_conjugate
        self.assertTrue(np.all(self.layout.representative_flat[nonself] != self.layout.conjugate_flat[nonself]))
        self.assertEqual(set(self.layout.mode_band.tolist()), {0, 1})

    def test_bandwise_whitening_and_equal_band_loss(self):
        accumulator = empty_whitening_accumulator(4)
        for seed in range(20):
            field = torch.randn((1, 1, 8, 10, 12), generator=torch.Generator().manual_seed(seed))
            update_whitening_accumulator(
                accumulator, pack_fourier_components(field, self.layout), self.layout
            )
        whitening = finalize_whitening(accumulator)
        vector = pack_fourier_components(
            torch.randn((1, 1, 8, 10, 12), generator=torch.Generator().manual_seed(90)),
            self.layout,
        )
        white = whiten_components(vector, whitening, self.layout)
        self.assertTrue(torch.isfinite(white).all())
        target = torch.zeros_like(white)
        prediction = torch.zeros_like(white)
        band = self.layout.component_group // 2
        prediction[:, torch.as_tensor(band == 0)] = 1.0
        prediction[:, torch.as_tensor(band == 1)] = 3.0
        self.assertAlmostEqual(float(equal_band_flow_loss(prediction, target, self.layout, 2)), 5.0, places=6)

    def test_flow_pair_and_directional_model_shapes(self):
        whitening = {"mean": [0.0] * 4, "std": [1.0] * 4, "count": [10] * 4}
        target = torch.randn((1, self.layout.components), generator=torch.Generator().manual_seed(3))
        state, time, velocity = rectified_flow_pair(target)
        self.assertEqual(state.shape, target.shape)
        self.assertEqual(velocity.shape, target.shape)
        condition = torch.randn((1, 3, *self.layout.shape))
        model = ConditionalFourierVelocityUNet(condition_channels=3, base=2)
        predicted = model(state, time, condition, layout=self.layout, whitening=whitening)
        self.assertEqual(predicted.shape, target.shape)
        self.assertTrue(torch.isfinite(predicted).all())

    def test_synthetic_known_covariance_survives_coordinate_roundtrip(self):
        generator = torch.Generator().manual_seed(17)
        draws = torch.randn((256, self.layout.components), generator=generator)
        # Induce a known joint correlation between two independent coefficients.
        draws[:, 1] = 0.7 * draws[:, 0] + np.sqrt(1.0 - 0.7**2) * draws[:, 1]
        fields = unpack_fourier_components(draws, self.layout)
        recovered = pack_fourier_components(fields, self.layout)
        correlation = float(torch.corrcoef(torch.stack((recovered[:, 0], recovered[:, 1])))[0, 1])
        self.assertLess(float(torch.max(torch.abs(recovered - draws))), 3e-6)
        self.assertGreater(correlation, 0.62)
        self.assertLess(correlation, 0.78)

    def test_registered_contract_is_sealed_and_forbids_pooling(self):
        path = REPO_ROOT / "configs/p12f3_fourier_lowmode_v1.json"
        config = load_config(path)
        self.assertEqual(config["roles"]["validation"], "ph006")
        self.assertEqual(config["roles"]["sealed_blind_test"], "ph001")
        self.assertFalse(config["target"]["pooling"])
        self.assertFalse(config["target"]["interpolation"])

    def test_contract_rejects_blind_training_phase(self):
        path = REPO_ROOT / "configs/p12f3_fourier_lowmode_v1.json"
        config = json.loads(path.read_text())
        config = copy.deepcopy(config)
        config["roles"]["training"].append("ph001")
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory) / "config.json"
            temporary.write_text(json.dumps(config))
            with self.assertRaises(PermissionError):
                load_config(temporary)


if __name__ == "__main__":
    unittest.main()
