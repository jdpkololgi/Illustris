import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


PATH = Path(__file__).parents[2] / "workflows/abacus_tweb/p7_ftier_patch_utils.py"
SPEC = importlib.util.spec_from_file_location("p7_utils", PATH)
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


class P7FTierPatchUtilsTests(unittest.TestCase):
    def test_cic_and_tsc_scatter_conserve_every_channel(self):
        frac = np.asarray([[3.2, 3.3, 3.4], [5.1, 4.8, 4.2]])
        latent = np.asarray([[1.0, 2.0], [3.0, -1.0]], dtype=np.float32)
        for scheme in ("cic", "tsc"):
            grid, diagnostics = MOD.scatter_nodes(
                latent, frac, (9, 9, 9), scheme=scheme
            )
            np.testing.assert_allclose(
                grid.sum(axis=(1, 2, 3)), latent.sum(axis=0), atol=2e-6
            )
            self.assertLess(diagnostics["maximum_weight_error"], 2e-6)
            self.assertEqual(diagnostics["dropped_weight_sum"], 0.0)

    def test_scatter_overlap_parity_in_global_voxels(self):
        frac_global = np.asarray([[4.2, 4.4, 4.6], [5.1, 5.2, 5.3]])
        latent = np.asarray([[1.0], [2.0]], dtype=np.float32)
        left, _ = MOD.scatter_nodes(
            latent, frac_global, (8, 8, 8), scheme="tsc"
        )
        right, _ = MOD.scatter_nodes(
            latent, frac_global - 2.0, (8, 8, 8), scheme="tsc"
        )
        np.testing.assert_allclose(
            left[:, 2:8, 2:8, 2:8],
            right[:, 0:6, 0:6, 0:6],
            atol=1e-7,
        )

    def test_incomplete_scatter_is_rejected(self):
        with self.assertRaises(ValueError):
            MOD.scatter_nodes(
                np.ones((1, 1)), np.asarray([[0.0, 0.0, 0.0]]),
                (4, 4, 4), scheme="tsc", require_complete=True,
            )

    def test_fft_tensor_trace_and_eigen_ordering(self):
        rng = np.random.default_rng(17)
        delta = rng.normal(size=(18, 20, 22))
        components, smoothed = MOD.fft_tidal_components(
            delta, cell_mpc=5.0, rsmooth_mpc=10.4
        )
        self.assertLess(MOD.trace_max_abs_error(components, smoothed), 2e-12)
        tensor, eigenvalues, eigenvectors = MOD.tensor_and_eigensystem(components)
        np.testing.assert_allclose(
            tensor, np.swapaxes(tensor, -1, -2), atol=0.0
        )
        self.assertTrue(np.all(np.diff(eigenvalues, axis=-1) >= -1e-12))
        identity = np.einsum("...ji,...jk->...ik", eigenvectors, eigenvectors)
        np.testing.assert_allclose(
            identity, np.broadcast_to(np.eye(3), identity.shape), atol=2e-12)

    def test_padded_fft_crops_back_and_preserves_trace(self):
        rng = np.random.default_rng(21)
        delta = rng.normal(size=(12, 14, 16))
        components, smoothed = MOD.fft_tidal_components(
            delta, cell_mpc=5.0, rsmooth_mpc=10.4,
            apodization_width_voxels=3, padding_voxels=4,
        )
        self.assertEqual(components["xx"].shape, delta.shape)
        self.assertLess(MOD.trace_max_abs_error(components, smoothed), 2e-12)

    def test_constant_field_has_zero_tidal_components(self):
        components, smoothed = MOD.fft_tidal_components(
            np.ones((12, 12, 12)), cell_mpc=5.0, rsmooth_mpc=10.4
        )
        self.assertEqual(float(np.max(np.abs(smoothed))), 0.0)
        self.assertEqual(
            max(float(np.max(np.abs(value))) for value in components.values()),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
