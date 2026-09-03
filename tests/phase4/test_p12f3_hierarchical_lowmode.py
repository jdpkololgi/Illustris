import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from workflows.sbi.p12f3_hierarchical_lowmode import (
    coarse_core_mask,
    crop_tensor_to_patch,
    physical_low_mode_mask,
    prepare_low_mode_example,
    spectral_split,
)
from workflows.sbi.p12f3_train_lowmode_flow import restore_rng_states
from workflows.sbi.p12f3_train_lowmode_flow import load_config


class P12F3HierarchicalLowModeTests(unittest.TestCase):
    def test_rng_resume_accepts_array_backed_state(self):
        expected = torch.get_rng_state().clone()
        torch.manual_seed(123456)
        restore_rng_states({"torch_rng": expected.numpy(), "cuda_rng": []})
        self.assertTrue(torch.equal(torch.get_rng_state(), expected))

    def test_physical_split_is_exact_and_excludes_dc(self):
        generator = torch.Generator().manual_seed(9)
        field = torch.randn((2, 1, 24, 20, 16), generator=generator)
        low, high = spectral_split(
            field, voxel_mpc_h=5.0, maximum_k_h_mpc=0.1813799364234218
        )
        self.assertTrue(torch.allclose(low + high, field, atol=2e-6, rtol=2e-6))
        self.assertLess(float(torch.abs(low.mean())), 1e-6)
        mask = physical_low_mode_mask(
            field.shape[-3:],
            voxel_mpc_h=5.0,
            maximum_k_h_mpc=0.1813799364234218,
            device="cpu",
        )
        coefficients = torch.fft.rfftn(low, dim=(-3, -2, -1), norm="ortho")
        self.assertEqual(bool(mask[0, 0, 0]), False)
        self.assertLess(float(torch.max(torch.abs(coefficients[..., ~mask]))), 2e-5)

    def test_nested_crop_uses_global_lattice_bounds(self):
        source = torch.arange(32 * 32 * 32).reshape(1, 1, 32, 32, 32)
        cropped = crop_tensor_to_patch(
            source,
            source_start=np.array([8, 16, 24]),
            target_start=np.array([16, 24, 32]),
            target_stop=np.array([32, 40, 48]),
        )
        self.assertEqual(tuple(cropped.shape), (1, 1, 16, 16, 16))
        self.assertTrue(torch.equal(cropped, source[:, :, 8:24, 8:24, 8:24]))

    def test_pooled_state_and_authoritative_mask_align(self):
        condition = torch.randn(1, 3, 19, 20, 21)
        target = torch.randn(1, 1, 19, 20, 21)
        core = (slice(3, 16), slice(4, 18), slice(5, 19))
        pooled_condition, pooled_target, science = prepare_low_mode_example(
            condition=condition,
            low_residual=target,
            core_slice=core,
            coarse_factor=2,
        )
        self.assertEqual(tuple(pooled_condition.shape), (1, 3, 10, 10, 11))
        self.assertEqual(tuple(pooled_target.shape), (1, 1, 10, 10, 11))
        self.assertEqual(science.shape, pooled_target.shape)
        self.assertGreater(int(science.sum()), 0)
        expected = coarse_core_mask(
            (19, 20, 21), core, factor=2, device="cpu"
        )
        self.assertTrue(torch.equal(science, expected))

    def test_config_rejects_blind_phase_in_sources(self):
        config = json.loads(
            (Path(__file__).resolve().parents[2] / "configs/p12f3_hierarchical_lowmode_v1.json").read_text()
        )
        config["sources"]["bad"] = "/tmp/ph001_truth.npy"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(config))
            with self.assertRaises(PermissionError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
