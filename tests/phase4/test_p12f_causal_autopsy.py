import tempfile
import unittest
from pathlib import Path

import numpy as np

from workflows.sbi.p12f_causal_autopsy import (
    _donor_order,
    cube_symmetry,
    load_config,
    low_k_intervention,
    physical_invariants,
    radial_low_k_mask,
    selected_autopsy_entries,
)


class P12FCausalAutopsyTests(unittest.TestCase):
    def test_physical_invariants_trace_q_and_lode_limits(self):
        values = np.asarray([[-1.0, -1.0, 2.0], [-2.0, 1.0, 1.0]])
        result = physical_invariants(values)
        np.testing.assert_allclose(result["trace"], 0.0)
        np.testing.assert_allclose(result["shear_q"], 3.0)
        np.testing.assert_allclose(result["lode_eta"], [1.0, -1.0], atol=1e-6)
        np.testing.assert_allclose(result["gap12"], [0.0, 3.0])
        np.testing.assert_allclose(result["gap23"], [3.0, 0.0])

    def test_low_k_mean_oracle_changes_only_registered_modes(self):
        rng = np.random.default_rng(4)
        samples = rng.normal(size=(12, 8, 8, 8))
        truth = rng.normal(size=(8, 8, 8))
        mask, label, _ = radial_low_k_mask(
            truth.shape, bins=12, corrected_bins=(0, 1), exclude_dc=True
        )
        output = low_k_intervention(
            samples,
            truth,
            low_mask=mask,
            bin_label=label,
            power_scale_by_bin=np.ones(12),
            correct_mean=True,
            correct_power=False,
        )
        before = np.fft.rfftn(samples.mean(axis=0), norm="ortho")
        after = np.fft.rfftn(output.mean(axis=0), norm="ortho")
        target = np.fft.rfftn(truth, norm="ortho")
        np.testing.assert_allclose(after[mask], target[mask], atol=2e-6)
        np.testing.assert_allclose(after[~mask], before[~mask], atol=2e-6)
        self.assertFalse(mask[0, 0, 0])

    def test_low_k_power_oracle_scales_only_registered_scatter(self):
        rng = np.random.default_rng(8)
        samples = rng.normal(size=(24, 8, 8, 8))
        truth = rng.normal(size=(8, 8, 8))
        mask, label, _ = radial_low_k_mask(
            truth.shape, bins=12, corrected_bins=(0,), exclude_dc=True
        )
        scale = np.ones(12)
        scale[0] = 1.7
        output = low_k_intervention(
            samples,
            truth,
            low_mask=mask,
            bin_label=label,
            power_scale_by_bin=scale,
            correct_mean=False,
            correct_power=True,
        )
        before = np.fft.rfftn(samples - samples.mean(axis=0), axes=(-3, -2, -1), norm="ortho")
        after = np.fft.rfftn(output - output.mean(axis=0), axes=(-3, -2, -1), norm="ortho")
        np.testing.assert_allclose(after[:, mask], 1.7 * before[:, mask], atol=3e-6)
        np.testing.assert_allclose(after[:, ~mask], before[:, ~mask], atol=3e-6)

    def test_cube_symmetry_preserves_shape_and_values(self):
        field = np.arange(4**3).reshape(4, 4, 4)
        output = cube_symmetry(field, seed=19)
        self.assertEqual(output.shape, field.shape)
        np.testing.assert_array_equal(np.sort(output.ravel()), np.sort(field.ravel()))

    def test_shell_balanced_existing_subpanel_selection(self):
        entries = []
        metadata = []
        for shell in range(4):
            for index in range(8):
                core = shell * 100 + index
                entries.append({"core_id": core})
                metadata.append({"core_id": core, "shell": shell})
        panel = {"selected_core_metadata": metadata}
        chosen = selected_autopsy_entries(panel, entries, expected=8)
        shells = [next(row["shell"] for row in metadata if row["core_id"] == value["core_id"]) for value in chosen]
        np.testing.assert_array_equal(np.bincount(shells, minlength=4), [2, 2, 2, 2])

    def test_donor_order_requires_shape_then_prefers_shell_cap_support(self):
        entries = [
            {"shape": [4, 4, 4], "shell": 1, "cap": 0, "support_fraction": 0.7, "core_id": 1},
            {"shape": [4, 4, 4], "shell": 2, "cap": 1, "support_fraction": 0.2, "core_id": 2},
            {"shape": [5, 4, 4], "shell": 1, "cap": 0, "support_fraction": 0.7, "core_id": 3},
        ]
        ordered = _donor_order(
            entries, shape=(4, 4, 4), shell=1, cap=0, support_fraction=0.72, seed=1
        )
        self.assertEqual([row["core_id"] for row in ordered], [1, 2])
        with self.assertRaises(RuntimeError):
            _donor_order(entries, shape=(3, 3, 3), shell=1, cap=0, support_fraction=1.0, seed=1)

    def test_config_rejects_blind_path(self):
        payload = {
            "schema_version": "p12f-causal-autopsy-v1",
            "roles": {
                "diagnostic_validation": "ph006",
                "sealed_blind_test": "ph001",
            },
            "sources": {"bad": "/tmp/ph001/truth.npy"},
            "guards": {
                "ph001_opened": False,
                "ph006_recalibration_allowed": False,
                "may_change_no_field_finalist": False,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(__import__("json").dumps(payload))
            with self.assertRaises(PermissionError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
