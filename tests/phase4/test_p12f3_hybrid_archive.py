from __future__ import annotations

import unittest

import numpy as np

from workflows.sbi.p12f3_export_hybrid_archive import (
    lowpass_numpy,
    selected_subpanel,
)


class P12F3HybridArchiveTests(unittest.TestCase):
    def test_registered_subpanel_is_balanced_and_truth_independent(self):
        rows = []
        for shell in range(4):
            for index in range(256):
                rows.append({"core_id": shell * 10_000 + index, "shell": shell})
        selected = selected_subpanel({"selected_core_metadata": rows})
        self.assertEqual(len(selected), 256)
        self.assertEqual(
            np.bincount([row["shell"] for row in selected], minlength=4).tolist(),
            [64, 64, 64, 64],
        )
        self.assertEqual([row["core_id"] for row in selected[:3]], [0, 4, 8])

    def test_lowpass_excludes_dc_and_rejects_short_wave(self):
        shape = (24, 24, 24)
        x = np.arange(shape[0], dtype=np.float32)[:, None, None]
        constant = np.full(shape, 7.0, dtype=np.float32)
        long_wave = np.broadcast_to(
            np.sin(2.0 * np.pi * x / shape[0]), shape
        ).astype(np.float32)
        short_wave = np.broadcast_to(
            np.sin(2.0 * np.pi * 10.0 * x / shape[0]), shape
        ).astype(np.float32)
        values = np.stack((constant, long_wave, short_wave))
        filtered = lowpass_numpy(
            values, voxel_mpc_h=5.0, maximum_k=0.1813799364234218
        )
        self.assertLess(float(np.max(np.abs(filtered[0]))), 1e-5)
        self.assertLess(float(np.max(np.abs(filtered[1] - long_wave))), 1e-5)
        self.assertLess(float(np.max(np.abs(filtered[2]))), 1e-5)


if __name__ == "__main__":
    unittest.main()
