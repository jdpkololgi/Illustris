from __future__ import annotations

import unittest

import numpy as np

from workflows.sbi.p12a_blind_throughput_smoke import (
    core_aligned_stop,
    projected_four_gpu_seconds,
)


class P12ABlindThroughputSmokeTest(unittest.TestCase):
    def test_core_aligned_stop_never_splits_a_core(self):
        core = np.repeat(np.arange(6), [2, 3, 4, 1, 5, 2])
        stop = core_aligned_stop(core, 7)
        self.assertGreaterEqual(stop, 7)
        self.assertNotEqual(core[stop - 1], core[stop])

    def test_projection_uses_four_independent_gpus(self):
        self.assertEqual(projected_four_gpu_seconds(20.0, 100, 1000), 50.0)
        with self.assertRaises(ValueError):
            projected_four_gpu_seconds(0.0, 100, 1000)


if __name__ == "__main__":
    unittest.main()
