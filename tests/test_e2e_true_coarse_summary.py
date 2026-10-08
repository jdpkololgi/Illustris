"""Focused safeguards for the new diagnostic reporting/foreground wrapper."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'workflows/sbi'))
import e2e_true_coarse_fraction_summary as summary
import e2e_true_coarse_fraction_worker as worker


class SummaryTests(unittest.TestCase):
    def test_closure_error_preserves_class_order(self):
        target = np.full((3, 2, 4), .25)
        shift = np.array([.01, -.02, .03, -.02])
        got = summary.closure_error(target+shift, target)
        np.testing.assert_allclose(got['bias'], shift)
        np.testing.assert_allclose(got['rms'], abs(shift))
        np.testing.assert_allclose(got['maxabs'], abs(shift))

    def test_worker_rejects_unregistered_addresses_without_launch(self):
        with patch.object(worker.io, 'compute_guard'), patch.object(worker.subprocess, 'run') as launch:
            for addresses in ([], ['ph014:SGC:0'], ['ph012:SGC:1'],
                    ['ph012:SGC:0']*2, ['ph012:SGC:0', 'ph012:SGC:4', 'ph012:SGC:8']):
                with self.subTest(addresses=addresses), self.assertRaises(ValueError):
                    worker.run(addresses)
            launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
