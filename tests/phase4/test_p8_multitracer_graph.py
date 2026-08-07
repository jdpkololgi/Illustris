from __future__ import annotations

import unittest

import numpy as np

from workflows.abacus_tweb.p8_build_multitracer_graph_adapter import bright_prefix_rows


class MultitracerGraphTests(unittest.TestCase):
    def test_bright_prefix_contract(self) -> None:
        index = {
            "tracer_type": np.array([0, 0, 1, 1], dtype=np.uint8),
            "bright_parent_id": np.array([0, 1, -1, -1], dtype=np.int64),
        }
        self.assertEqual(bright_prefix_rows(index), 2)

    def test_interleaved_tracers_rejected(self) -> None:
        index = {
            "tracer_type": np.array([0, 1, 0], dtype=np.uint8),
            "bright_parent_id": np.array([0, -1, 2], dtype=np.int64),
        }
        with self.assertRaises(RuntimeError):
            bright_prefix_rows(index)


if __name__ == "__main__":
    unittest.main()
