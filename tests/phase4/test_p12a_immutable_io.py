from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from workflows.sbi.p12a_immutable_io import (
    write_json_exclusive,
    write_or_validate_npz_exclusive,
)


class P12AImmutableIOTest(unittest.TestCase):
    def test_json_publication_is_exclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "marker.json"
            write_json_exclusive(path, {"pass": True})
            self.assertEqual(json.loads(path.read_text()), {"pass": True})
            with self.assertRaises(FileExistsError):
                write_json_exclusive(path, {"pass": True})

    def test_npz_orphan_adoption_requires_exact_arrays(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "score.npz"
            values = np.arange(8, dtype=np.float64)
            write_or_validate_npz_exclusive(path, values=values)
            write_or_validate_npz_exclusive(path, values=values.copy())
            with self.assertRaises(RuntimeError):
                write_or_validate_npz_exclusive(path, values=values + 1.0)


if __name__ == "__main__":
    unittest.main()
