from __future__ import annotations

import unittest

import numpy as np

from workflows.abacus_tweb.p8_train_multitracer_unet_patch import zscore


class MultitracerUNetTests(unittest.TestCase):
    def test_faint_count_normalization_uses_frozen_log1p_contract(self) -> None:
        values = np.array([0.0, np.e - 1.0], dtype=np.float32)
        result = zscore(values, {"mean": 0.5, "std": 0.5})
        np.testing.assert_allclose(result, [-1.0, 1.0], rtol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
