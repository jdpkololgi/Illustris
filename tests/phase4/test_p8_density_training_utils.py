import unittest

import numpy as np
import torch

from workflows.abacus_tweb.p8_density_training_utils import extract_core_prediction


class DensityTrainingUtilsTests(unittest.TestCase):
    def test_extract_core_prediction_uses_exact_three_axis_slice(self):
        values = torch.arange(1 * 1 * 8 * 9 * 10, dtype=torch.float32).reshape(1, 1, 8, 9, 10)
        core = (slice(2, 6), slice(3, 8), slice(1, 9))
        got = extract_core_prediction(values, core)
        np.testing.assert_array_equal(got.numpy(), values.numpy()[0, 0, 2:6, 3:8, 1:9])

    def test_extract_core_prediction_rejects_non_scalar_field(self):
        with self.assertRaises(ValueError):
            extract_core_prediction(torch.zeros((1, 2, 8, 8, 8)), (slice(None),) * 3)


if __name__ == "__main__":
    unittest.main()
