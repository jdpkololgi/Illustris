from types import SimpleNamespace
import unittest

import numpy as np

from workflows.abacus_tweb.p10_r3_random_field_contract import R3_RF_MODEL_CHANNELS
from workflows.abacus_tweb.p10_train_r3_random_field import r3_rf_model_inputs


class TestP10R3RandomField(unittest.TestCase):
    def test_registered_six_channel_interface(self):
        self.assertEqual(len(R3_RF_MODEL_CHANNELS), 6)
        self.assertEqual(
            R3_RF_MODEL_CHANNELS[3:],
            ("expected_counts_random", "angular_response", "support_random"),
        )
        self.assertFalse(any("faint" in name.lower() for name in R3_RF_MODEL_CHANNELS))

    def test_model_mapping_is_bright_plus_random_triplet(self):
        values = np.zeros((6, 2, 2, 2), dtype=np.float32)
        values[0] = 3.0
        values[1] = 0.75
        values[2] = 0.5
        values[3] = 1.0
        values[4] = 1.4
        values[5] = 1.0
        patch = SimpleNamespace(
            values=values,
            channel_names=R3_RF_MODEL_CHANNELS,
            authoritative_frac_index_local=np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32),
        )
        normalization = {"channels": {
            "counts": {"policy": "zscore", "mean": 0.0, "std": 1.0},
            "exposure_apodized": {"policy": "identity"},
            "log_count_ratio": {"policy": "zscore", "mean": 0.0, "std": 1.0},
            "expected_counts_random": {
                "policy": "zscore", "pretransform": "log1p", "mean": 0.5, "std": 0.25
            },
            "angular_response": {"policy": "identity"},
            "support_random": {"policy": "identity"},
        }}
        tensor, points = r3_rf_model_inputs(patch, normalization, "cpu")
        output = tensor.numpy()[0]
        self.assertEqual(output.shape, (6, 2, 2, 2))
        self.assertTrue(np.allclose(output[2], 0.75))
        self.assertTrue(np.allclose(output[3], (np.log(2.0) - 0.5) / 0.25))
        self.assertTrue(np.allclose(output[4], 0.4))
        self.assertTrue(np.allclose(output[5], 1.0))
        self.assertEqual(tuple(points.shape), (1, 1, 1, 1, 3))


if __name__ == "__main__":
    unittest.main()

