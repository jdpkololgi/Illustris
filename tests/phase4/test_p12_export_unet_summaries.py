import unittest

import numpy as np
import torch

from workflows.sbi.p12_export_unet_summaries import (
    ntilde_at_rows,
    validate_oof_checkpoint,
)


class P12ExportUnetSummaryTests(unittest.TestCase):
    def checkpoint(self):
        return {
            "schema_version": "p10-arm-a-best-v1",
            "model": "unet",
            "training_phases": ["ph000", "ph002", "ph003", "ph004"],
            "validation_phase": "ph005",
            "state_dict": {"unet.output.weight": torch.zeros((32, 24, 1, 1, 1))},
        }

    def test_out_of_fold_checkpoint_passes(self):
        validate_oof_checkpoint(self.checkpoint(), "ph005", 32)

    def test_in_sample_checkpoint_fails(self):
        row = self.checkpoint()
        row["training_phases"].append("ph005")
        with self.assertRaises(RuntimeError):
            validate_oof_checkpoint(row, "ph005", 32)

    def test_ntilde_is_cap_specific(self):
        selection = {"rotations": {"0": {"caps": {
            "SGC": {"grid_z": [0.1, 0.6], "ntilde": [1.0, 2.0]},
            "NGC": {"grid_z": [0.1, 0.6], "ntilde": [3.0, 5.0]},
        }}}}
        result = ntilde_at_rows(
            selection,
            np.asarray([0, 1], dtype=np.uint8),
            np.asarray([0.35, 0.35]),
        )
        np.testing.assert_allclose(result, [1.5, 4.0])


if __name__ == "__main__":
    unittest.main()
