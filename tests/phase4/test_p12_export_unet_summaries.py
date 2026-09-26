import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from workflows.sbi.p12_export_unet_summaries import (
    main,
    ntilde_at_rows,
    parent_to_assignment_index,
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

    def test_existing_export_is_never_overwritten(self):
        with TemporaryDirectory() as tmp:
            phase = Path(tmp) / "ph002"
            phase.mkdir()
            old = phase / "parent_node_id.npy"
            old.write_bytes(b"immutable")
            argv = ["export", "--phase", "ph002", "--output-root", tmp,
                    "--checkpoint", "unused", "--contract-root", "unused"]
            with patch("sys.argv", argv), self.assertRaises(FileExistsError):
                main()
            self.assertEqual(old.read_bytes(), b"immutable")

    def test_negative_context_is_rejected_before_loading(self):
        argv = ["export", "--phase", "ph002", "--output-root", "unused",
                "--checkpoint", "unused", "--contract-root", "unused", "--context-halo", "-1"]
        with patch("sys.argv", argv), self.assertRaises(ValueError):
            main()

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

    def test_parent_index_uses_assignment_rows_not_archive_fields(self):
        class AssignmentArchive(dict):
            def __len__(self):
                return 15

        assignment = AssignmentArchive(
            parent_node_id=np.asarray([3, 0, 2], dtype=np.int64)
        )
        np.testing.assert_array_equal(
            parent_to_assignment_index(assignment, 5),
            np.asarray([1, -1, 2, 0, -1], dtype=np.int64),
        )

    def test_parent_index_rejects_duplicates(self):
        assignment = {"parent_node_id": np.asarray([1, 1], dtype=np.int64)}
        with self.assertRaises(RuntimeError):
            parent_to_assignment_index(assignment, 3)


if __name__ == "__main__":
    unittest.main()
