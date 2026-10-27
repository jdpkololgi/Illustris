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
    terminal_export_checkpoint,
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

    def test_multiple_validation_phases_are_explicitly_supported(self):
        row = self.checkpoint()
        row['validation_phase'] = None
        row['validation_phases'] = ['ph006', 'ph012', 'ph013']
        for phase in row['validation_phases']:
            validate_oof_checkpoint(row, phase, 32)
        with self.assertRaises(RuntimeError):
            validate_oof_checkpoint(row, 'ph014', 32)

    def test_in_sample_checkpoint_fails(self):
        row = self.checkpoint()
        row["training_phases"].append("ph005")
        with self.assertRaises(RuntimeError):
            validate_oof_checkpoint(row, "ph005", 32)

    def test_terminal_export_requires_complete_fixed_budget(self):
        from types import SimpleNamespace
        from copy import deepcopy
        from workflows.abacus_tweb.p10_training_contract import sha256
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            completion={'epochs_completed':2,'global_steps':6,'training_cores_per_epoch':3,
                        'source_contract':{'train':'bound'},'training_phases':['ph000']}
            for key,relative in [('training_ready_marker_sha256','TRAINING_LOADER_READY.json'),
                                 ('target_scaler_sha256','transforms/target_scaler.json'),
                                 ('field_transform_sha256','transforms/field/field_transform.json')]:
                path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
                completion[key]=sha256(path)
            state={'schema_version':'p10-arm-a-checkpoint-v1','model':'unet',
                   'frozen_arguments':{'disable_early_stopping':True,'epochs':2},
                   'global_step':6,'resume':{'epoch':2,'cursor':3},'source_contract':{'train':'bound'},
                   'model_state':{'weight':torch.tensor([7.])}}
            loader=SimpleNamespace(root=root,training_phases=('ph000',),validation_phases=('ph002',),
                                   target_scaler={},field_normalization={})
            result=terminal_export_checkpoint(state,completion,loader,2)
            self.assertEqual(result['selection_policy'],'terminal_fixed_budget')
            self.assertEqual(result['state_dict']['weight'].item(),7.)
            for change in ({'schema_version':'p10-arm-a-best-v1'},
                           {'resume':{'epoch':2,'cursor':2}},
                           {'frozen_arguments':{'disable_early_stopping':False,'epochs':2}},
                           {'source_contract':{'train':'different'}}):
                bad=deepcopy(state);bad.update(change)
                with self.assertRaises(RuntimeError):terminal_export_checkpoint(bad,completion,loader,2)
            bad=deepcopy(state);bad['global_step']=5
            incomplete=deepcopy(completion);incomplete['global_steps']=5
            with self.assertRaises(RuntimeError):terminal_export_checkpoint(bad,incomplete,loader,2)
            (root/'transforms/target_scaler.json').write_text('{"changed":true}')
            with self.assertRaises(RuntimeError):terminal_export_checkpoint(state,completion,loader,2)

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
