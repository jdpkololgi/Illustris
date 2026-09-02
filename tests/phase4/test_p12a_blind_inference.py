from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from workflows.abacus_tweb.p8_deterministic_common import sha256
from workflows.sbi import p12a_blind_inference as blind
from workflows.sbi.p12_production_contract import P12A_SCHEMA


class _FakeAdapter:
    def __init__(self, root, **_):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "adapter_manifest.json").read_text())
        self.core_cap = np.asarray([0], dtype=np.int8)

    def extract(self, core_id, *_args, **_kwargs):
        if core_id != 0:
            raise AssertionError("unexpected core")
        return SimpleNamespace(authoritative_parent_id=np.asarray([0], dtype=np.int64))

    def close(self):
        return None


class _FakeUPatch:
    constructed_base = None

    def __init__(self, base=24, latent_channels=32):
        type(self).constructed_base = base
        self.latent_channels = latent_channels

    def to(self, _device):
        return self

    def load_state_dict(self, _state):
        return None

    def eval(self):
        return self

    def sample_latent(self, _values, _coordinates):
        return torch.zeros((1, self.latent_channels), dtype=torch.float32)

    def head(self, latent):
        return torch.zeros((len(latent), 3), dtype=torch.float32)


class P12ABlindInferenceTest(unittest.TestCase):
    def test_context_uses_numeric_base_and_stays_truth_free(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter = root / "ph001" / "adapter"
            adapter.mkdir(parents=True)
            p3_manifest = root / "ph001" / "p3_fields" / "field_manifest.json"
            p3_manifest.parent.mkdir(parents=True)
            p3_manifest.write_text(json.dumps({"phase": "ph001", "pass": True}))
            cap_field = root / "ph001" / "p3_fields" / "ngc_fields.h5"
            cap_field.write_bytes(b"field placeholder")
            (adapter / "adapter_manifest.json").write_text(
                json.dumps(
                    {
                        "phase": "ph001",
                        "ph001_opened": False,
                        "pass": True,
                        "channel_order": list(blind.unet_impl.CHANNELS),
                        "p3_manifest": str(p3_manifest),
                        "p3_manifest_sha256": sha256(p3_manifest),
                        "p4_active_assignment": str(root / "ph001" / "active_assignment.npz"),
                        "points": str(root / "ph001" / "points.npy"),
                        "caps": {"NGC": {"field_path": str(cap_field)}},
                    }
                )
            )
            assignment = root / "ph001" / "active_assignment.npz"
            np.savez(
                assignment,
                parent_node_id=np.asarray([0], dtype=np.int64),
                supervised_eligible=np.asarray([True]),
                cap=np.asarray([0], dtype=np.uint8),
                shell=np.asarray([0], dtype=np.int8),
            )
            points = root / "ph001" / "points.npy"
            np.save(points, np.asarray([[1.0, 2.0, 3.0, 0.0, 0.2]], dtype=np.float32))
            adapter_manifest_path = adapter / "adapter_manifest.json"
            adapter_manifest = json.loads(adapter_manifest_path.read_text())
            adapter_manifest["p4_active_assignment_sha256"] = sha256(assignment)
            adapter_manifest_path.write_text(json.dumps(adapter_manifest))
            response = root / "response.json"
            response.write_text(
                json.dumps(
                    {
                        "phase": "ph001",
                        "pass": True,
                        "ph001_opened": False,
                        "truth_files_read": [],
                    }
                )
            )
            selection = root / "selection.json"
            selection.write_text(
                json.dumps(
                    {
                        "pass": True,
                        "fit_phases": ["ph000", "ph002", "ph003", "ph004", "ph005"],
                        "application_phases": ["ph006", "ph001"],
                        "gates": {"no_validation_or_blind_fit": True},
                    }
                )
            )
            checkpoint_path = root / "unet.pt"
            checkpoint_path.write_bytes(b"frozen checkpoint placeholder")
            candidate = root / "candidate.json"
            candidate.write_text(
                json.dumps(
                    {
                        "schema_version": P12A_SCHEMA,
                        "pass": True,
                        "truth_files_read": [],
                        "open_count": 0,
                        "sealed_phase_opened": False,
                        "base_encoder": {
                            "selected_epoch": 20,
                            "response_aware_encoder": False,
                            "checkpoint": {"sha256": sha256(checkpoint_path)},
                        },
                    }
                )
            )
            checkpoint = {
                "schema_version": "p10-arm-a-best-v1",
                "model": "unet",
                "validation_phase": "ph006",
                "training_phases": ("ph000", "ph002", "ph003", "ph004", "ph005"),
                "state_dict": {
                    "unet.output.weight": torch.zeros((32, 1, 1, 1, 1))
                },
                "normalization": {},
                "scaler": {},
            }
            output = root / "ph001_context.npz"
            _FakeUPatch.constructed_base = None
            with (
                mock.patch.object(blind, "CanonicalFieldPatchAdapter", _FakeAdapter),
                mock.patch.object(blind, "torch_load", return_value=checkpoint),
                mock.patch.object(blind.unet_impl, "UPatch", _FakeUPatch),
                mock.patch.object(
                    blind.unet_impl,
                    "model_inputs",
                    return_value=(torch.zeros((1, 3, 2, 2, 2)), torch.zeros((1, 1, 1, 1, 3))),
                ),
                mock.patch.object(blind, "unscale_increments", side_effect=lambda x, _: x),
                mock.patch.object(blind, "increments_to_eigenvalues", side_effect=lambda x: x),
                mock.patch.object(
                    blind,
                    "sample_random_support_distance",
                    return_value=(np.asarray([10.0]), np.asarray([True])),
                ),
                mock.patch.object(
                    blind,
                    "ntilde_at_rows",
                    return_value=np.asarray([1.0e-4], dtype=np.float32),
                ),
            ):
                manifest = blind.export_blind_unet_context(
                    adapter_root=adapter,
                    assignment_path=assignment,
                    points_path=points,
                    response_field_manifest_path=response,
                    selection_manifest_path=selection,
                    candidate_marker_path=candidate,
                    checkpoint_path=checkpoint_path,
                    output_path=output,
                    device="cpu",
                    base=24,
                    audit_rows=1,
                )
            self.assertEqual(_FakeUPatch.constructed_base, 24)
            self.assertTrue(manifest["pass"])
            self.assertEqual(manifest["truth_files_read"], [])
            with np.load(output) as archive:
                blind.validate_context_archive(archive)
                self.assertEqual(archive["context"].shape, (1, 7))
                self.assertNotIn("truth", archive.files)


if __name__ == "__main__":
    unittest.main()
