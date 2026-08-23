#!/usr/bin/env python3
"""P10 R2 trainer with explicit, audited fibre-assignment response channels."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb import p10_train_arm_a as trainer
from workflows.abacus_tweb import p8_train_unet_patch as unet_impl
from workflows.abacus_tweb.p10_build_r2_assignment_overlays import R2_MODEL_CHANNELS
from workflows.abacus_tweb.p10_r2_training_contract import P10AssignmentResponseLoader
from workflows.abacus_tweb.p6_field_patch_utils import apply_frozen_normalization
from workflows.abacus_tweb.p8_deterministic_common import sha256


def requested_model(arguments: list[str]) -> str | None:
    for index, value in enumerate(arguments):
        if value == "--model" and index + 1 < len(arguments):
            return arguments[index + 1]
        if value.startswith("--model="):
            return value.split("=", 1)[1]
    return None


if requested_model(sys.argv[1:]) not in (None, "unet"):
    raise SystemExit("P10 R2 is registered only for U-PATCH")


class R2UPatch(unet_impl.UPatch):
    def __init__(self, base: int = 24, latent_channels: int = 32, head_width: int = 128):
        super().__init__(base=base, latent_channels=latent_channels, head_width=head_width)
        self.unet = unet_impl.UNet3D(len(R2_MODEL_CHANNELS), latent_channels, base)


def r2_model_inputs(patch, normalization: dict, device: str):
    normalized = apply_frozen_normalization(patch, normalization)
    at = {name: index for index, name in enumerate(patch.channel_names)}
    density_proxy = np.clip(
        np.expm1(np.clip(patch.values[at["log_count_ratio"]], -20.0, 4.0)),
        -1.0,
        20.0,
    )
    values = np.stack(
        (
            normalized[at["counts"]],
            density_proxy,
            patch.values[at["exposure_apodized"]],
            patch.values[at["c_fibre_tileloc"]],
            patch.values[at["c_fibre_tiles"]],
            patch.values[at["c_fibre_defined"]],
        )
    ).astype(np.float32)
    tensor = torch.from_numpy(values[None]).to(device)
    points = unet_impl.grid_coordinates(
        patch.authoritative_frac_index_local, tuple(values.shape[1:]), device
    )
    return tensor, points


trainer.P10PhaseBalancedLoader = P10AssignmentResponseLoader
unet_impl.CHANNELS = R2_MODEL_CHANNELS
unet_impl.UPatch = R2UPatch
unet_impl.model_inputs = r2_model_inputs
_source_contract = trainer.source_contract
_atomic_json = trainer.atomic_json


def response_source_contract() -> dict[str, str]:
    result = _source_contract()
    for path in (
        Path(__file__).resolve(),
        REPO_ROOT / "workflows/abacus_tweb/p10_r2_training_contract.py",
        REPO_ROOT / "workflows/abacus_tweb/p10_build_r2_assignment_overlays.py",
    ):
        result[str(path.relative_to(REPO_ROOT))] = sha256(path)
    return result


def response_atomic_json(path: Path, payload: dict) -> None:
    if (
        Path(path).name == "run_manifest.json"
        or payload.get("stage") == "P10 deterministic multi-phase Arm A final-view R0"
    ):
        payload = dict(payload)
        payload["schema_version"] = "p10-r2-assignment-response-run-v1"
        payload["stage"] = "P10 deterministic multi-phase assignment-response R2"
        payload["view"] = (
            "R2: frozen R1 random response plus FRACZ_TILELOCID, "
            "FRAC_TLOBS_TILES and C_fibre_defined"
        )
        payload["response_scope"] = (
            "mock assignment response is informative; mock C_z is constant and deliberately "
            "excluded from model inputs"
        )
    _atomic_json(path, payload)


trainer.source_contract = response_source_contract
trainer.atomic_json = response_atomic_json


if __name__ == "__main__":
    trainer.main()
