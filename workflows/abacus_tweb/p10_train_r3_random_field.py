#!/usr/bin/env python3
"""P10 R3-RF trainer with the high-S/N empirical random response field."""
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
from workflows.abacus_tweb.p10_r3_random_field_contract import (
    P10RawRandomFieldLoader,
    R3_RF_MODEL_CHANNELS,
)
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
    raise SystemExit("P10 R3-RF is registered only for U-PATCH")


class R3RFUPatch(unet_impl.UPatch):
    def __init__(self, base: int = 24, latent_channels: int = 32, head_width: int = 128):
        super().__init__(base=base, latent_channels=latent_channels, head_width=head_width)
        self.unet = unet_impl.UNet3D(len(R3_RF_MODEL_CHANNELS), latent_channels, base)


def r3_rf_model_inputs(patch, normalization: dict, device: str):
    normalized = apply_frozen_normalization(patch, normalization)
    at = {name: index for index, name in enumerate(patch.channel_names)}
    bright_density = np.clip(
        np.expm1(np.clip(patch.values[at["log_count_ratio"]], -20.0, 4.0)),
        -1.0,
        20.0,
    ).astype(np.float32)
    random_spec = normalization["channels"]["expected_counts_random"]
    random_count = np.log1p(
        np.maximum(patch.values[at["expected_counts_random"]], 0.0)
    )
    random_count = (
        (random_count - np.float32(random_spec["mean"]))
        / np.float32(max(random_spec["std"], 1.0e-6))
    ).astype(np.float32)
    random_response = np.clip(
        patch.values[at["angular_response"]] - 1.0, -1.0, 20.0
    ).astype(np.float32)
    values = np.stack(
        (
            normalized[at["counts"]],
            bright_density,
            patch.values[at["exposure_apodized"]],
            random_count,
            random_response,
            patch.values[at["support_random"]],
        )
    ).astype(np.float32)
    tensor = torch.from_numpy(values[None]).to(device)
    points = unet_impl.grid_coordinates(
        patch.authoritative_frac_index_local, tuple(values.shape[1:]), device
    )
    return tensor, points


trainer.P10PhaseBalancedLoader = P10RawRandomFieldLoader
unet_impl.CHANNELS = R3_RF_MODEL_CHANNELS
unet_impl.UPatch = R3RFUPatch
unet_impl.model_inputs = r3_rf_model_inputs
_source_contract = trainer.source_contract
_atomic_json = trainer.atomic_json


def response_source_contract() -> dict[str, str]:
    result = _source_contract()
    for path in (
        Path(__file__).resolve(),
        REPO_ROOT / "workflows/abacus_tweb/p10_r3_random_field_contract.py",
        REPO_ROOT / "workflows/abacus_tweb/p10_prepare_r3_random_field_contract.py",
    ):
        result[str(path.relative_to(REPO_ROOT))] = sha256(path)
    return result


def response_atomic_json(path: Path, payload: dict) -> None:
    if (
        Path(path).name == "run_manifest.json"
        or payload.get("stage") == "P10 deterministic multi-phase Arm A final-view R0"
    ):
        payload = dict(payload)
        payload["schema_version"] = "p10-r3-rf-run-v1"
        payload["stage"] = "P10 deterministic multi-phase raw-random-field R3-RF"
        payload["view"] = (
            "R3-RF: BRIGHT count/density/exposure plus all-18 expected random "
            "intensity, angular response and binary support"
        )
        payload["response_scope"] = (
            "high-S/N voxel-resolved empirical BRIGHT response; no FAINT context, "
            "no clustering-random redshift and no density-matched stochastic control"
        )
    _atomic_json(path, payload)


trainer.source_contract = response_source_contract
trainer.atomic_json = response_atomic_json


if __name__ == "__main__":
    trainer.main()

