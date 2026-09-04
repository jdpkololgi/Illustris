#!/usr/bin/env python3
"""Quantify the frozen conditional-standardization round trip on training cores.

F3-L2d/D2 standardizes in real space and then projects into the registered
low-k subspace.  Because a spatially varying location/scale does not commute
with that projection, reconstructing the standardized target need not recover
the original low-mode target exactly.  This preflight records that inherited
approximation before D2 optimization; it is descriptive and cannot be tuned on
ph006.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from workflows.abacus_tweb.p8_deterministic_common import (
    acquire_run_lock,
    atomic_json,
    sha256,
)
from workflows.sbi.p12f3_conditional_models import reconstruct_conditional_low
from workflows.sbi.p12f3_d2_contract import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    digest,
    utc_now,
    validate_frozen_contract,
    validate_output_root,
)
from workflows.sbi.p12f3_d2_train import build_d2_example
from workflows.sbi.p12f3_d2_models import configure_d2_determinism
from workflows.sbi.p12f3_fourier_modes import (
    pack_fourier_components,
    unpack_fourier_components,
    unwhiten_components,
)
from workflows.sbi.p12f3_train_conditional_generative import (
    load_config as load_conditional,
    load_location_scale,
)
from workflows.sbi.p12f3_train_fourier_lowmode_flow import _open_common
from workflows.sbi.p12f3_train_lowmode_flow import load_g1_model


SCHEMA = "p12f3-d2-transform-roundtrip-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def roundtrip_metrics(
    standardized_vector: torch.Tensor,
    target_low: torch.Tensor,
    location: torch.Tensor,
    log_scale: torch.Tensor,
    layout,
    whitening: dict,
) -> dict:
    standard_field = unpack_fourier_components(
        unwhiten_components(standardized_vector, whitening, layout), layout
    )
    reconstructed = reconstruct_conditional_low(
        standard_field[:, 0], location, log_scale, layout
    )
    truth = target_low[:, 0]
    error = reconstructed - truth
    truth_rms = torch.sqrt(torch.mean(torch.square(truth)))
    rmse = torch.sqrt(torch.mean(torch.square(error)))
    reconstructed_components = pack_fourier_components(
        reconstructed[:, None], layout
    )[0]
    truth_components = pack_fourier_components(target_low, layout)[0]
    groups = torch.as_tensor(layout.component_group // 2, device=truth.device)
    power_ratios = []
    for band in range(2):
        mask = groups == band
        denominator = torch.sum(torch.square(truth_components[mask]))
        numerator = torch.sum(torch.square(reconstructed_components[mask]))
        power_ratios.append(float((numerator / denominator).detach().cpu()))
    return {
        "physical_lowmode_rmse": float(rmse.detach().cpu()),
        "physical_lowmode_truth_rms": float(truth_rms.detach().cpu()),
        "relative_rmse": float((rmse / torch.clamp(truth_rms, min=1.0e-12)).detach().cpu()),
        "reconstructed_to_target_power_by_registered_band": power_ratios,
        "finite": bool(
            torch.all(torch.isfinite(reconstructed))
            and torch.all(torch.isfinite(error))
            and np.all(np.isfinite(power_ratios))
        ),
    }


def main() -> None:
    args = parse_args()
    contract_path = args.contract or args.output_root / "D2_CONTRACT_FROZEN.json"
    contract, config = validate_frozen_contract(contract_path, args.config)
    validate_output_root(contract, args.output_root, contract_path)
    deterministic_runtime = configure_d2_determinism(
        config["reproducibility"], args.device
    )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("D2 transform round-trip requires a compute GPU")
    conditional, f3_parent, _ = load_conditional(
        Path(config["sources"]["parent_config"])
        if Path(config["sources"]["parent_config"]).is_absolute()
        else Path(__file__).resolve().parents[2] / config["sources"]["parent_config"]
    )
    phases = tuple(contract["frozen"]["training_phases"])
    refs = contract["frozen"]["internal_selection_refs"]
    representative = []
    for phase in phases:
        row = next(item for item in refs if item["phase"] == phase)
        representative.append((phase, int(row["core_id"])))
    frozen = {
        "contract": str(contract_path.resolve()),
        "contract_sha256": sha256(contract_path),
        "contract_digest": contract["frozen_digest"],
        "deterministic_runtime": deterministic_runtime,
        "representative_training_cores": [
            {"phase": phase, "core_id": core_id} for phase, core_id in representative
        ],
        "definition": (
            "reconstruct L[(target_low-location)/scale] through "
            "L[location+scale*standardized] and compare with target_low"
        ),
        "role": "descriptive inherited-transform audit; no ph006-fitted threshold",
        "ph006_used": False,
        "ph001_opened": False,
    }
    frozen_digest = digest(frozen)
    output = args.output_root / "D2_TRANSFORM_ROUNDTRIP.json"
    lock = acquire_run_lock(
        args.output_root / ".transform_roundtrip.lock",
        purpose="P12-F3-D2 training-only transform roundtrip",
    )
    loader = store = None
    try:
        if output.exists():
            marker = json.loads(output.read_text())
            if (
                marker.get("schema_version") != SCHEMA
                or not marker.get("technical_pass")
                or marker.get("frozen_digest") != frozen_digest
                or marker.get("ph006_used")
                or marker.get("ph001_opened")
            ):
                raise RuntimeError("existing D2 transform round-trip changed")
            print(json.dumps(marker, indent=2, sort_keys=True))
            return

        _, _, opened_phases, _, _, loader, store, selected = _open_common(f3_parent)
        if tuple(opened_phases) != phases or selected != contract["frozen"]["selected_core_ids"]:
            raise RuntimeError("D2 round-trip inherited core contract changed")
        conditional_root = Path(config["sources"]["conditional_output_root"])
        location_model, _, _, _ = load_location_scale(
            SimpleNamespace(
                output_root=conditional_root,
                gaussian_arm=config["sources"]["conditional_gaussian_arm"],
                gaussian_run=config["sources"]["conditional_gaussian_run"],
            ),
            conditional,
            args.device,
        )
        g1_model, scaler = load_g1_model(f3_parent, args.device)
        whitening_marker = json.loads(Path(config["sources"]["conditional_whitening"]).read_text())
        whitening = whitening_marker["whitening"]
        rows = []
        for phase, core_id in representative:
            (
                _,
                vector,
                layout,
                location,
                log_scale,
                _,
                _,
                target_low,
            ) = build_d2_example(
                loader=loader,
                store=store,
                g1_model=g1_model,
                location_model=location_model,
                scaler=scaler,
                phase=phase,
                core_id=core_id,
                conditional_config=conditional,
                f3_parent=f3_parent,
                device=args.device,
                whitening=whitening,
            )
            rows.append(
                {
                    "phase": phase,
                    "core_id": core_id,
                    **roundtrip_metrics(
                        vector,
                        target_low,
                        location,
                        log_scale,
                        layout,
                        whitening,
                    ),
                }
            )
        technical_pass = bool(all(row["finite"] for row in rows))
        marker = {
            "schema_version": SCHEMA,
            "created_utc": utc_now(),
            "pass": technical_pass,
            "technical_pass": technical_pass,
            "frozen": frozen,
            "frozen_digest": frozen_digest,
            "per_core": rows,
            "summary": {
                "maximum_relative_rmse": max(row["relative_rmse"] for row in rows),
                "median_relative_rmse": float(np.median([row["relative_rmse"] for row in rows])),
                "minimum_and_maximum_band_power_ratio": [
                    float(min(value for row in rows for value in row["reconstructed_to_target_power_by_registered_band"])),
                    float(max(value for row in rows for value in row["reconstructed_to_target_power_by_registered_band"])),
                ],
            },
            "interpretation": (
                "Quantifies an approximation inherited unchanged from F3-L2d. "
                "Finite completion licenses a matched D2 comparison but does not establish an exact inverse."
            ),
            "truth_files_read": [f"{phase} training delta_R7" for phase in phases],
            "ph006_used": False,
            "ph001_opened": False,
        }
        if not technical_pass:
            raise RuntimeError("non-finite D2 conditional-transform round trip")
        atomic_json(output, marker)
        print(json.dumps(marker, indent=2, sort_keys=True))
    finally:
        if store is not None:
            store.close()
        if loader is not None:
            loader.close()
        lock.close()


if __name__ == "__main__":
    main()
