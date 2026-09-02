#!/usr/bin/env python3
"""Freeze the truth-free, response-stratified ph006 P12-F selection panel."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from astropy.cosmology import Planck18
import numpy as np

from workflows.abacus_tweb.p3br_training_contract import P10RandomResponseLoader
from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.sbi.p12f_challenger_common import select_truth_free_panel


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/p12f_matched_challengers_v1.json"
DEFAULT_CONTRACT = Path(
    "/global/homes/d/dkololgi/p11_contracts/"
    "training_contract_r1_random_repair_v2_20260901"
)
SHELL_REDSHIFT = (0.15, 0.25, 0.35, 0.45, 0.55)
OBSERVATION_CHANNELS = (
    "support_random",
    "angular_response",
    "distance_to_support_boundary",
    "counts",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--contract-root", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def core_radius_mpc(grid: dict, patch, support: np.ndarray) -> float:
    """Median radius of supported core voxels using the canonical P3 lattice."""
    if support.shape != patch.core_values.shape[1:] or not np.any(support):
        raise ValueError("core support geometry is invalid")
    axes = [
        float(grid["origin_mpc"][axis])
        + (
            np.arange(support.shape[axis], dtype=np.float64)
            + int(patch.core_start[axis])
            + 0.5
        )
        * float(grid["cell_mpc"])
        for axis in range(3)
    ]
    radius = np.sqrt(
        axes[0][:, None, None] ** 2
        + axes[1][None, :, None] ** 2
        + axes[2][None, None, :] ** 2
    )
    return float(np.median(radius[support]))


def shell_from_radius(radius_mpc: float) -> int:
    bounds = np.asarray(
        [Planck18.comoving_distance(value).value for value in SHELL_REDSHIFT],
        dtype=np.float64,
    )
    shell = int(np.searchsorted(bounds[1:-1], float(radius_mpc), side="right"))
    if not bounds[0] <= radius_mpc < bounds[-1]:
        return -1
    return shell


def summarize_observed_core(adapter, core_id: int) -> dict | None:
    """Summarize only observed response/count fields; no truth object is accepted."""
    patch = adapter.extract(
        int(core_id),
        0,
        OBSERVATION_CHANNELS,
        alignment_voxels=1,
    )
    at = {name: index for index, name in enumerate(patch.channel_names)}
    support_raw = patch.core_values[at["support_random"]]
    if not np.all((support_raw == 0) | (support_raw == 1)):
        raise RuntimeError("support_random is not exactly binary")
    support = support_raw.astype(bool)
    if not np.any(support):
        return None
    response = patch.core_values[at["angular_response"]]
    boundary = patch.core_values[at["distance_to_support_boundary"]]
    counts = patch.core_values[at["counts"]]
    if not np.all(np.isfinite(response[support] + boundary[support] + counts[support])):
        raise RuntimeError("non-finite observation covariate in ph006 core")
    cap_name = "NGC" if int(patch.cap) == 1 else "SGC"
    grid = adapter.manifest["caps"][cap_name]
    radius = core_radius_mpc(grid, patch, support)
    return {
        "core_id": int(core_id),
        "shell": shell_from_radius(radius),
        "cap": int(patch.cap),
        "median_radius_mpc": radius,
        "median_angular_response": float(np.median(response[support])),
        "median_boundary_distance_mpc": float(np.median(boundary[support])),
        "tracer_density_per_supported_voxel": float(np.sum(counts[support]) / support.sum()),
        "supported_voxels": int(support.sum()),
        "authoritative_galaxies": int(len(patch.authoritative_parent_id)),
    }


def build_panel(config: dict, loader: P10RandomResponseLoader) -> tuple[list[dict], np.ndarray]:
    phase = config["roles"]["validation_and_selection"]
    if phase != "ph006" or loader.validation_phase != phase:
        raise RuntimeError("truth-free selection panel is frozen to ph006")
    adapter = loader.field_adapter(phase)
    rows = []
    for core_id in range(len(adapter.core_start)):
        row = summarize_observed_core(adapter, core_id)
        if row is not None:
            rows.append(row)
    if not rows:
        raise RuntimeError("no ph006 cores have exact random support")
    core_id = np.asarray([row["core_id"] for row in rows], dtype=np.int64)
    shell = np.asarray([row["shell"] for row in rows], dtype=np.int8)
    cap = np.asarray([row["cap"] for row in rows], dtype=np.int8)
    response = np.asarray(
        [row["median_angular_response"] for row in rows], dtype=np.float64
    )
    boundary = np.asarray(
        [row["median_boundary_distance_mpc"] for row in rows], dtype=np.float64
    )
    selected_index = select_truth_free_panel(
        core_id=core_id,
        shell=shell,
        cap=cap,
        response=response,
        boundary_distance=boundary,
        per_shell=int(config["matched_contract"]["selection_cores_per_shell"]),
        seed=42,
    )
    return rows, selected_index


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    if config.get("schema_version") != "p12f-matched-challengers-v1":
        raise RuntimeError("unsupported P12-F challenger configuration")
    if config["roles"]["sealed_blind_test"] != "ph001":
        raise PermissionError("P12-F blind phase contract changed")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise RuntimeError("selection-panel output must be new and empty")
    args.output_root.mkdir(parents=True, exist_ok=True)
    loader = P10RandomResponseLoader(args.contract_root, include_blind=False)
    rows, selected_index = build_panel(config, loader)
    selected_rows = [rows[int(index)] for index in selected_index]
    selected_core_id = np.asarray(
        [row["core_id"] for row in selected_rows], dtype=np.int64
    )
    expected = int(config["matched_contract"]["selection_panel_cores"])
    if len(selected_core_id) != expected or len(np.unique(selected_core_id)) != expected:
        raise RuntimeError("selection panel is not complete and unique")
    shell_counts = np.bincount(
        np.asarray([row["shell"] for row in selected_rows]), minlength=4
    )
    required = int(config["matched_contract"]["selection_cores_per_shell"])
    if not np.array_equal(shell_counts, np.full(4, required)):
        raise RuntimeError("selection panel is not balanced by shell")

    arrays_path = args.output_root / "ph006_observation_metadata.npz"
    temporary = arrays_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            core_id=np.asarray([row["core_id"] for row in rows], dtype=np.int64),
            shell=np.asarray([row["shell"] for row in rows], dtype=np.int8),
            cap=np.asarray([row["cap"] for row in rows], dtype=np.int8),
            median_radius_mpc=np.asarray(
                [row["median_radius_mpc"] for row in rows], dtype=np.float64
            ),
            angular_response=np.asarray(
                [row["median_angular_response"] for row in rows], dtype=np.float32
            ),
            boundary_distance_mpc=np.asarray(
                [row["median_boundary_distance_mpc"] for row in rows], dtype=np.float32
            ),
            tracer_density=np.asarray(
                [row["tracer_density_per_supported_voxel"] for row in rows],
                dtype=np.float32,
            ),
            supported_voxels=np.asarray(
                [row["supported_voxels"] for row in rows], dtype=np.int32
            ),
            authoritative_galaxies=np.asarray(
                [row["authoritative_galaxies"] for row in rows], dtype=np.int32
            ),
            selected_core_id=selected_core_id,
        )
    temporary.replace(arrays_path)

    adapter_manifest = (
        args.contract_root / "adapters/ph006/field/adapter_manifest.json"
    )
    adapter_payload = json.loads(adapter_manifest.read_text())
    marker = {
        "schema_version": "p12f-truth-free-selection-panel-v1",
        "created_utc": utc_now(),
        "git_revision": git_revision(),
        "phase": "ph006",
        "selected_core_id": selected_core_id.tolist(),
        "selected_core_metadata": selected_rows,
        "shell_counts": shell_counts.tolist(),
        "supported_core_count": int(len(rows)),
        "outside_science_shell_core_count": int(
            np.count_nonzero(np.asarray([row["shell"] for row in rows]) < 0)
        ),
        "outside_science_shell_policy": "retained in audit metadata; ineligible for the 128-core science panel",
        "selection_covariates": [
            "cap",
            "angular_response",
            "boundary_distance",
            "redshift_shell_from_median_supported_voxel_radius",
        ],
        "selection_uses_truth": False,
        "truth_files_read": [],
        "target_store_instantiated": False,
        "ph001_opened": False,
        "open_count": 0,
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "training_ready": str(
            (args.contract_root / "TRAINING_LOADER_READY.json").resolve()
        ),
        "training_ready_sha256": sha256(
            args.contract_root / "TRAINING_LOADER_READY.json"
        ),
        "adapter_manifest": str(adapter_manifest.resolve()),
        "adapter_manifest_sha256": sha256(adapter_manifest),
        "response_field_hashes": {
            cap: adapter_payload["caps"][cap]["field_sha256"]
            for cap in ("NGC", "SGC")
        },
        "observation_metadata": str(arrays_path.resolve()),
        "observation_metadata_sha256": sha256(arrays_path),
        "pass": True,
    }
    atomic_json(args.output_root / "P12F_PH006_PANEL_128.json", marker)
    loader.close()
    print(json.dumps(marker, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
