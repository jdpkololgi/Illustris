#!/usr/bin/env python3
"""Prepare five leakage-safe leave-one-phase-out U-PATCH contracts for P12."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p10_prepare_training_contract import (
    BIN_WIDTH,
    CAP_NAME,
    CONTRAST_EPSILON,
    CURVE_STEP,
    FIT_Z_MAX,
    FIT_Z_MIN,
    KNOT_SPACING,
    MINIMUM_EXPOSURE,
    ROOT,
    Z_MAX,
    Z_MIN,
    phase_files,
    p1_manifest,
)
from workflows.abacus_tweb.p10_training_contract import (
    BLIND_PHASE,
    TRAINING_PHASES,
    P10PhaseBalancedLoader,
    atomic_json,
    epoch_hash,
    sha256,
)
from workflows.abacus_tweb.p6_refit_fullcap_selection import (
    build_cap_lookup,
    fit_log_spline,
    histogram_counts,
    histogram_effective_volume,
    radius_to_redshift_grid,
)


FULL_CONTRACT = ROOT / "training_contract"
OUTPUT = ROOT / "p12_crossfit_contracts"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hardlink_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def link_directory(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    target.symlink_to(source, target_is_directory=True)


def selection_fit(
    *, root: Path, output: Path, train: tuple[str, ...], omitted: str
) -> dict:
    edges = np.arange(Z_MIN, Z_MAX + 0.5 * BIN_WIDTH, BIN_WIDTH)
    centers = 0.5 * (edges[:-1] + edges[1:])
    grid_z = np.arange(Z_MIN, Z_MAX + 0.5 * CURVE_STEP, CURVE_STEP)
    radius_grid, redshift_grid = radius_to_redshift_grid(Z_MIN, Z_MAX)
    counts_total = np.zeros((2, len(centers)), dtype=np.int64)
    volume_total = np.zeros((2, len(centers)), dtype=np.float64)
    sources = {}
    for phase in train:
        files = phase_files(root, phase)
        p3 = json.loads(files["p3_manifest"].read_text())
        cores = np.load(files["cores"], mmap_mode="r")
        widths = np.asarray(cores["upper_mpc"] - cores["lower_mpc"], dtype=np.float64)
        core_mpc = float(np.median(widths))
        lookups = {
            name: build_cap_lookup(cores, cap, core_mpc)
            for cap, name in CAP_NAME.items()
        }
        counts, count_audit = histogram_counts(
            parent_path=Path(p1_manifest(root, phase)["parent"]),
            context_path=files["p4_root"] / "context_assignment.npz",
            edges=edges,
        )
        volume, volume_audit = histogram_effective_volume(
            p3=p3,
            lookups=lookups,
            core_mpc=core_mpc,
            edges=edges,
            radius_grid_mpc=radius_grid,
            redshift_grid=redshift_grid,
        )
        counts_total += counts.sum(axis=1)
        volume_total += volume.sum(axis=1)
        sources[phase] = {"counts": count_audit, "volume": volume_audit}
    caps = {}
    for cap, name in CAP_NAME.items():
        curve, fit = fit_log_spline(
            centers,
            counts_total[cap],
            volume_total[cap],
            grid_z,
            knot_spacing=KNOT_SPACING,
            fit_z_min=FIT_Z_MIN,
            fit_z_max=FIT_Z_MAX,
        )
        caps[name] = {"grid_z": grid_z.tolist(), "ntilde": curve.tolist(), "fit": fit}
    selection = {
        "schema_version": "p12-crossfit-selection-v1",
        "fit_phases": list(train),
        "application_phases": [omitted],
        "out_of_fold_phase": omitted,
        "rotations": {"0": {"caps": caps}},
        "cosmology": {
            "name": "Planck18",
            "radius_grid_mpc": radius_grid.tolist(),
            "redshift_grid": redshift_grid.tolist(),
        },
        "contrast": {"epsilon": CONTRAST_EPSILON, "minimum_exposure": MINIMUM_EXPOSURE},
        "sources": sources,
        "phase_is_model_input": False,
        "pass": bool(
            omitted not in train
            and all(np.all(np.asarray(row["ntilde"]) > 0) for row in caps.values())
        ),
    }
    path = output / "transforms/field/selection_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, selection)
    if not selection["pass"]:
        raise RuntimeError(f"{omitted} crossfit selection failed")
    return selection


def leave_one_out_field_transform(
    *, output: Path, train: tuple[str, ...], selection: dict
) -> dict:
    full = json.loads((FULL_CONTRACT / "transforms/field/field_transform.json").read_text())
    normalization = {"channels": {}}
    for channel in ("counts", "expected_counts", "log_count_ratio", "ntilde_mpc3"):
        means = [full["per_phase_diagnostics"][p][channel]["mean"] for p in train]
        seconds = [full["per_phase_diagnostics"][p][channel]["second_moment"] for p in train]
        mean = float(np.mean(means))
        second = float(np.mean(seconds))
        normalization["channels"][channel] = {
            "policy": "zscore",
            "mean": mean,
            "std": float(np.sqrt(max(second - mean * mean, 0.0))),
        }
    for channel in ("exposure_apodized", "exposure_binary", "los_x", "los_y", "los_z"):
        normalization["channels"][channel] = {"policy": "identity"}
    result = {
        "schema_version": "p12-crossfit-field-transform-v1",
        "fit_phases": list(train),
        "selection_manifest_sha256": sha256(output / "transforms/field/selection_manifest.json"),
        "normalization": normalization,
        "note": "counts moments are exact leave-one-phase-out combinations; unused normalized derived-channel moments inherit per-phase diagnostics from the full fit",
        "pass": bool(all(row.get("std", 1.0) > 0 for row in normalization["channels"].values())),
    }
    atomic_json(output / "transforms/field/field_transform.json", result)
    return result


def leave_one_out_target_scaler(*, output: Path, train: tuple[str, ...]) -> dict:
    full = json.loads((FULL_CONTRACT / "transforms/target_scaler.json").read_text())
    means, seconds = [], []
    diagnostics = {}
    for phase in train:
        row = full["per_phase_diagnostics"][phase]
        mean = np.asarray(row["mean"], dtype=np.float64)
        std = np.asarray(row["std"], dtype=np.float64)
        means.append(mean)
        seconds.append(std * std + mean * mean)
        diagnostics[phase] = row
    mean = np.mean(means, axis=0)
    second = np.mean(seconds, axis=0)
    std = np.sqrt(np.maximum(second - mean * mean, 0.0))
    result = {
        "schema_version": "p12-crossfit-target-scaler-v1",
        "representation": "linear increments",
        "definition": ["lambda1", "lambda2-lambda1", "lambda3-lambda2"],
        "fit_phases": list(train),
        "fit_policy": "equal phase mixture of exact authoritative-row moments",
        "mean": mean.tolist(),
        "std": std.tolist(),
        "per_phase_diagnostics": diagnostics,
        "phase_not_feature": True,
        "pass": bool(np.all(np.isfinite(std)) and np.all(std > 0)),
    }
    atomic_json(output / "transforms/target_scaler.json", result)
    return result


def materialize_phase_contract(
    *, output: Path, phase: str, role: str
) -> dict:
    source = FULL_CONTRACT / "phases" / phase
    target = output / "phases" / phase
    target.mkdir(parents=True, exist_ok=True)
    record = json.loads((source / "phase_contract.json").read_text())
    record["role"] = role
    record["p12_crossfit_role_override"] = True
    atomic_json(target / "phase_contract.json", record)
    for name in ("parent_targetid.npy", "parent_redshift.npy", "parent_eigenvalues.npy", "active_row_weight.npy"):
        hardlink_or_copy(source / name, target / name)
    if role == "training":
        for name in ("training_core_id.npy", "training_core_weight.npy"):
            hardlink_or_copy(source / name, target / name)
    else:
        hardlink_or_copy(source / "training_core_id.npy", target / "validation_core_id.npy")
    link_directory(FULL_CONTRACT / "adapters" / phase / "field", output / "adapters" / phase / "field")
    return record


def build_contract(*, root: Path, omitted: str) -> dict:
    train = tuple(phase for phase in TRAINING_PHASES if phase != omitted)
    output = root / f"omit_{omitted}"
    output.mkdir(parents=True, exist_ok=True)
    phase_records = {
        phase: materialize_phase_contract(
            output=output,
            phase=phase,
            role="validation_and_selection" if phase == omitted else "training",
        )
        for phase in train + (omitted,)
    }
    selection = selection_fit(root=ROOT, output=output, train=train, omitted=omitted)
    field = leave_one_out_field_transform(output=output, train=train, selection=selection)
    target = leave_one_out_target_scaler(output=output, train=train)
    loader_manifest = {
        "schema_version": "p12-crossfit-training-loader-v1",
        "created_utc": utc_now(),
        "roles": {
            "training": list(train),
            "validation_and_selection": omitted,
            "sealed_blind_test": BLIND_PHASE,
        },
        "epoch": {"policy": "complete phase-balanced without-replacement cores"},
        "objective": {"policy": "equal phase, within-phase sqrt-shell row MSE"},
        "out_of_fold_phase": omitted,
        "phase_is_model_input": False,
        "sealed_phase_opened": False,
        "pass": bool(
            omitted not in train
            and field["pass"]
            and target["pass"]
            and all(row["pass"] for row in phase_records.values())
        ),
    }
    atomic_json(output / "TRAINING_LOADER_READY.json", loader_manifest)
    loader = P10PhaseBalancedLoader(output, include_blind=False)
    refs = loader.training_epoch(seed=42, epoch=1)
    validation = loader.validation_refs()
    canary = {
        "training_cores": len(refs),
        "training_epoch_sha256": epoch_hash(refs),
        "validation_cores": len(validation),
        "validation_core_unique": len({row.core_id for row in validation}) == len(validation),
        "out_of_fold_phase_absent_from_training": omitted not in {row.phase for row in refs},
    }
    loader_manifest["canary"] = canary
    loader_manifest["pass"] = bool(loader_manifest["pass"] and all(
        value for key, value in canary.items() if isinstance(value, bool)
    ))
    atomic_json(output / "TRAINING_LOADER_READY.json", loader_manifest)
    return {"root": str(output), "marker_sha256": sha256(output / "TRAINING_LOADER_READY.json"), **loader_manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--omitted-phases", nargs="+", choices=TRAINING_PHASES, default=list(TRAINING_PHASES))
    args = parser.parse_args()
    ready = args.output_root / "P12_CROSSFIT_CONTRACTS_READY.json"
    if ready.exists():
        existing = json.loads(ready.read_text())
        if (
            existing.get("pass") is True
            and existing.get("sealed_phase_opened") is False
            and existing.get("omitted_phases") == list(args.omitted_phases)
        ):
            print(json.dumps(existing, indent=2), flush=True)
            return
    contracts = {
        phase: build_contract(root=args.output_root, omitted=phase)
        for phase in args.omitted_phases
    }
    report = {
        "schema_version": "p12-crossfit-contract-set-v1",
        "created_utc": utc_now(),
        "omitted_phases": list(args.omitted_phases),
        "sealed_phase": BLIND_PHASE,
        "sealed_phase_opened": False,
        "contracts": contracts,
        "pass": bool(all(row["pass"] for row in contracts.values())),
    }
    atomic_json(args.output_root / "P12_CROSSFIT_CONTRACTS_READY.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
