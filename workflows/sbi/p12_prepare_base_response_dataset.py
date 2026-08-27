#!/usr/bin/env python3
"""Materialize the coordinate-aligned P12-A base-prediction dataset.

P12-A deliberately excludes the independently trained 32-d fold latents: their
coordinate systems are not guaranteed to align.  It conditions on the physical
three-eigenvalue U-PATCH OOF prediction plus deployable response variables.  P4
fold/superblock identifiers are serialized only for spatially disjoint ph006
calibration and evaluation; they are never model features.

The sealed ph001 phase is refused unconditionally.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p8_deterministic_common import sha256
from workflows.abacus_tweb.p10_training_contract import atomic_json
from workflows.sbi.p12_export_unet_summaries import parent_to_assignment_index


TRAIN_PHASES = ("ph000", "ph002", "ph003", "ph004", "ph005")
VALIDATION_PHASE = "ph006"
SEALED_PHASE = "ph001"
ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase")
SUMMARY_ROOT = ROOT / "p12_oof_summaries"
CONTRACT_ROOT = ROOT / "p12_crossfit_contracts"
FULL_CONTRACT = ROOT / "training_contract"
OUTPUT = ROOT / "p12a_base_response_v1"
FEATURE_NAMES = (
    "base_lambda1",
    "base_lambda2",
    "base_lambda3",
    "redshift",
    "log_ntilde_mpc3",
    "cap_ngc",
    "log1p_field_support_distance_mpc",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def softplus_coordinates(eigenvalues: np.ndarray, epsilon: float = 1.0e-7) -> np.ndarray:
    """Map ordered eigenvalues to the canonical real softplus coordinates."""
    values = np.asarray(eigenvalues, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("eigenvalues must have shape (N, 3)")
    if not np.all(np.isfinite(values)):
        raise ValueError("eigenvalues contain non-finite values")
    gaps = np.maximum(np.diff(values, axis=1), epsilon)
    transformed = np.where(gaps > 20.0, gaps, np.log(np.expm1(gaps)))
    result = np.column_stack((values[:, 0], transformed)).astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("softplus-coordinate transform produced non-finite values")
    return result


def stratified_indices(
    shell: np.ndarray, maximum: int, seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, float | int]]]:
    """Deterministically sample shells with sqrt-count allocation.

    The returned weights undo the sampling for natural-population evaluation.
    Reweighting the training x-distribution is admissible because redshift (and
    hence shell) is included in the conditioning vector.
    """
    shell = np.asarray(shell, dtype=np.int8)
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    unique = np.unique(shell)
    if not np.array_equal(unique, np.asarray([0, 1, 2, 3], dtype=np.int8)):
        raise RuntimeError(f"expected four shells, found {unique.tolist()}")
    counts = np.asarray([np.count_nonzero(shell == value) for value in unique])
    target = np.sqrt(counts.astype(np.float64))
    target = np.floor(min(maximum, len(shell)) * target / target.sum()).astype(int)
    target = np.minimum(target, counts)
    remaining = min(maximum, len(shell)) - int(target.sum())
    while remaining > 0:
        available = counts - target
        chosen = int(np.argmax(available))
        if available[chosen] <= 0:
            break
        add = min(remaining, int(available[chosen]))
        target[chosen] += add
        remaining -= add
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    audit: dict[str, dict[str, float | int]] = {}
    for value, count, take in zip(unique, counts, target, strict=True):
        candidates = np.flatnonzero(shell == value)
        selected = rng.choice(candidates, size=int(take), replace=False)
        parts.append(selected)
        weight = float(count / take)
        weights.append(np.full(int(take), weight, dtype=np.float32))
        audit[str(int(value))] = {
            "available": int(count),
            "selected": int(take),
            "natural_weight": weight,
        }
    index = np.concatenate(parts)
    weight = np.concatenate(weights)
    order = rng.permutation(len(index))
    return index[order], weight[order], audit


def phase_contract_root(phase: str) -> Path:
    if phase == VALIDATION_PHASE:
        return FULL_CONTRACT
    return CONTRACT_ROOT / f"omit_{phase}"


def load_phase_sample(
    phase: str, maximum: int, seed: int
) -> tuple[dict[str, np.ndarray], dict]:
    if phase == SEALED_PHASE:
        raise PermissionError("ph001 is sealed")
    marker_path = SUMMARY_ROOT / phase / "OOF_SUMMARY_COMPLETE.json"
    if not marker_path.exists():
        raise FileNotFoundError(f"missing OOF marker for {phase}: {marker_path}")
    marker = json.loads(marker_path.read_text())
    if not marker.get("pass") or marker.get("sealed_phase_opened"):
        raise RuntimeError(f"invalid OOF marker for {phase}")
    if phase in TRAIN_PHASES and phase in marker.get("training_phases", ()):
        raise RuntimeError(f"{phase} summary is not out of fold")
    if phase == VALIDATION_PHASE and tuple(marker.get("training_phases", ())) != TRAIN_PHASES:
        raise RuntimeError("ph006 summary is not from the frozen all-five-phase encoder")

    parent = np.load(marker["arrays"]["parent_node_id"], mmap_mode="r")
    base = np.load(marker["arrays"]["base_prediction"], mmap_mode="r")
    truth = np.load(marker["arrays"]["truth"], mmap_mode="r")
    response = np.load(marker["arrays"]["response"])
    if not (len(parent) == len(base) == len(truth) == len(response["shell"])):
        raise RuntimeError(f"{phase} OOF arrays disagree in length")
    selected, natural_weight, sample_audit = stratified_indices(
        response["shell"], maximum, seed
    )

    contract = json.loads(
        (phase_contract_root(phase) / "phases" / phase / "phase_contract.json").read_text()
    )
    assignment = np.load(contract["inputs"]["assignment"], mmap_mode="r")
    target_length = len(np.load(
        phase_contract_root(phase) / "phases" / phase / "parent_eigenvalues.npy",
        mmap_mode="r",
    ))
    parent_lookup = parent_to_assignment_index(assignment, target_length)
    chosen_parent = np.asarray(parent[selected], dtype=np.int64)
    row = parent_lookup[chosen_parent]
    if np.any(row < 0):
        raise RuntimeError(f"{phase} sampled parent lacks assignment row")
    shell = np.asarray(assignment["shell"][row], dtype=np.int8)
    cap = np.asarray(assignment["cap"][row], dtype=np.uint8)
    if not np.array_equal(shell, np.asarray(response["shell"][selected], dtype=np.int8)):
        raise RuntimeError(f"{phase} response/assignment shell mismatch")
    if not np.array_equal(cap, np.asarray(response["cap"][selected], dtype=np.uint8)):
        raise RuntimeError(f"{phase} response/assignment cap mismatch")
    field_distance = np.asarray(assignment["field_support_distance_mpc"][row], dtype=np.float32)
    if not np.all(np.isfinite(field_distance)) or np.any(field_distance < 0):
        raise RuntimeError(f"{phase} invalid field-support distance")
    base_selected = np.asarray(base[selected], dtype=np.float32)
    truth_selected = np.asarray(truth[selected], dtype=np.float32)
    redshift = np.asarray(response["redshift"][selected], dtype=np.float32)
    ntilde = np.asarray(response["ntilde_mpc3"][selected], dtype=np.float32)
    context = np.column_stack(
        (
            base_selected,
            redshift,
            np.log(np.maximum(ntilde, 1.0e-12)),
            cap.astype(np.float32),
            np.log1p(field_distance),
        )
    ).astype(np.float32)
    if context.shape[1] != len(FEATURE_NAMES) or not np.all(np.isfinite(context)):
        raise RuntimeError(f"{phase} invalid P12-A context")
    if not np.all(np.diff(truth_selected, axis=1) >= -1.0e-6):
        raise RuntimeError(f"{phase} truth is not ordered")
    arrays = {
        "context": context,
        "theta_softplus": softplus_coordinates(truth_selected),
        "truth_eigenvalues": truth_selected,
        "base_prediction_eigenvalues": base_selected,
        "parent_node_id": chosen_parent,
        "shell": shell,
        "cap": cap,
        "fold": np.asarray(assignment["fold"][row], dtype=np.uint8),
        "superblock_id": np.asarray(assignment["superblock_id"][row], dtype=np.int32),
        "natural_weight": natural_weight,
    }
    assignment.close()
    response.close()
    audit = {
        "phase": phase,
        "rows_available": int(marker["rows"]),
        "rows_selected": int(len(selected)),
        "summary_marker": str(marker_path),
        "summary_marker_sha256": sha256(marker_path),
        "sample": sample_audit,
        "feature_names": list(FEATURE_NAMES),
        "fold_is_feature": False,
        "superblock_is_feature": False,
    }
    return arrays, audit


def concatenate(rows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = tuple(rows[0])
    if any(tuple(row) != keys for row in rows[1:]):
        raise RuntimeError("phase samples do not share a schema")
    return {key: np.concatenate([row[key] for row in rows], axis=0) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--train-rows", type=int, default=2_000_000)
    parser.add_argument("--validation-rows", type=int, default=600_000)
    parser.add_argument("--seed", type=int, default=12042)
    args = parser.parse_args()
    output = args.output_root
    ready = output / "P12A_DATASET_READY.json"
    if ready.exists():
        existing = json.loads(ready.read_text())
        if existing.get("pass") and not existing.get("sealed_phase_opened"):
            print(json.dumps(existing, indent=2), flush=True)
            return
    per_phase = max(1, args.train_rows // len(TRAIN_PHASES))
    train_rows, audits = [], {}
    for offset, phase in enumerate(TRAIN_PHASES):
        row, audit = load_phase_sample(phase, per_phase, args.seed + offset)
        row["phase_index"] = np.full(len(row["shell"]), offset, dtype=np.uint8)
        train_rows.append(row)
        audits[phase] = audit
    training = concatenate(train_rows)
    validation, audits[VALIDATION_PHASE] = load_phase_sample(
        VALIDATION_PHASE, args.validation_rows, args.seed + 100
    )
    validation["phase_index"] = np.full(
        len(validation["shell"]), len(TRAIN_PHASES), dtype=np.uint8
    )
    output.mkdir(parents=True, exist_ok=True)
    train_path = output / "training_oof_sample.npz"
    validation_path = output / "ph006_selection_sample.npz"
    np.savez(train_path, **training)
    np.savez(validation_path, **validation)
    report = {
        "schema_version": "p12a-base-response-dataset-v1",
        "created_utc": utc_now(),
        "conditioning_contract": "three physical OOF base eigenvalue predictions plus deployable response",
        "feature_names": list(FEATURE_NAMES),
        "target_parameterization": "ordered softplus increments",
        "training_phases": list(TRAIN_PHASES),
        "validation_phase": VALIDATION_PHASE,
        "sealed_phase": SEALED_PHASE,
        "sealed_phase_opened": False,
        "phase_is_feature": False,
        "fold_is_feature": False,
        "superblock_is_feature": False,
        "artificial_fold_boundary_distance_is_feature": False,
        "training": {
            "path": str(train_path),
            "sha256": sha256(train_path),
            "rows": int(len(training["shell"])),
        },
        "validation": {
            "path": str(validation_path),
            "sha256": sha256(validation_path),
            "rows": int(len(validation["shell"])),
            "calibration_folds": [0, 1],
            "evaluation_folds": [2, 3, 4],
        },
        "phase_audit": audits,
        "pass": bool(
            len(training["shell"]) > 0
            and len(validation["shell"]) > 0
            and not np.any(training["phase_index"] == len(TRAIN_PHASES))
            and np.all(np.isfinite(training["context"]))
            and np.all(np.isfinite(validation["context"]))
        ),
    }
    atomic_json(ready, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
