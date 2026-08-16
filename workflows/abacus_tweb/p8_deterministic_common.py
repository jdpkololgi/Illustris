#!/usr/bin/env python3
"""Shared contracts and metrics for deterministic P8 patch experiments.

This module deliberately contains no model code. It binds every candidate to
the same P4 authoritative rows, linear-increment targets, complete-fold metric,
and spatial-block uncertainty calculation.
"""
from __future__ import annotations

import hashlib
import json
import fcntl
import os
from pathlib import Path
import socket
import tempfile
from datetime import datetime, timezone

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    r2_score,
)


SHELL_NAMES = ("0p15_0p25", "0p25_0p35", "0p35_0p45", "0p45_0p55")
LAMBDA_NAMES = ("lambda1", "lambda2", "lambda3")
LAMBDA_THRESHOLD = 0.2


def acquire_run_lock(path: Path, *, purpose: str):
    """Acquire a non-blocking process-lifetime lock for one mutable run.

    The returned handle must remain live for the duration of the run.  Kernel
    ownership makes the lock safe across tmux sessions, allocations, and
    compute nodes sharing the same filesystem; a crashed process releases it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.seek(0)
        owner = handle.read().strip() or "unknown owner"
        handle.close()
        raise RuntimeError(f"run already has an active owner: {path}: {owner}") from error
    owner = {
        "schema_version": "p8-run-lock-v1",
        "purpose": str(purpose),
        "pid": int(os.getpid()),
        "host": socket.gethostname(),
        "acquired_utc": datetime.now(timezone.utc).isoformat(),
    }
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps(owner, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def sha256(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def linear_increments(eigenvalues: np.ndarray) -> np.ndarray:
    eig = np.asarray(eigenvalues, dtype=np.float64)
    if eig.ndim != 2 or eig.shape[1] != 3:
        raise ValueError(f"eigenvalues must have shape (N,3), got {eig.shape}")
    return np.column_stack((eig[:, 0], eig[:, 1] - eig[:, 0], eig[:, 2] - eig[:, 1]))


def increments_to_eigenvalues(increments: np.ndarray) -> np.ndarray:
    inc = np.asarray(increments, dtype=np.float64)
    if inc.ndim != 2 or inc.shape[1] != 3:
        raise ValueError(f"increments must have shape (N,3), got {inc.shape}")
    return np.column_stack((inc[:, 0], inc[:, 0] + inc[:, 1], inc.sum(axis=1)))


def fit_target_scaler(eigenvalues: np.ndarray) -> dict[str, list[float]]:
    inc = linear_increments(eigenvalues)
    mean = inc.mean(axis=0, dtype=np.float64)
    std = inc.std(axis=0, dtype=np.float64)
    if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(std)) or np.any(std <= 0):
        raise RuntimeError("invalid training-core target scaler")
    return {"mean": mean.tolist(), "std": std.tolist()}


def scale_increments(increments: np.ndarray, scaler: dict) -> np.ndarray:
    return (
        (np.asarray(increments, dtype=np.float64) - np.asarray(scaler["mean"]))
        / np.asarray(scaler["std"])
    ).astype(np.float32)


def unscale_increments(increments: np.ndarray, scaler: dict) -> np.ndarray:
    return (
        np.asarray(increments, dtype=np.float64) * np.asarray(scaler["std"])
        + np.asarray(scaler["mean"])
    )


def shell_weights(shell: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    shell = np.asarray(shell, dtype=np.int8)
    counts = np.bincount(shell, minlength=4).astype(np.int64)
    if np.any(counts[:4] == 0):
        raise RuntimeError(f"all four training shells are required, got {counts[:4]}")
    weights = 1.0 / np.sqrt(counts[shell])
    return weights.astype(np.float32), {
        SHELL_NAMES[index]: int(counts[index]) for index in range(4)
    }


def fold_roles(rotations: dict, rotation: int) -> tuple[tuple[int, ...], int, int]:
    row = rotations[str(int(rotation))]
    return (
        tuple(int(v) for v in row["train_folds"]),
        int(row["validation_fold"]),
        int(row["development_test_fold"]),
    )


def authoritative_mask(assignment) -> np.ndarray:
    return np.asarray(assignment["supervised_eligible"], dtype=bool)


def ordered_violation_rate(predicted_eigenvalues: np.ndarray) -> float:
    pred = np.asarray(predicted_eigenvalues)
    return float(np.mean((pred[:, 1] < pred[:, 0]) | (pred[:, 2] < pred[:, 1])))


def _safe_slope(truth: np.ndarray, prediction: np.ndarray) -> float:
    if len(truth) < 2 or float(np.var(prediction)) == 0.0:
        return float("nan")
    return float(np.polyfit(truth, prediction, 1)[0])


def _one_shell_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    result: dict = {}
    for index, name in enumerate(LAMBDA_NAMES):
        y = truth[:, index]
        p = prediction[:, index]
        result[name] = {
            "r2": float(r2_score(y, p)),
            "spearman": float(spearmanr(y, p).statistic),
            "mae": float(np.mean(np.abs(p - y))),
            "bias": float(np.mean(p - y)),
            "slope_prediction_on_truth": _safe_slope(y, p),
            "truth_variance": float(np.var(y)),
            "prediction_variance": float(np.var(p)),
            "prediction_to_truth_variance": float(np.var(p) / max(np.var(y), 1e-30)),
        }
    true_class = np.sum(truth > LAMBDA_THRESHOLD, axis=1)
    predicted_class = np.sum(prediction > LAMBDA_THRESHOLD, axis=1)
    recall = {}
    for label, name in ((0, "void"), (3, "knot")):
        selected = true_class == label
        recall[name] = (
            float(np.mean(predicted_class[selected] == label)) if np.any(selected) else None
        )
    result["classification"] = {
        "balanced_accuracy": float(balanced_accuracy_score(true_class, predicted_class)),
        "macro_f1": float(f1_score(true_class, predicted_class, average="macro", zero_division=0)),
        "confusion_matrix_true_rows": confusion_matrix(
            true_class, predicted_class, labels=(0, 1, 2, 3)
        ).tolist(),
        "void_recall": recall["void"],
        "knot_recall": recall["knot"],
    }
    result["n"] = int(len(truth))
    return result


def _macro_r2(truth: np.ndarray, prediction: np.ndarray, shell: np.ndarray) -> float:
    values = []
    for shell_id in range(4):
        selected = shell == shell_id
        if selected.sum() < 3 or float(np.var(truth[selected, 0])) <= 0:
            return float("nan")
        values.append(r2_score(truth[selected, 0], prediction[selected, 0]))
    return float(np.mean(values))


def spatial_block_interval(
    truth: np.ndarray,
    prediction: np.ndarray,
    shell: np.ndarray,
    superblock: np.ndarray,
    *,
    seed: int = 314159,
    draws: int = 400,
) -> dict:
    """Bootstrap super-blocks, never individual galaxies."""
    unique = np.unique(superblock)
    if len(unique) < 2:
        return {"draws": 0, "blocks": int(len(unique)), "p16": None, "p50": None, "p84": None}
    groups = {int(block): np.flatnonzero(superblock == block) for block in unique}
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(draws):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([groups[int(block)] for block in sampled])
        value = _macro_r2(truth[indices], prediction[indices], shell[indices])
        if np.isfinite(value):
            scores.append(value)
    if not scores:
        return {"draws": 0, "blocks": int(len(unique)), "p16": None, "p50": None, "p84": None}
    q = np.quantile(scores, (0.16, 0.5, 0.84))
    return {
        "draws": int(len(scores)),
        "blocks": int(len(unique)),
        "p16": float(q[0]),
        "p50": float(q[1]),
        "p84": float(q[2]),
    }


def _boundary_summary(
    truth: np.ndarray,
    prediction: np.ndarray,
    shell: np.ndarray,
    distance_mpc: np.ndarray,
) -> dict:
    residual = np.abs(prediction[:, 0] - truth[:, 0])
    finite = np.isfinite(distance_mpc) & np.isfinite(residual)
    rho = (
        float(spearmanr(distance_mpc[finite], residual[finite]).statistic)
        if finite.sum() > 2
        else None
    )
    result = {"abs_error_vs_distance_spearman": rho}
    for margin in (10.4, 20.8):
        selected = finite & (distance_mpc > margin)
        result[f"beyond_{margin:g}_mpc"] = {
            "n": int(selected.sum()),
            "macro_r2_lambda1": (
                _macro_r2(truth[selected], prediction[selected], shell[selected])
                if selected.sum() > 0
                else None
            ),
        }
    return result


def evaluate_complete_fold(
    *,
    parent_node_id: np.ndarray,
    predicted_eigenvalues: np.ndarray,
    truth_by_parent: np.ndarray,
    assignment,
    validation_fold: int,
    runtime: dict | None = None,
    strict_mask_by_parent: np.ndarray | None = None,
) -> dict:
    """Evaluate exactly one complete authoritative P4 validation fold."""
    parent = np.asarray(parent_node_id, dtype=np.int64)
    pred = np.asarray(predicted_eigenvalues, dtype=np.float64)
    if pred.shape != (len(parent), 3):
        raise ValueError("predictions must align with parent_node_id and have three columns")
    if len(np.unique(parent)) != len(parent):
        raise RuntimeError("duplicate parent predictions are forbidden")

    auth = authoritative_mask(assignment)
    required_rows = np.flatnonzero(auth & (np.asarray(assignment["fold"]) == validation_fold))
    required_parent = np.asarray(assignment["parent_node_id"][required_rows], dtype=np.int64)
    order = np.argsort(parent)
    sorted_parent = parent[order]
    lookup = np.searchsorted(sorted_parent, required_parent)
    if np.any(lookup == len(sorted_parent)) or not np.array_equal(
        sorted_parent[np.minimum(lookup, len(sorted_parent) - 1)], required_parent
    ):
        present = np.isin(required_parent, parent)
        raise RuntimeError(
            f"incomplete validation core coverage: {present.sum()}/{len(required_parent)}"
        )
    if len(parent) != len(required_parent) or not np.array_equal(
        np.sort(parent), np.sort(required_parent)
    ):
        raise RuntimeError("prediction artifact must contain exactly the complete validation fold")
    aligned_prediction = pred[order][lookup]
    truth = np.asarray(truth_by_parent[required_parent], dtype=np.float64)
    shell = np.asarray(assignment["shell"][required_rows], dtype=np.int8)
    superblock = np.asarray(assignment["superblock_id"][required_rows], dtype=np.int32)
    distance = np.asarray(
        assignment["distance_to_conservative_fold_boundary_mpc"][required_rows],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(aligned_prediction)):
        raise RuntimeError("non-finite truth or prediction in authoritative validation rows")

    per_shell = {}
    for shell_id, name in enumerate(SHELL_NAMES):
        selected = shell == shell_id
        if selected.sum() < 3:
            raise RuntimeError(f"validation shell {name} has only {selected.sum()} rows")
        per_shell[name] = _one_shell_metrics(truth[selected], aligned_prediction[selected])
    macro = float(np.mean([per_shell[name]["lambda1"]["r2"] for name in SHELL_NAMES]))
    # Mandatory diagnostic only: this prevents a learned model from looking
    # superior solely because a classical reconstruction collapses in shell 4.
    tracer_supported_macro = float(
        np.mean([per_shell[name]["lambda1"]["r2"] for name in SHELL_NAMES[:3]])
    )
    pooled = _one_shell_metrics(truth, aligned_prediction)
    strict = None
    if strict_mask_by_parent is not None:
        selected = np.asarray(strict_mask_by_parent[required_parent], dtype=bool)
        strict = {
            "n": int(selected.sum()),
            "retained_fraction": float(selected.mean()),
            "macro_r2_lambda1": (
                _macro_r2(truth[selected], aligned_prediction[selected], shell[selected])
                if selected.sum() > 0
                else None
            ),
        }
    return {
        "validation_fold": int(validation_fold),
        "n_authoritative": int(len(required_parent)),
        "complete_core_coverage": True,
        "primary_macro_r2_lambda1": macro,
        "diagnostic_first_three_shell_macro_r2_lambda1": tracer_supported_macro,
        "pooled": pooled,
        "per_shell": per_shell,
        "worst_shell_r2_lambda1": float(
            min(per_shell[name]["lambda1"]["r2"] for name in SHELL_NAMES)
        ),
        "ordering_violation_rate": ordered_violation_rate(aligned_prediction),
        "spatial_block_interval": spatial_block_interval(
            truth, aligned_prediction, shell, superblock
        ),
        "boundary": _boundary_summary(truth, aligned_prediction, shell, distance),
        "strict_hop_diagnostic": strict,
        "runtime": {} if runtime is None else runtime,
    }


def evaluate_complete_phase(
    *,
    parent_node_id: np.ndarray,
    predicted_eigenvalues: np.ndarray,
    truth_by_parent: np.ndarray,
    assignment,
    phase: str,
    runtime: dict | None = None,
) -> dict:
    """Evaluate exactly all authoritative rows in one independent phase.

    P10 uses an entire Abacus phase for model selection rather than one internal
    spatial fold.  This contract deliberately rejects subsets and duplicates so
    an apparently favourable partial ph006 score cannot enter model selection.
    """
    parent = np.asarray(parent_node_id, dtype=np.int64)
    prediction = np.asarray(predicted_eigenvalues, dtype=np.float64)
    if prediction.shape != (len(parent), 3):
        raise ValueError("predictions must align with parent_node_id and have three columns")
    if len(np.unique(parent)) != len(parent):
        raise RuntimeError("duplicate parent predictions are forbidden")

    auth = authoritative_mask(assignment)
    required_rows = np.flatnonzero(auth)
    required_parent = np.asarray(
        assignment["parent_node_id"][required_rows], dtype=np.int64
    )
    order = np.argsort(parent)
    sorted_parent = parent[order]
    lookup = np.searchsorted(sorted_parent, required_parent)
    if np.any(lookup == len(sorted_parent)) or not np.array_equal(
        sorted_parent[np.minimum(lookup, len(sorted_parent) - 1)], required_parent
    ):
        present = np.isin(required_parent, parent)
        raise RuntimeError(
            f"incomplete {phase} authoritative coverage: "
            f"{present.sum()}/{len(required_parent)}"
        )
    if len(parent) != len(required_parent) or not np.array_equal(
        np.sort(parent), np.sort(required_parent)
    ):
        raise RuntimeError(
            f"prediction artifact must contain exactly complete {phase} authoritative rows"
        )

    aligned_prediction = prediction[order][lookup]
    truth = np.asarray(truth_by_parent[required_parent], dtype=np.float64)
    shell = np.asarray(assignment["shell"][required_rows], dtype=np.int8)
    superblock = np.asarray(
        assignment["superblock_id"][required_rows], dtype=np.int32
    )
    distance = np.asarray(
        assignment["distance_to_conservative_fold_boundary_mpc"][required_rows],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(aligned_prediction)):
        raise RuntimeError("non-finite truth or prediction in authoritative phase rows")

    per_shell = {}
    for shell_id, name in enumerate(SHELL_NAMES):
        selected = shell == shell_id
        if selected.sum() < 3:
            raise RuntimeError(f"validation shell {name} has only {selected.sum()} rows")
        per_shell[name] = _one_shell_metrics(
            truth[selected], aligned_prediction[selected]
        )
    macro = float(
        np.mean([per_shell[name]["lambda1"]["r2"] for name in SHELL_NAMES])
    )
    tracer_supported_macro = float(
        np.mean([per_shell[name]["lambda1"]["r2"] for name in SHELL_NAMES[:3]])
    )
    return {
        "validation_phase": str(phase),
        "n_authoritative": int(len(required_parent)),
        "complete_phase_coverage": True,
        "primary_macro_r2_lambda1": macro,
        "diagnostic_first_three_shell_macro_r2_lambda1": tracer_supported_macro,
        "pooled": _one_shell_metrics(truth, aligned_prediction),
        "per_shell": per_shell,
        "worst_shell_r2_lambda1": float(
            min(per_shell[name]["lambda1"]["r2"] for name in SHELL_NAMES)
        ),
        "ordering_violation_rate": ordered_violation_rate(aligned_prediction),
        "spatial_block_interval": spatial_block_interval(
            truth, aligned_prediction, shell, superblock
        ),
        "boundary": _boundary_summary(truth, aligned_prediction, shell, distance),
        "runtime": {} if runtime is None else runtime,
    }


def fit_affine_on_training(
    raw_prediction: np.ndarray,
    truth: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Fit three scalar affine maps on training rows only."""
    calibrated = np.empty_like(raw_prediction, dtype=np.float64)
    coefficients = []
    for index in range(3):
        selected = np.asarray(train_mask, dtype=bool) & np.isfinite(raw_prediction[:, index])
        design = np.column_stack((raw_prediction[selected, index], np.ones(selected.sum())))
        coef, *_ = np.linalg.lstsq(design, truth[selected, index], rcond=None)
        calibrated[:, index] = coef[0] * raw_prediction[:, index] + coef[1]
        coefficients.append({"slope": float(coef[0]), "intercept": float(coef[1])})
    return calibrated, {"fit_split": "training cores", "coefficients": coefficients}


def atomic_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
