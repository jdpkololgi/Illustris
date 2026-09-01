#!/usr/bin/env python3
"""Audit paired-view P11 JEPA latent snapshots without opening ph001.

The audit deliberately separates three questions which are easy to conflate:

1. Do dense and degraded views retain a shared, predictable latent subspace?
2. Is apparent agreement caused by collapse, a response shortcut, or forced equality?
3. Does the deployable degraded-view representation retain target information?

Linear CKA and cross-fitted CCA describe shared relational structure and are not
coordinate-wise equality tests.  Cross-view retrieval and a cross-fitted orthogonal
Procrustes map test whether individual paired cores remain identifiable.  Effective
rank, per-axis variance, paired-versus-shuffled controls, response strata, and held-out
linear probes guard against misleading alignment scores.

This module never interprets latent agreement as posterior calibration.  If a JEPA
encoder is promoted, its posterior head must be refit and independently calibrated by
P12.  The only supported selection phase is ph006; ph001 is rejected explicitly.

Snapshot NPZ contract
---------------------
Required arrays:
  sample_id[N]             stable parent/core identity, unique within the snapshot
  dense_latent[N,Dt]       teacher representation of the dense view
  degraded_latent[N,Ds]    deployable student representation of the degraded view
  response_strength[N]     deployable response/exposure summary
  probe_split[N]           0 = spatially disjoint probe fit, 1 = probe evaluation
  metadata_json            scalar JSON with run_id, epoch, global_step, phase and
                           sealed_phase_opened=false

Optional arrays:
  predicted_dense_latent[N,Dt]  JEPA predictor output P(z_student)
  target[N,K]                    physical held-out targets for linear probes
  response_features[N,R]         response-only control features
  response_only_latent[N,Ds]     student encoding with signal fields zeroed
  sample_weight[N]               natural-volume/scientific weights
  group_id[N]                    retained for provenance/future block bootstrap
  core_id[N]                     canonical P4 core containing each exported galaxy
  fold_id[N]                     frozen spatial fold (0--1 probe fit, 2--4 evaluation)

The same sample_id ordering and run_id are required across a checkpoint series.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


SCHEMA_VERSION = "p11-jepa-latent-diagnostics-v1"
GATE_VERSION = "p11-jepa-latent-gate-v1"
ALLOWED_PHASES = {"ph002", "ph003", "ph004", "ph005", "ph006"}
SEALED_PHASE = "ph001"
MATCHED_ARMS = {"supervised_masked", "masked_reconstruction", "response_only", "jepa"}
REGISTERED_TRAJECTORY_STEPS = (0, 250, 500)

DEFAULT_THRESHOLDS = {
    "minimum_effective_rank_fraction": 0.25,
    "minimum_total_standard_deviation": 1.0e-4,
    "minimum_cka_over_shuffle": 0.05,
    "minimum_retrieval_mrr_over_shuffle": 0.02,
    "high_alignment": 0.95,
    "material_probe_gap": 0.03,
    "maximum_probe_regression": 0.01,
    "material_alignment_gain": 0.05,
    "material_rank_fraction_loss": 0.10,
    "response_shortcut_margin": 0.01,
    "low_response_error_ratio": 1.5,
    "deceptively_flat_alignment_ratio": 0.9,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def atomic_json(path: Path, payload: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def save_latent_snapshot(
    path: Path,
    *,
    metadata: Mapping,
    sample_id: np.ndarray,
    dense_latent: np.ndarray,
    degraded_latent: np.ndarray,
    response_strength: np.ndarray,
    probe_split: np.ndarray,
    predicted_dense_latent: np.ndarray | None = None,
    target: np.ndarray | None = None,
    response_features: np.ndarray | None = None,
    response_only_latent: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
    group_id: np.ndarray | None = None,
    core_id: np.ndarray | None = None,
    fold_id: np.ndarray | None = None,
) -> None:
    """Atomically write a no-pickle snapshot that can be audited during training."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata_json": np.asarray(json.dumps(dict(metadata), sort_keys=True)),
        "sample_id": np.asarray(sample_id),
        "dense_latent": np.asarray(dense_latent, dtype=np.float32),
        "degraded_latent": np.asarray(degraded_latent, dtype=np.float32),
        "response_strength": np.asarray(response_strength, dtype=np.float32),
        "probe_split": np.asarray(probe_split, dtype=np.int8),
    }
    optional = {
        "predicted_dense_latent": predicted_dense_latent,
        "target": target,
        "response_features": response_features,
        "response_only_latent": response_only_latent,
        "sample_weight": sample_weight,
        "group_id": group_id,
        "core_id": core_id,
        "fold_id": fold_id,
    }
    for name, values in optional.items():
        if values is not None:
            payload[name] = np.asarray(values)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


@dataclass(frozen=True)
class LatentSnapshot:
    path: Path
    metadata: dict
    sample_id: np.ndarray
    dense_latent: np.ndarray
    degraded_latent: np.ndarray
    predicted_dense_latent: np.ndarray | None
    response_strength: np.ndarray
    response_features: np.ndarray
    response_only_latent: np.ndarray | None
    probe_split: np.ndarray
    target: np.ndarray | None
    sample_weight: np.ndarray
    group_id: np.ndarray | None
    core_id: np.ndarray | None
    fold_id: np.ndarray | None


def _two_dimensional(name: str, values: np.ndarray, rows: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != rows or values.shape[1] < 1:
        raise ValueError(f"{name} must have shape [N,D] with D>=1")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    return values


def load_latent_snapshot(path: Path) -> LatentSnapshot:
    path = Path(path)
    if SEALED_PHASE in str(path):
        raise ValueError("ph001 is sealed and cannot enter P11 diagnostics")
    with np.load(path, allow_pickle=False) as data:
        required = {
            "metadata_json",
            "sample_id",
            "dense_latent",
            "degraded_latent",
            "response_strength",
            "probe_split",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"snapshot is missing required arrays: {missing}")
        raw_metadata = np.asarray(data["metadata_json"])
        if raw_metadata.ndim != 0:
            raise ValueError("metadata_json must be a scalar JSON string")
        metadata = json.loads(str(raw_metadata.item()))
        for key in ("run_id", "arm", "epoch", "global_step", "phase"):
            if key not in metadata:
                raise ValueError(f"metadata_json is missing {key}")
        if metadata["arm"] not in MATCHED_ARMS:
            raise ValueError(f"unsupported matched arm: {metadata['arm']}")
        phase = str(metadata["phase"])
        if phase == SEALED_PHASE or metadata.get("sealed_phase_opened", False):
            raise ValueError("ph001 is sealed and cannot enter P11 diagnostics")
        if phase not in ALLOWED_PHASES:
            raise ValueError(f"unsupported P11 phase: {phase}")
        for source in metadata.get("source_paths", []):
            if SEALED_PHASE in str(source):
                raise ValueError("source provenance contains the sealed ph001 phase")

        sample_id = np.asarray(data["sample_id"])
        if sample_id.ndim != 1 or sample_id.size < 8:
            raise ValueError("sample_id must be a one-dimensional array with >=8 rows")
        if len(np.unique(sample_id)) != len(sample_id):
            raise ValueError("sample_id must be unique")
        rows = len(sample_id)
        dense = _two_dimensional("dense_latent", data["dense_latent"], rows)
        degraded = _two_dimensional("degraded_latent", data["degraded_latent"], rows)
        predicted = None
        if "predicted_dense_latent" in data.files:
            predicted = _two_dimensional(
                "predicted_dense_latent", data["predicted_dense_latent"], rows
            )
            if predicted.shape[1] != dense.shape[1]:
                raise ValueError("predicted_dense_latent must use the teacher dimension")
        response = np.asarray(data["response_strength"], dtype=np.float64)
        if response.shape != (rows,) or not np.all(np.isfinite(response)):
            raise ValueError("response_strength must be finite with shape [N]")
        probe_split = np.asarray(data["probe_split"], dtype=np.int8)
        if probe_split.shape != (rows,) or set(np.unique(probe_split)) != {0, 1}:
            raise ValueError("probe_split must contain spatially disjoint 0/1 rows")
        weights = (
            np.asarray(data["sample_weight"], dtype=np.float64)
            if "sample_weight" in data.files
            else np.ones(rows, dtype=np.float64)
        )
        if weights.shape != (rows,) or np.any(weights <= 0) or not np.all(np.isfinite(weights)):
            raise ValueError("sample_weight must be positive and finite with shape [N]")
        response_features = (
            _two_dimensional("response_features", data["response_features"], rows)
            if "response_features" in data.files
            else response[:, None]
        )
        response_only_latent = None
        if "response_only_latent" in data.files:
            response_only_latent = _two_dimensional(
                "response_only_latent", data["response_only_latent"], rows
            )
        target = None
        if "target" in data.files:
            target = _two_dimensional("target", data["target"], rows)
        group_id = np.asarray(data["group_id"]) if "group_id" in data.files else None
        if group_id is not None and group_id.shape != (rows,):
            raise ValueError("group_id must have shape [N]")
        core_id = np.asarray(data["core_id"]) if "core_id" in data.files else None
        if core_id is not None and core_id.shape != (rows,):
            raise ValueError("core_id must have shape [N]")
        fold_id = np.asarray(data["fold_id"]) if "fold_id" in data.files else None
        if fold_id is not None and fold_id.shape != (rows,):
            raise ValueError("fold_id must have shape [N]")
        if fold_id is not None:
            expected_split = np.where(np.isin(fold_id, (0, 1)), 0, 1).astype(np.int8)
            if not np.array_equal(expected_split, probe_split):
                raise ValueError("probe_split disagrees with the frozen ph006 fold contract")

    return LatentSnapshot(
        path=path,
        metadata=metadata,
        sample_id=sample_id,
        dense_latent=dense,
        degraded_latent=degraded,
        predicted_dense_latent=predicted,
        response_strength=response,
        response_features=response_features,
        response_only_latent=response_only_latent,
        probe_split=probe_split,
        target=target,
        sample_weight=weights,
        group_id=group_id,
        core_id=core_id,
        fold_id=fold_id,
    )


def _weighted_center(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    mean = np.sum(values * weights[:, None], axis=0) / total
    return values - mean, mean


def linear_cka(x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Feature-space linear CKA, avoiding an O(N^2) Gram matrix."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("CKA inputs must have shapes [N,Dx] and [N,Dy]")
    weights = np.ones(len(x), dtype=np.float64) if weights is None else np.asarray(weights)
    xc, _ = _weighted_center(x, weights)
    yc, _ = _weighted_center(y, weights)
    root = np.sqrt(weights / weights.sum())[:, None]
    xw, yw = xc * root, yc * root
    cross = xw.T @ yw
    xx, yy = xw.T @ xw, yw.T @ yw
    numerator = float(np.square(cross).sum())
    denominator = float(np.sqrt(np.square(xx).sum() * np.square(yy).sum()))
    return numerator / denominator if denominator > 0 else 0.0


def spread_metrics(values: np.ndarray, weights: np.ndarray | None = None) -> dict:
    values = np.asarray(values, dtype=np.float64)
    weights = np.ones(len(values), dtype=np.float64) if weights is None else np.asarray(weights)
    centered, _ = _weighted_center(values, weights)
    covariance = (centered * np.sqrt(weights / weights.sum())[:, None]).T @ (
        centered * np.sqrt(weights / weights.sum())[:, None]
    )
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    total = float(eigenvalues.sum())
    if total <= 1.0e-20:
        effective_rank = 0.0
        participation_ratio = 0.0
    else:
        probability = eigenvalues[eigenvalues > 0] / total
        effective_rank = float(np.exp(-np.sum(probability * np.log(probability))))
        participation_ratio = float(total * total / np.square(eigenvalues).sum())
    std = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return {
        "dimension": int(values.shape[1]),
        "per_dimension_variance": np.diag(covariance).tolist(),
        "effective_rank": effective_rank,
        "effective_rank_fraction": effective_rank / values.shape[1],
        "participation_ratio": participation_ratio,
        "total_standard_deviation": float(np.sqrt(total)),
        "axis_std_min": float(std.min(initial=np.inf)),
        "axis_std_median": float(np.median(std)),
        "axis_std_max": float(std.max(initial=0.0)),
        "collapsed_axis_fraction_1e-4_median": float(
            np.mean(std <= max(1.0e-12, 1.0e-4 * float(np.median(std))))
        ),
    }


@dataclass(frozen=True)
class ProcrustesMap:
    mean_x: np.ndarray
    mean_y: np.ndarray
    rotation: np.ndarray
    scale: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values) - self.mean_x) @ self.rotation * self.scale + self.mean_y


def fit_procrustes(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None
) -> ProcrustesMap:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.shape[0] != y.shape[0]:
        raise ValueError("Procrustes pairs must have identical row counts")
    weights = np.ones(len(x), dtype=np.float64) if weights is None else np.asarray(weights)
    xc, mean_x = _weighted_center(x, weights)
    yc, mean_y = _weighted_center(y, weights)
    root = np.sqrt(weights / weights.sum())[:, None]
    u, _, vt = np.linalg.svd((xc * root).T @ (yc * root), full_matrices=False)
    rotation = u @ vt
    x_energy = float(np.sum(weights[:, None] * np.square(xc)))
    y_energy = float(np.sum(weights[:, None] * np.square(yc)))
    scale = np.sqrt(y_energy / x_energy) if x_energy > 0 else 0.0
    return ProcrustesMap(mean_x, mean_y, rotation, float(scale))


def _weighted_r2(truth: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> np.ndarray:
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    _, mean = _weighted_center(truth, weights)
    residual = np.sum(weights[:, None] * np.square(truth - prediction), axis=0)
    total = np.sum(weights[:, None] * np.square(truth - mean), axis=0)
    return np.where(total > 0, 1.0 - residual / total, np.nan)


def procrustes_crossfit(snapshot: LatentSnapshot) -> tuple[dict, np.ndarray]:
    fit = snapshot.probe_split == 0
    evaluate = snapshot.probe_split == 1
    mapping = fit_procrustes(
        snapshot.degraded_latent[fit],
        snapshot.dense_latent[fit],
        snapshot.sample_weight[fit],
    )
    mapped = mapping.transform(snapshot.degraded_latent)
    truth, prediction = snapshot.dense_latent[evaluate], mapped[evaluate]
    weights = snapshot.sample_weight[evaluate]
    r2 = _weighted_r2(truth, prediction, weights)
    centered, _ = _weighted_center(truth, weights)
    denominator = np.sum(weights[:, None] * np.square(centered)) / weights.sum()
    mse = np.sum(weights[:, None] * np.square(truth - prediction)) / (
        weights.sum() * truth.shape[1]
    )
    return (
        {
            "latent_r2_per_dimension": r2.tolist(),
            "latent_r2_macro": float(np.nanmean(r2)),
            "normalized_rmse": float(np.sqrt(mse / max(denominator / truth.shape[1], 1e-20))),
        },
        mapped,
    )


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denominator) if denominator > 0 else 0.0


def cca_crossfit(snapshot: LatentSnapshot, ridge: float = 1.0e-4) -> dict:
    fit = snapshot.probe_split == 0
    evaluate = snapshot.probe_split == 1
    x, y = snapshot.degraded_latent, snapshot.dense_latent
    weights = snapshot.sample_weight
    x_fit, mx = _weighted_center(x[fit], weights[fit])
    y_fit, my = _weighted_center(y[fit], weights[fit])
    root = np.sqrt(weights[fit] / weights[fit].sum())[:, None]
    cxx = (x_fit * root).T @ (x_fit * root)
    cyy = (y_fit * root).T @ (y_fit * root)
    cxy = (x_fit * root).T @ (y_fit * root)

    def inverse_root(covariance: np.ndarray) -> np.ndarray:
        eigenvalues, vectors = np.linalg.eigh(covariance)
        floor = ridge * max(float(eigenvalues.max(initial=0.0)), 1.0)
        return (vectors * (1.0 / np.sqrt(np.maximum(eigenvalues, floor)))) @ vectors.T

    wx, wy = inverse_root(cxx), inverse_root(cyy)
    u, singular, vt = np.linalg.svd(wx @ cxy @ wy, full_matrices=False)
    x_projection = wx @ u
    y_projection = wy @ vt.T
    x_eval = (x[evaluate] - mx) @ x_projection
    y_eval = (y[evaluate] - my) @ y_projection
    correlations = np.asarray(
        [_correlation(x_eval[:, index], y_eval[:, index]) for index in range(len(singular))]
    )
    top = min(8, len(correlations))
    return {
        "fit_canonical_correlations": singular.tolist(),
        "evaluation_correlations": correlations.tolist(),
        "evaluation_mean_top8": float(np.mean(correlations[:top])),
        "evaluation_median_top8": float(np.median(correlations[:top])),
    }


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norm, 1.0e-12)


def response_matched_permutation(
    response: np.ndarray, *, seed: int, bins: int = 10
) -> np.ndarray:
    """Shuffle pairs within response quantiles, preserving the response shortcut."""
    response = np.asarray(response, dtype=np.float64)
    rng = np.random.default_rng(seed)
    edges = np.quantile(response, np.linspace(0.0, 1.0, bins + 1))
    labels = np.clip(np.searchsorted(edges[1:-1], response, side="right"), 0, bins - 1)
    permutation = np.arange(len(response))
    for label in np.unique(labels):
        rows = np.flatnonzero(labels == label)
        permutation[rows] = rng.permutation(rows)
    return permutation


def retrieval_metrics(
    mapped_student: np.ndarray,
    dense: np.ndarray,
    sample_id: np.ndarray,
    *,
    max_rows: int,
    seed: int,
    response_strength: np.ndarray | None = None,
) -> dict:
    rows = len(mapped_student)
    rng = np.random.default_rng(seed)
    chosen = (
        np.arange(rows)
        if rows <= max_rows
        else np.sort(rng.choice(rows, size=max_rows, replace=False))
    )
    student = mapped_student[chosen]
    teacher = dense[chosen]
    teacher_mean = teacher.mean(axis=0)
    student = _normalize_rows(student - teacher_mean)
    teacher = _normalize_rows(teacher - teacher_mean)
    similarity = student @ teacher.T
    diagonal = np.diag(similarity)
    rank = 1 + np.sum(similarity > diagonal[:, None], axis=1)
    result = {
        "rows": int(len(chosen)),
        "top1": float(np.mean(rank == 1)),
        "top5": float(np.mean(rank <= 5)),
        "mean_reciprocal_rank": float(np.mean(1.0 / rank)),
        "median_rank": float(np.median(rank)),
        "sample_id_sha256": hashlib.sha256(
            np.ascontiguousarray(sample_id[chosen]).view(np.uint8)
        ).hexdigest(),
    }
    permutation = rng.permutation(len(chosen))
    shuffled_similarity = student @ teacher[permutation].T
    shuffled_diagonal = np.diag(shuffled_similarity)
    shuffled_rank = 1 + np.sum(
        shuffled_similarity > shuffled_diagonal[:, None], axis=1
    )
    result["shuffled"] = {
        "top1": float(np.mean(shuffled_rank == 1)),
        "top5": float(np.mean(shuffled_rank <= 5)),
        "mean_reciprocal_rank": float(np.mean(1.0 / shuffled_rank)),
        "median_rank": float(np.median(shuffled_rank)),
    }
    result["mrr_over_shuffle"] = (
        result["mean_reciprocal_rank"] - result["shuffled"]["mean_reciprocal_rank"]
    )
    if response_strength is not None:
        response = np.asarray(response_strength)[chosen]
        matched_permutation = response_matched_permutation(response, seed=seed + 1)
        matched_similarity = student @ teacher[matched_permutation].T
        matched_diagonal = np.diag(matched_similarity)
        matched_rank = 1 + np.sum(matched_similarity > matched_diagonal[:, None], axis=1)
        result["response_matched_shuffled"] = {
            "top1": float(np.mean(matched_rank == 1)),
            "top5": float(np.mean(matched_rank <= 5)),
            "mean_reciprocal_rank": float(np.mean(1.0 / matched_rank)),
            "median_rank": float(np.median(matched_rank)),
        }
        result["mrr_over_response_matched_shuffle"] = (
            result["mean_reciprocal_rank"]
            - result["response_matched_shuffled"]["mean_reciprocal_rank"]
        )
    return result


@dataclass(frozen=True)
class RidgeProbe:
    mean_x: np.ndarray
    mean_y: np.ndarray
    coefficient: np.ndarray

    def predict(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values) - self.mean_x) @ self.coefficient + self.mean_y


def fit_ridge_probe(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    ridge: float = 1.0e-3,
) -> RidgeProbe:
    x_centered, mean_x = _weighted_center(np.asarray(x), weights)
    y_centered, mean_y = _weighted_center(np.asarray(y), weights)
    root = np.sqrt(weights / weights.sum())[:, None]
    xw, yw = x_centered * root, y_centered * root
    covariance = xw.T @ xw
    scale = max(float(np.trace(covariance)) / max(covariance.shape[0], 1), 1.0e-12)
    coefficient = np.linalg.solve(
        covariance + ridge * scale * np.eye(covariance.shape[0]), xw.T @ yw
    )
    return RidgeProbe(mean_x, mean_y, coefficient)


def downstream_probes(snapshot: LatentSnapshot) -> tuple[dict | None, dict[str, np.ndarray]]:
    if snapshot.target is None:
        return None, {}
    fit = snapshot.probe_split == 0
    evaluate = snapshot.probe_split == 1
    representations = {
        "dense_teacher": snapshot.dense_latent,
        "degraded_student": snapshot.degraded_latent,
        "response_covariates_only": snapshot.response_features,
    }
    if snapshot.response_only_latent is not None:
        representations["response_only_encoder"] = snapshot.response_only_latent
    predictor_trained = bool(
        snapshot.metadata["arm"] == "jepa"
        and snapshot.metadata.get("predictor_trained", False)
    )
    if snapshot.predicted_dense_latent is not None and predictor_trained:
        representations["predicted_teacher_space"] = snapshot.predicted_dense_latent
    report, predictions = {}, {}
    for name, values in representations.items():
        probe = fit_ridge_probe(
            values[fit], snapshot.target[fit], snapshot.sample_weight[fit]
        )
        prediction = probe.predict(values)
        predictions[name] = prediction
        r2 = _weighted_r2(
            snapshot.target[evaluate], prediction[evaluate], snapshot.sample_weight[evaluate]
        )
        report[name] = {"r2": r2.tolist(), "macro_r2": float(np.nanmean(r2))}
    response_control = (
        "response_only_encoder"
        if "response_only_encoder" in report
        else "response_covariates_only"
    )
    report["response_control"] = response_control
    report["response_control_strength"] = (
        "full_patch_response_only_encoder"
        if response_control == "response_only_encoder"
        else "pointwise_response_covariates_only_advisory"
    )
    report["student_minus_response_macro_r2"] = (
        report["degraded_student"]["macro_r2"]
        - report[response_control]["macro_r2"]
    )
    report["dense_minus_student_macro_r2"] = (
        report["dense_teacher"]["macro_r2"] - report["degraded_student"]["macro_r2"]
    )
    return report, predictions


def _cosine_pairs(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sum(_normalize_rows(x) * _normalize_rows(y), axis=1)


def displacement_metrics(mapped: np.ndarray, dense: np.ndarray, weights: np.ndarray) -> dict:
    centered, _ = _weighted_center(dense, weights)
    scale = np.sqrt(np.sum(weights[:, None] * np.square(centered)) / weights.sum())
    displacement = np.linalg.norm(mapped - dense, axis=1) / max(scale, 1.0e-20)
    return {
        "mean": float(np.average(displacement, weights=weights)),
        "median": float(np.median(displacement)),
        "p90": float(np.quantile(displacement, 0.90)),
        "p99": float(np.quantile(displacement, 0.99)),
        "paired_cosine_mean": float(
            np.average(_cosine_pairs(mapped, dense), weights=weights)
        ),
    }


def response_strata(
    snapshot: LatentSnapshot,
    mapped: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    *,
    bins: int = 4,
) -> list[dict]:
    evaluate = snapshot.probe_split == 1
    response = snapshot.response_strength[evaluate]
    edges = np.quantile(response, np.linspace(0.0, 1.0, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    rows = []
    dense = snapshot.dense_latent[evaluate]
    student = mapped[evaluate]
    weights = snapshot.sample_weight[evaluate]
    for index in range(bins):
        chosen = (response > edges[index]) & (response <= edges[index + 1])
        if index == 0:
            chosen = (response >= edges[index]) & (response <= edges[index + 1])
        if int(chosen.sum()) < 8:
            continue
        row = {
            "bin": index,
            "rows": int(chosen.sum()),
            "response_min": float(response[chosen].min()),
            "response_max": float(response[chosen].max()),
            "response_median": float(np.median(response[chosen])),
            "cka_mapped_to_dense": linear_cka(
                student[chosen], dense[chosen], weights[chosen]
            ),
            "displacement": displacement_metrics(
                student[chosen], dense[chosen], weights[chosen]
            ),
        }
        if snapshot.target is not None and "degraded_student" in predictions:
            truth = snapshot.target[evaluate][chosen]
            prediction = predictions["degraded_student"][evaluate][chosen]
            rmse = np.sqrt(
                np.average(np.square(truth - prediction), weights=weights[chosen], axis=0)
            )
            row["student_target_rmse"] = rmse.tolist()
            row["student_target_rmse_macro"] = float(np.mean(rmse))
        rows.append(row)
    return rows


def evaluate_snapshot(
    snapshot: LatentSnapshot,
    *,
    max_retrieval_rows: int = 2048,
    seed: int = 1729,
    thresholds: Mapping[str, float] = DEFAULT_THRESHOLDS,
) -> dict:
    evaluate = snapshot.probe_split == 1
    weights = snapshot.sample_weight[evaluate]
    dense = snapshot.dense_latent[evaluate]
    degraded = snapshot.degraded_latent[evaluate]
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(dense))
    response_permutation = response_matched_permutation(
        snapshot.response_strength[evaluate], seed=seed + 1
    )
    cka_native = linear_cka(degraded, dense, weights)
    cka_shuffle = linear_cka(degraded, dense[shuffled], weights)
    cka_response_shuffle = linear_cka(
        degraded, dense[response_permutation], weights
    )
    procrustes_report, mapped_all = procrustes_crossfit(snapshot)
    mapped = mapped_all[evaluate]
    retrieval = retrieval_metrics(
        mapped,
        dense,
        snapshot.sample_id[evaluate],
        max_rows=max_retrieval_rows,
        seed=seed,
        response_strength=snapshot.response_strength[evaluate],
    )
    probe_report, probe_predictions = downstream_probes(snapshot)
    strata = response_strata(snapshot, mapped_all, probe_predictions)
    spread = {
        "dense_teacher": spread_metrics(dense, weights),
        "degraded_student": spread_metrics(degraded, weights),
        "mapped_student": spread_metrics(mapped, weights),
    }
    predicted_report = None
    predictor_trained = bool(
        snapshot.metadata["arm"] == "jepa"
        and snapshot.metadata.get("predictor_trained", False)
    )
    if snapshot.predicted_dense_latent is not None and predictor_trained:
        predicted = snapshot.predicted_dense_latent[evaluate]
        predicted_report = {
            "cka_to_dense": linear_cka(predicted, dense, weights),
            "displacement_to_dense": displacement_metrics(predicted, dense, weights),
            "spread": spread_metrics(predicted, weights),
        }
    response_encoder_report = None
    if snapshot.response_only_latent is not None:
        response_latent = snapshot.response_only_latent[evaluate]
        response_encoder_report = {
            "cka_to_dense": linear_cka(response_latent, dense, weights),
            "cka_to_student": linear_cka(response_latent, degraded, weights),
            "spread": spread_metrics(response_latent, weights),
        }
    collapse = any(
        row["effective_rank_fraction"] < thresholds["minimum_effective_rank_fraction"]
        or row["total_standard_deviation"] < thresholds["minimum_total_standard_deviation"]
        for name, row in spread.items()
        if name != "mapped_student"
    )
    shared_subspace = (
        cka_native - cka_response_shuffle >= thresholds["minimum_cka_over_shuffle"]
        and retrieval.get("mrr_over_response_matched_shuffle", retrieval["mrr_over_shuffle"])
        >= thresholds["minimum_retrieval_mrr_over_shuffle"]
        and not collapse
    )
    response_shortcut = False
    forced_equality = False
    low_response_masking = False
    if probe_report is not None:
        response_shortcut = (
            probe_report["student_minus_response_macro_r2"]
            <= thresholds["response_shortcut_margin"]
        )
        alignment = (
            predicted_report["cka_to_dense"] if predicted_report is not None else cka_native
        )
        forced_equality = (
            alignment >= thresholds["high_alignment"]
            and probe_report["dense_minus_student_macro_r2"]
            >= thresholds["material_probe_gap"]
        )
        if len(strata) >= 2 and "student_target_rmse_macro" in strata[0]:
            error_ratio = strata[0]["student_target_rmse_macro"] / max(
                strata[-1]["student_target_rmse_macro"], 1.0e-12
            )
            alignment_ratio = strata[0]["cka_mapped_to_dense"] / max(
                strata[-1]["cka_mapped_to_dense"], 1.0e-12
            )
            low_response_masking = (
                error_ratio >= thresholds["low_response_error_ratio"]
                and alignment_ratio >= thresholds["deceptively_flat_alignment_ratio"]
            )
    return {
        "epoch": int(snapshot.metadata["epoch"]),
        "global_step": int(snapshot.metadata["global_step"]),
        "phase": str(snapshot.metadata["phase"]),
        "arm": str(snapshot.metadata["arm"]),
        "rows": int(len(snapshot.sample_id)),
        "evaluation_rows": int(evaluate.sum()),
        "linear_cka": {
            "native_student_to_dense": cka_native,
            "shuffled_pair_control": cka_shuffle,
            "paired_over_shuffled": cka_native - cka_shuffle,
            "response_matched_shuffled_control": cka_response_shuffle,
            "paired_over_response_matched_shuffled": cka_native - cka_response_shuffle,
        },
        "crossfit_procrustes": procrustes_report,
        "crossfit_cca": cca_crossfit(snapshot),
        "cross_view_retrieval": retrieval,
        "latent_displacement": displacement_metrics(mapped, dense, weights),
        "spread": spread,
        "predictor": predicted_report,
        "predictor_trained": predictor_trained,
        "response_only_encoder": response_encoder_report,
        "downstream_linear_probe": probe_report,
        "response_strata": strata,
        "risk_signals": {
            "collapse": collapse,
            "response_only_shortcut": response_shortcut,
            "high_alignment_with_dense_student_probe_gap": forced_equality,
            "alignment_masks_low_response_target_error": low_response_masking,
        },
        "shared_predictable_subspace_gate": bool(shared_subspace),
    }


def _validate_series(snapshots: Iterable[LatentSnapshot]) -> list[LatentSnapshot]:
    snapshots = sorted(
        list(snapshots),
        key=lambda row: (
            int(row.metadata["global_step"]),
            int(row.metadata["epoch"]),
        ),
    )
    if not snapshots:
        raise ValueError("at least one snapshot is required")
    reference = snapshots[0]
    for snapshot in snapshots[1:]:
        if snapshot.metadata["run_id"] != reference.metadata["run_id"]:
            raise ValueError("all snapshots must belong to one run_id")
        if snapshot.metadata["arm"] != reference.metadata["arm"]:
            raise ValueError("matched arm changed across checkpoints")
        if snapshot.metadata["phase"] != reference.metadata["phase"]:
            raise ValueError("selection phase changed across checkpoints")
        if not np.array_equal(snapshot.sample_id, reference.sample_id):
            raise ValueError("sample_id order must be frozen across checkpoints")
        if not np.array_equal(snapshot.probe_split, reference.probe_split):
            raise ValueError("probe_split must be frozen across checkpoints")
        if not np.array_equal(snapshot.response_strength, reference.response_strength):
            raise ValueError("response_strength changed across checkpoints")
        if not np.array_equal(snapshot.response_features, reference.response_features):
            raise ValueError("response_features changed across checkpoints")
        if not np.array_equal(snapshot.sample_weight, reference.sample_weight):
            raise ValueError("sample_weight changed across checkpoints")
        if (snapshot.target is None) != (reference.target is None) or (
            snapshot.target is not None
            and not np.array_equal(snapshot.target, reference.target)
        ):
            raise ValueError("downstream probe targets changed across checkpoints")
        for name in ("core_id", "fold_id", "group_id"):
            current = getattr(snapshot, name)
            original = getattr(reference, name)
            if (current is None) != (original is None) or (
                current is not None and not np.array_equal(current, original)
            ):
                raise ValueError(f"{name} changed across checkpoints")
        if snapshot.dense_latent.shape != reference.dense_latent.shape:
            raise ValueError("teacher latent shape changed across checkpoints")
        if snapshot.degraded_latent.shape != reference.degraded_latent.shape:
            raise ValueError("student latent shape changed across checkpoints")
        if (snapshot.response_only_latent is None) != (
            reference.response_only_latent is None
        ) or (
            snapshot.response_only_latent is not None
            and snapshot.response_only_latent.shape
            != reference.response_only_latent.shape
        ):
            raise ValueError("response-only latent shape changed across checkpoints")
    steps = [int(row.metadata["global_step"]) for row in snapshots]
    if len(set(steps)) != len(steps):
        raise ValueError("checkpoint global_step values must be unique")
    return snapshots


def temporal_risk_signals(
    metrics: list[dict], thresholds: Mapping[str, float] = DEFAULT_THRESHOLDS
) -> dict:
    if len(metrics) < 2:
        return {
            "available": False,
            "alignment_gain_with_probe_regression": False,
            "alignment_gain_with_rank_loss": False,
        }
    first, last = metrics[0], metrics[-1]
    first_alignment = (
        first["predictor"]["cka_to_dense"]
        if first["predictor"] is not None
        else first["linear_cka"]["native_student_to_dense"]
    )
    last_alignment = (
        last["predictor"]["cka_to_dense"]
        if last["predictor"] is not None
        else last["linear_cka"]["native_student_to_dense"]
    )
    alignment_gain = last_alignment - first_alignment
    rank_loss = (
        first["spread"]["degraded_student"]["effective_rank_fraction"]
        - last["spread"]["degraded_student"]["effective_rank_fraction"]
    )
    probe_regression = None
    if (
        first["downstream_linear_probe"] is not None
        and last["downstream_linear_probe"] is not None
    ):
        probe_regression = (
            first["downstream_linear_probe"]["degraded_student"]["macro_r2"]
            - last["downstream_linear_probe"]["degraded_student"]["macro_r2"]
        )
    return {
        "available": True,
        "alignment_gain": alignment_gain,
        "student_effective_rank_fraction_loss": rank_loss,
        "student_probe_macro_r2_regression": probe_regression,
        "alignment_gain_with_probe_regression": bool(
            alignment_gain >= thresholds["material_alignment_gain"]
            and probe_regression is not None
            and probe_regression > thresholds["maximum_probe_regression"]
        ),
        "alignment_gain_with_rank_loss": bool(
            alignment_gain >= thresholds["material_alignment_gain"]
            and rank_loss >= thresholds["material_rank_fraction_loss"]
        ),
    }


def registered_status_gate(
    snapshots: list[LatentSnapshot],
    metrics: list[dict],
    temporal: Mapping,
) -> dict:
    """Derive pass/advisory/fail from the frozen 0/250/500 canary gates."""
    arm = str(snapshots[0].metadata["arm"])
    by_step = {int(row["global_step"]): row for row in metrics}
    missing = [step for step in REGISTERED_TRAJECTORY_STEPS if step not in by_step]
    trajectory_complete = not missing
    response_encoder_available = all(
        snapshot.response_only_latent is not None for snapshot in snapshots
    )

    if arm != "jepa":
        return {
            "version": GATE_VERSION,
            "status": "advisory",
            "pass": False,
            "arm": arm,
            "required_steps": list(REGISTERED_TRAJECTORY_STEPS),
            "observed_steps": sorted(by_step),
            "missing_steps": missing,
            "response_only_encoder_available": response_encoder_available,
            "reasons": [
                "matched non-JEPA arms are reference controls; their untrained "
                "predictor cannot license a representation-alignment pass"
            ],
        }

    gate_row = by_step.get(500)
    gate_probe = None if gate_row is None else gate_row["downstream_linear_probe"]
    response_control_evaluable = bool(
        gate_probe is not None
        and gate_probe.get("response_control") == "response_only_encoder"
    )
    reasons: list[str] = []
    fatal = bool(
        gate_row is not None and gate_row["risk_signals"]["collapse"]
    )
    fatal = fatal or bool(temporal.get("alignment_gain_with_probe_regression"))
    fatal = fatal or bool(temporal.get("alignment_gain_with_rank_loss"))
    if fatal:
        reasons.append(
            "step-500 collapse or an adverse alignment trajectory violates a hard guard"
        )
    if not trajectory_complete:
        reasons.append("registered 0/250/500 trajectory is incomplete")

    scientific_failure = False
    ambiguous = False
    if gate_row is not None:
        if not gate_row["shared_predictable_subspace_gate"]:
            scientific_failure = True
            reasons.append("step-500 shared predictable-subspace gate failed")
        if gate_row["risk_signals"]["response_only_shortcut"]:
            if response_control_evaluable:
                scientific_failure = True
                reasons.append("step-500 student does not beat the response-only control")
            else:
                reasons.append(
                    "the weaker pointwise-response fallback flags a possible shortcut; "
                    "this is advisory rather than a registered failure"
                )
        ambiguous = bool(
            gate_row["risk_signals"]["high_alignment_with_dense_student_probe_gap"]
            or gate_row["risk_signals"]["alignment_masks_low_response_target_error"]
        )
        if ambiguous:
            reasons.append("step-500 hallucination-risk proxy requires scientific review")
    if not response_encoder_available:
        reasons.append(
            "explicit response-only encoder is absent; pointwise response covariates "
            "are a weaker advisory shortcut control"
        )
    elif not response_control_evaluable:
        reasons.append(
            "the explicit response-only encoder lacks a held-out target probe at step 500"
        )

    if fatal or scientific_failure:
        status = "fail"
    elif (
        trajectory_complete
        and gate_row is not None
        and gate_row["shared_predictable_subspace_gate"]
        and not ambiguous
        and response_encoder_available
        and response_control_evaluable
    ):
        status = "pass"
        reasons.append("all registered latent canary gates pass at step 500")
    else:
        status = "advisory"
    return {
        "version": GATE_VERSION,
        "status": status,
        "pass": status == "pass",
        "arm": arm,
        "required_steps": list(REGISTERED_TRAJECTORY_STEPS),
        "observed_steps": sorted(by_step),
        "missing_steps": missing,
        "response_only_encoder_available": response_encoder_available,
        "response_only_control_evaluable": response_control_evaluable,
        "reasons": reasons,
    }


def evaluate_series(
    paths: Iterable[Path],
    *,
    max_retrieval_rows: int = 2048,
    seed: int = 1729,
    thresholds: Mapping[str, float] = DEFAULT_THRESHOLDS,
) -> dict:
    if dict(thresholds) != DEFAULT_THRESHOLDS:
        raise ValueError(
            "scientific series reports must use the registered latent gate thresholds"
        )
    snapshots = _validate_series(load_latent_snapshot(Path(path)) for path in paths)
    metrics = [
        evaluate_snapshot(
            snapshot,
            max_retrieval_rows=max_retrieval_rows,
            seed=seed,
            thresholds=thresholds,
        )
        for snapshot in snapshots
    ]
    temporal = temporal_risk_signals(metrics, thresholds)
    # Epoch-end exports may coexist with the frozen canary trajectory.  They are
    # useful for visualization, but must not silently move a registered gate
    # from 0 -> 500 to (for example) 0 -> 743.
    registered_metrics = [
        row
        for row in metrics
        if int(row["global_step"]) in REGISTERED_TRAJECTORY_STEPS
    ]
    registered_temporal = temporal_risk_signals(registered_metrics, thresholds)
    status_gate = registered_status_gate(snapshots, metrics, registered_temporal)
    per_checkpoint_risks = [
        any(row["risk_signals"].values()) for row in metrics
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "status": status_gate["status"],
        "pass": status_gate["pass"],
        "run_id": str(snapshots[0].metadata["run_id"]),
        "selection_phase": str(snapshots[0].metadata["phase"]),
        "sealed_phase": SEALED_PHASE,
        "sealed_phase_opened": False,
        "snapshot_sources": [
            {
                "path": str(snapshot.path),
                "sha256": sha256(snapshot.path),
                "epoch": int(snapshot.metadata["epoch"]),
                "global_step": int(snapshot.metadata["global_step"]),
            }
            for snapshot in snapshots
        ],
        "thresholds": dict(thresholds),
        "checkpoints": metrics,
        "temporal_risk_signals": temporal,
        "registered_temporal_risk_signals": registered_temporal,
        "registered_status_gate": status_gate,
        "interpretation_contract": {
            "shared_representation_meaning": (
                "paired views preserve predictable relational geometry and target-relevant "
                "information; latent coordinates are not ordered or bounded by the dense view"
            ),
            "information_bound": (
                "I(environment; z_student) <= I(environment; V_final); this is an "
                "information inequality, not a coordinate-wise latent bound"
            ),
            "many_to_one_observation_implication": (
                "when multiple latent density fields are compatible with V_final, a "
                "deterministic student can preserve only shared/predictable factors and "
                "cannot encode the missing conditional distribution by alignment alone"
            ),
            "forced_coordinate_equality_required": False,
            "latent_alignment_is_posterior_calibration": False,
            "posterior_overconfidence_claim_licensed": False,
            "posterior_retraining_and_full_p12_recalibration_required_if_promoted": True,
            "risk_proxy_present": bool(any(per_checkpoint_risks) or any(
                value is True for key, value in temporal.items() if key != "available"
            )),
            "notes": [
                "CKA/CCA can be high even when individual dense-view details are unrecoverable.",
                "A deterministic JEPA student represents a summary, not p(lambda|observations).",
                "Low-response alignment with worsening target error is a "
                "hallucination-risk proxy, not a calibration test.",
                "Promotion requires a newly fitted posterior head plus SBC, TARP "
                "and conditional coverage.",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-retrieval-rows", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    report = evaluate_series(
        args.snapshots,
        max_retrieval_rows=args.max_retrieval_rows,
        seed=args.seed,
    )
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
