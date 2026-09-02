#!/usr/bin/env python3
"""Method-independent sampling, scoring and selection for P12-F challengers."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np


class FieldSampler(Protocol):
    def sample(
        self, condition: np.ndarray, draws: int, seed: int
    ) -> np.ndarray:
        """Return delta_R7 with shape [draw,x,y,z]."""


@dataclass(frozen=True)
class FieldSampleContract:
    method: str
    core_id: int
    samples: np.ndarray
    truth: np.ndarray
    support: np.ndarray

    def validate(self) -> None:
        samples = np.asarray(self.samples)
        truth = np.asarray(self.truth)
        support = np.asarray(self.support, dtype=bool)
        if samples.ndim != 4 or samples.shape[1:] != truth.shape:
            raise ValueError("field samples must have shape [draw,x,y,z]")
        if support.shape != truth.shape:
            raise ValueError("support and truth geometry mismatch")
        if samples.shape[0] < 2 or not np.any(support):
            raise ValueError("field posterior has no usable draws/support")
        if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(truth)):
            raise ValueError("field posterior contains non-finite values")
        if float(np.var(samples[:, support])) <= 1e-10:
            raise ValueError("field posterior is degenerate")


def select_truth_free_panel(
    *,
    core_id: np.ndarray,
    shell: np.ndarray,
    cap: np.ndarray,
    response: np.ndarray,
    boundary_distance: np.ndarray,
    per_shell: int = 32,
    seed: int = 42,
) -> np.ndarray:
    """Select 32 cores/shell balanced across cap, response and boundary strata.

    This function receives observation metadata only. It has no truth argument and its
    output is therefore safe to freeze before evaluating any candidate.
    """
    core_id = np.asarray(core_id, dtype=np.int64)
    shell = np.asarray(shell, dtype=np.int8)
    cap = np.asarray(cap, dtype=np.int8)
    response = np.asarray(response, dtype=np.float64)
    boundary = np.asarray(boundary_distance, dtype=np.float64)
    if not (len(core_id) == len(shell) == len(cap) == len(response) == len(boundary)):
        raise ValueError("panel covariates are not aligned")
    if len(np.unique(core_id)) != len(core_id):
        raise ValueError("core IDs must be unique")
    if not np.all(np.isfinite(response)) or not np.all(np.isfinite(boundary)):
        raise ValueError("panel response covariates must be finite")
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    for shell_value in range(4):
        candidates = np.flatnonzero(shell == shell_value)
        if len(candidates) < per_shell:
            raise ValueError(f"shell {shell_value} has fewer than {per_shell} cores")
        response_edges = np.quantile(response[candidates], [0.0, 0.5, 1.0])
        boundary_edges = np.quantile(boundary[candidates], [0.0, 0.5, 1.0])
        response_bin = np.searchsorted(response_edges[1:-1], response[candidates])
        boundary_bin = np.searchsorted(boundary_edges[1:-1], boundary[candidates])
        group = cap[candidates].astype(int) * 4 + response_bin * 2 + boundary_bin
        shell_parts = []
        for group_value in range(8):
            rows = candidates[group == group_value]
            take = min(4, len(rows))
            if take:
                shell_parts.append(rng.choice(rows, size=take, replace=False))
        selected = np.concatenate(shell_parts) if shell_parts else np.empty(0, dtype=int)
        if len(selected) < per_shell:
            remaining = np.setdiff1d(candidates, selected)
            extra = rng.choice(remaining, size=per_shell - len(selected), replace=False)
            selected = np.concatenate((selected, extra))
        elif len(selected) > per_shell:
            selected = rng.choice(selected, size=per_shell, replace=False)
        chosen.append(selected)
    result = np.sort(np.concatenate(chosen).astype(np.int64))
    if len(result) != 4 * per_shell:
        raise RuntimeError("truth-free panel size mismatch")
    return result


def fixed_indices(size: int, count: int, *, seed: int) -> np.ndarray:
    if not 0 < count <= size:
        raise ValueError("fixed-index count is invalid")
    return np.sort(np.random.default_rng(seed).choice(size, size=count, replace=False))


def energy_score(samples: np.ndarray, truth: np.ndarray) -> float:
    draws = np.asarray(samples, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    if draws.ndim != 2 or target.shape != (draws.shape[1],):
        raise ValueError("energy score expects [draw,features] and [features]")
    first = np.mean(np.linalg.norm(draws - target[None], axis=1))
    second = np.mean(
        np.linalg.norm(draws[:, None, :] - draws[None, :, :], axis=-1)
    )
    return float(first - 0.5 * second)


def fixed_pair_indices(size: int, count: int, *, seed: int) -> np.ndarray:
    if size < 2 or count <= 0:
        raise ValueError("variogram pair request is invalid")
    rng = np.random.default_rng(seed)
    left = rng.integers(0, size, size=count)
    right = rng.integers(0, size - 1, size=count)
    right = right + (right >= left)
    return np.stack((left, right), axis=1).astype(np.int64)


def variogram_score(
    samples: np.ndarray,
    truth: np.ndarray,
    pairs: np.ndarray,
    *,
    power: float = 0.5,
) -> float:
    draws = np.asarray(samples, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    pairs = np.asarray(pairs, dtype=np.int64)
    if draws.ndim != 2 or target.shape != (draws.shape[1],):
        raise ValueError("variogram score expects [draw,features]")
    if pairs.ndim != 2 or pairs.shape[1] != 2 or np.any(pairs < 0):
        raise ValueError("invalid variogram pair array")
    truth_difference = np.abs(target[pairs[:, 0]] - target[pairs[:, 1]]) ** power
    draw_difference = np.mean(
        np.abs(draws[:, pairs[:, 0]] - draws[:, pairs[:, 1]]) ** power,
        axis=0,
    )
    return float(np.mean(np.square(truth_difference - draw_difference)))


def haar_coarse(field: np.ndarray, levels: int = 2) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64)
    for _ in range(levels):
        nx, ny, nz = (2 * (value // 2) for value in values.shape[-3:])
        values = values[..., :nx, :ny, :nz]
        values = (
            values[..., 0::2, 0::2, 0::2]
            + values[..., 1::2, 0::2, 0::2]
            + values[..., 0::2, 1::2, 0::2]
            + values[..., 1::2, 1::2, 0::2]
            + values[..., 0::2, 0::2, 1::2]
            + values[..., 1::2, 0::2, 1::2]
            + values[..., 0::2, 1::2, 1::2]
            + values[..., 1::2, 1::2, 1::2]
        ) / 8.0
    return values


def core_joint_scores(
    samples: np.ndarray,
    truth: np.ndarray,
    support: np.ndarray,
    *,
    feature_count: int = 1024,
    pair_count: int = 2048,
    seed: int = 42,
) -> dict:
    samples = np.asarray(samples, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    support = np.asarray(support, dtype=bool)
    selected = np.flatnonzero(support.ravel())
    index = fixed_indices(len(selected), min(feature_count, len(selected)), seed=seed)
    voxel = selected[index]
    draw_features = samples.reshape(samples.shape[0], -1)[:, voxel]
    target_features = truth.ravel()[voxel]
    pairs = fixed_pair_indices(len(voxel), min(pair_count, len(voxel) * 2), seed=seed + 1)
    coarse_draws = haar_coarse(samples).reshape(samples.shape[0], -1)
    coarse_truth = haar_coarse(truth).ravel()
    coarse_index = fixed_indices(
        len(coarse_truth), min(feature_count, len(coarse_truth)), seed=seed + 2
    )
    return {
        "energy": energy_score(draw_features, target_features),
        "variogram_p0p5": variogram_score(
            draw_features, target_features, pairs, power=0.5
        ),
        "coarse_energy": energy_score(
            coarse_draws[:, coarse_index], coarse_truth[coarse_index]
        ),
    }


def paired_core_bootstrap(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    replicates: int = 4000,
    seed: int = 42,
) -> dict:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if candidate.shape != reference.shape or candidate.ndim != 1:
        raise ValueError("paired bootstrap arrays must be aligned one-dimensional cores")
    if len(candidate) < 2 or not np.all(np.isfinite(candidate + reference)):
        raise ValueError("paired bootstrap inputs are invalid")
    difference = (reference - candidate) / np.maximum(np.abs(reference), 1e-12)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(difference), size=(replicates, len(difference)))
    distribution = difference[index].mean(axis=1)
    return {
        "fractional_improvement": float(difference.mean()),
        "interval95": np.quantile(distribution, [0.025, 0.975]).tolist(),
        "cores": int(len(difference)),
        "replicates": int(replicates),
        "resampling_unit": "authoritative patch core",
        "voxel_independent_resampling": False,
    }


def maximum_coverage_error(report: Mapping[str, Any], keys: tuple[str, ...]) -> float:
    values = []
    for key in keys:
        rows = report.get(key, {})
        if isinstance(rows, Mapping):
            for row in rows.values():
                if isinstance(row, Mapping) and "maximum_coverage_error" in row:
                    values.append(float(row["maximum_coverage_error"]))
    return max(values, default=float("inf"))


def candidate_gate(
    candidate: Mapping[str, Any],
    correlated_gaussian: Mapping[str, Any],
    *,
    thresholds: Mapping[str, float],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not candidate.get("finite_non_degenerate", False):
        reasons.append("non_finite_or_degenerate")
    if float(candidate.get("tarp_maximum_deviation", np.inf)) > thresholds["tarp"]:
        reasons.append("tarp")
    global_errors = candidate.get("global_coverage_error", {})
    if max(map(float, global_errors.values()), default=np.inf) > thresholds["global_coverage"]:
        reasons.append("global_coverage")
    if float(candidate.get("maximum_conditional_coverage_error", np.inf)) > thresholds[
        "conditional_coverage"
    ]:
        reasons.append("conditional_coverage")
    bootstrap = candidate.get("joint_score_vs_g1_bootstrap", {})
    if float(bootstrap.get("fractional_improvement", -np.inf)) < thresholds[
        "joint_improvement"
    ]:
        reasons.append("joint_improvement")
    interval = bootstrap.get("interval95", [-np.inf, np.inf])
    if float(interval[0]) <= 0.0:
        reasons.append("joint_improvement_interval")
    candidate_scores = candidate.get("proper_scores", {})
    reference_scores = correlated_gaussian.get("proper_scores", {})
    for name, reference in reference_scores.items():
        if name == "primary_joint":
            continue
        value = float(candidate_scores.get(name, np.inf))
        if value > float(reference) * (1.0 + thresholds["other_score_worsening"]):
            reasons.append(f"proper_score_worse:{name}")
    return not reasons, reasons


def freeze_method_selection(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    thresholds: Mapping[str, float],
) -> dict:
    if "gaussian_correlated_g1" not in reports:
        raise RuntimeError("correlated Gaussian reference is absent")
    reference = reports["gaussian_correlated_g1"]
    evaluated = {}
    passing = []
    for name, report in reports.items():
        if name == "gaussian_correlated_g1":
            continue
        passed, reasons = candidate_gate(report, reference, thresholds=thresholds)
        evaluated[name] = {"pass": passed, "reasons": reasons}
        if passed:
            passing.append(name)
    if not passing:
        return {
            "schema_version": "p12f-no-field-finalist-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "evaluated": evaluated,
            "field_finalist": None,
            "truth_files_read": [],
            "open_count": 0,
            "ph001_opened": False,
            "pass": True,
        }
    passing.sort(
        key=lambda name: (
            float(reports[name]["proper_scores"]["primary_joint"]),
            float(reports[name].get("seconds_per_draw", np.inf)),
        )
    )
    winner = passing[0]
    return {
        "schema_version": "p12f-method-selection-frozen-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluated": evaluated,
        "field_finalist": winner,
        "selection_rule": "lowest primary joint proper score; simpler/faster tie break",
        "local_patch_posterior_only": True,
        "full_cap_coherence_established": False,
        "truth_files_read": [],
        "open_count": 0,
        "ph001_opened": False,
        "pass": True,
    }


@dataclass(frozen=True)
class ConditionedView:
    delta_r7: np.ndarray
    observation: np.ndarray
    response: np.ndarray
    support: np.ndarray
    core_id: int
    view: str


def choose_factorial_view(
    views: list[ConditionedView], *, seed: int, update: int
) -> ConditionedView:
    """Choose one view/core/update so repeated views never duplicate core weight."""
    if not views:
        raise ValueError("at least one conditioned view is required")
    core_ids = {item.core_id for item in views}
    if len(core_ids) != 1:
        raise ValueError("factorial views must share one latent core")
    shape = views[0].delta_r7.shape
    for item in views:
        if item.delta_r7.shape != shape or item.support.shape != shape:
            raise ValueError("factorial view geometry mismatch")
        if not np.array_equal(item.delta_r7, views[0].delta_r7):
            raise ValueError("factorial views do not share privileged delta_R7")
    index = np.random.default_rng(seed + 104729 * update).integers(0, len(views))
    return views[int(index)]
