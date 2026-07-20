"""Shared deterministic epoch semantics for P8 patch recovery training.

The original P8 smoke trainers sampled patches with replacement for a fixed
number of steps.  That was suitable for plumbing, but it did not define an
epoch or guarantee exposure of every eligible core.  This module makes the
scientific objective and the resume cursor explicit without depending on a
particular encoder implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def epoch_order(
    core_ids: np.ndarray,
    *,
    seed: int,
    epoch: int,
    core_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Return one deterministic weighted, without-replacement epoch order.

    The exponential-race construction is probability-proportional-to-weight
    without replacement.  Every core still appears exactly once; weights only
    set its expected position within the epoch.  Scientific weighting remains
    explicit in :func:`patch_objective`.
    """
    core_ids = np.asarray(core_ids, dtype=np.int64)
    if core_ids.ndim != 1 or len(core_ids) == 0:
        raise ValueError("core_ids must be a non-empty one-dimensional array")
    if len(np.unique(core_ids)) != len(core_ids):
        raise ValueError("core_ids must be unique within an epoch")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch)]))
    if core_weight is None:
        return core_ids[rng.permutation(len(core_ids))]
    weight = np.asarray(core_weight, dtype=np.float64)
    if weight.shape != core_ids.shape:
        raise ValueError("core_weight must match core_ids")
    if not np.all(np.isfinite(weight)) or np.any(weight <= 0):
        raise ValueError("core_weight must be finite and strictly positive")
    priority = -np.log(np.maximum(rng.random(len(core_ids)), np.finfo(float).tiny)) / weight
    return core_ids[np.argsort(priority, kind="stable")]


def patch_objective(
    weighted_loss_numerator,
    *,
    mean_core_weight: float,
):
    """Scale one patch numerator so equal patch steps recover the global loss.

    If every core is visited once, the arithmetic mean of these returned
    objectives across the epoch equals the globally row-weighted mean loss:

        mean_p [sum_i(w_i l_i)_p / mean_p W_p]
        = sum_i w_i l_i / sum_i w_i.

    Keeping the numerator intact avoids giving every patch equal scientific
    weight merely because it is one optimizer step.
    """
    if not np.isfinite(mean_core_weight) or mean_core_weight <= 0:
        raise ValueError("mean_core_weight must be finite and positive")
    return weighted_loss_numerator / float(mean_core_weight)


@dataclass
class EpochLossAccumulator:
    """Exact weighted loss/exposure totals for one complete patch epoch."""

    weighted_numerator: float = 0.0
    weight_denominator: float = 0.0
    rows: int = 0
    patches: int = 0

    def add(self, loss_per_row: np.ndarray, row_weight: np.ndarray) -> None:
        loss = np.asarray(loss_per_row, dtype=np.float64)
        weight = np.asarray(row_weight, dtype=np.float64)
        if loss.shape != weight.shape:
            raise ValueError("loss_per_row and row_weight must have identical shape")
        if not np.all(np.isfinite(loss)) or not np.all(np.isfinite(weight)):
            raise ValueError("epoch loss inputs must be finite")
        if np.any(weight < 0):
            raise ValueError("row weights must be non-negative")
        self.weighted_numerator += float(np.sum(weight * loss))
        self.weight_denominator += float(np.sum(weight))
        self.rows += int(loss.size)
        self.patches += 1

    @property
    def mean(self) -> float:
        if self.weight_denominator <= 0:
            return float("nan")
        return self.weighted_numerator / self.weight_denominator

    def as_dict(self) -> dict:
        return {
            "weighted_numerator": self.weighted_numerator,
            "weight_denominator": self.weight_denominator,
            "weighted_mean": self.mean,
            "rows": self.rows,
            "patches": self.patches,
        }

    @classmethod
    def from_dict(cls, row: dict | None) -> "EpochLossAccumulator":
        if not row:
            return cls()
        return cls(
            weighted_numerator=float(row["weighted_numerator"]),
            weight_denominator=float(row["weight_denominator"]),
            rows=int(row["rows"]),
            patches=int(row["patches"]),
        )


def improved(score: float, best_score: float, min_delta: float) -> bool:
    """Registered P8 validation improvement predicate."""
    return bool(np.isfinite(score) and score >= best_score + min_delta)


def should_stop(*, epoch: int, stale_epochs: int, min_epochs: int, patience: int) -> bool:
    """Early stopping is forbidden before the registered minimum epoch."""
    return epoch >= min_epochs and stale_epochs >= patience


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def reconcile_loss_trace(path: Path, *, maximum_global_step: int) -> int:
    """Roll a trace back to a checkpoint and remove replay duplicates.

    Loss windows may be flushed more frequently than model checkpoints.  After
    an allocation dies, rows newer than the checkpoint describe abandoned
    updates.  Retaining only the last row for each saved step also repairs a
    trace if a recovery was itself interrupted before reconciliation existed.
    Returns the number of retained records.
    """
    if not path.exists():
        return 0
    retained: dict[int, dict] = {}
    for raw in path.read_text().splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        step = int(row["global_step"])
        if step <= int(maximum_global_step):
            retained[step] = row
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for step in sorted(retained):
            handle.write(json.dumps(retained[step], sort_keys=True) + "\n")
    temporary.replace(path)
    return len(retained)


def rewrite_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """Atomically make a JSONL file agree with checkpoint-owned records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_resume_order(
    order: np.ndarray,
    expected_cores: np.ndarray,
    cursor: int,
) -> None:
    """Reject a checkpoint whose epoch order/cursor cannot be resumed safely."""
    order = np.asarray(order, dtype=np.int64)
    expected = np.asarray(expected_cores, dtype=np.int64)
    if cursor < 0 or cursor > len(order):
        raise ValueError("resume cursor is outside the epoch order")
    if len(order) != len(expected) or not np.array_equal(
        np.sort(order), np.sort(expected)
    ):
        raise ValueError("resume epoch order is not a permutation of eligible cores")
