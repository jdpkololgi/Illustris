#!/usr/bin/env python3
"""Identify what the strongly compressed P11 latent direction represents.

This is an advisory post-canary diagnostic.  It fits weighted PC1 separately to
the frozen ph006 JEPA and supervised-control student latents, then correlates it
with physical eigenvalue combinations and the deployable response strength.
It does not select a model and it never opens ph001.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


ROOT = Path(
    "/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p11_factorial_views_v1/"
    "training/paired_degrade_jepa_v2/paired_degrade_jepa_m25_v2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        action="append",
        type=Path,
        default=[],
        help="Repeat for each frozen step-500 latent snapshot.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        result[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return result


def weighted_correlation(left: np.ndarray, right: np.ndarray, weight: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    weight = weight / weight.sum()
    left = left - np.sum(weight * left)
    right = right - np.sum(weight * right)
    denominator = np.sqrt(np.sum(weight * left * left) * np.sum(weight * right * right))
    return float(np.sum(weight * left * right) / max(denominator, 1e-30))


def pc1(values: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(values, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    normalized = weight / weight.sum()
    centered = values - np.sum(normalized[:, None] * values, axis=0)
    covariance = (centered * np.sqrt(normalized[:, None])).T @ (
        centered * np.sqrt(normalized[:, None])
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, -1]
    score = centered @ direction
    explained = float(eigenvalues[-1] / max(eigenvalues.sum(), 1e-30))
    return score, explained


def audit_snapshot(path: Path) -> dict:
    if "ph001" in str(path):
        raise PermissionError("P11 latent diagnostic forbids ph001")
    with np.load(path, allow_pickle=False) as data:
        latent = np.asarray(data["degraded_latent"], dtype=np.float64)
        target = np.asarray(data["target"], dtype=np.float64)
        response = np.asarray(data["response_strength"], dtype=np.float64)
        weight = np.asarray(data["sample_weight"], dtype=np.float64)
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
    if metadata.get("phase") != "ph006" or metadata.get("sealed_phase_opened"):
        raise RuntimeError("snapshot is not a sealed ph006 advisory probe")
    if target.shape[1] != 3 or not np.all(np.diff(target, axis=1) >= -1e-6):
        raise RuntimeError("snapshot target is not ordered physical eigenvalues")
    score, explained = pc1(latent, weight)
    quantities = {
        "lambda1": target[:, 0],
        "lambda2": target[:, 1],
        "lambda3": target[:, 2],
        "trace": target.sum(axis=1),
        "gap12": target[:, 1] - target[:, 0],
        "gap23": target[:, 2] - target[:, 1],
        "anisotropy_range": target[:, 2] - target[:, 0],
        "response_strength": response,
    }
    correlations = {}
    for name, values in quantities.items():
        correlations[name] = {
            "pearson": weighted_correlation(score, values, weight),
            "spearman": weighted_correlation(rankdata(score), rankdata(values), weight),
        }
    strongest = max(correlations, key=lambda name: abs(correlations[name]["spearman"]))
    return {
        "path": str(path.resolve()),
        "run_id": metadata.get("run_id"),
        "arm": metadata.get("arm"),
        "global_step": metadata.get("global_step"),
        "rows": int(len(score)),
        "latent_dimensions": int(latent.shape[1]),
        "pc1_weighted_variance_fraction": explained,
        "pc1_sign_is_arbitrary": True,
        "correlations": correlations,
        "strongest_absolute_spearman": strongest,
        "ph001_opened": False,
    }


def main() -> None:
    args = parse_args()
    snapshots = args.snapshot or [
        ROOT / "jepa/seed_42/latent_exports/step_000000500.npz",
        ROOT / "supervised_masked/seed_42/latent_exports/step_000000500.npz",
    ]
    report = {
        "schema_version": "p11-latent-physics-diagnostic-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": (
            "Advisory identification of the dominant trained latent direction; "
            "not a JEPA promotion gate and not posterior uncertainty."
        ),
        "snapshots": [audit_snapshot(path) for path in snapshots],
        "ph001_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
