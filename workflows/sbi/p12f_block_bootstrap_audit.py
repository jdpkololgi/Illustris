#!/usr/bin/env python3
"""Patch-block bootstrap for a frozen P12-F posterior-sample artifact."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.sbi.p12f_field_posterior_diagnostics import quantile_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42017)
    return parser.parse_args()


def central_hits(samples: np.ndarray, truth: np.ndarray, nominal: float = 0.68) -> np.ndarray:
    alpha = (1.0 - nominal) / 2.0
    lower, upper = np.quantile(samples, (alpha, 1.0 - alpha), axis=0)
    return ((truth >= lower) & (truth <= upper)).astype(np.int64)


def blocked_fraction_interval(
    hits: np.ndarray,
    block_ids: np.ndarray,
    *,
    strata: np.ndarray | None,
    replicates: int,
    seed: int,
) -> dict:
    hits = np.asarray(hits, dtype=np.int64)
    block_ids = np.asarray(block_ids, dtype=np.int64)
    if strata is None:
        strata = np.zeros(len(hits), dtype=np.int64)
    else:
        strata = np.asarray(strata)
    blocks = np.unique(block_ids)
    labels = np.unique(strata)
    numerators = np.zeros((len(blocks), len(labels)), dtype=np.int64)
    denominators = np.zeros_like(numerators)
    for block_index, block in enumerate(blocks):
        block_mask = block_ids == block
        for label_index, label in enumerate(labels):
            mask = block_mask & (strata == label)
            denominators[block_index, label_index] = int(mask.sum())
            numerators[block_index, label_index] = int(hits[mask].sum())
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(blocks), size=(replicates, len(blocks)))
    output = {}
    for label_index, label in enumerate(labels):
        num = numerators[:, label_index]
        den = denominators[:, label_index]
        estimate = float(num.sum() / den.sum())
        draw_num = num[sampled].sum(axis=1)
        draw_den = den[sampled].sum(axis=1)
        valid = draw_den > 0
        fractions = draw_num[valid] / draw_den[valid]
        output[str(label)] = {
            "n_rows": int(den.sum()),
            "n_blocks": int(np.sum(den > 0)),
            "empirical_coverage": estimate,
            "absolute_error_from_0.68": abs(estimate - 0.68),
            "block_bootstrap_95_interval": [
                float(np.quantile(fractions, 0.025)),
                float(np.quantile(fractions, 0.975)),
            ],
        }
    return output


def main() -> None:
    args = parse_args()
    with np.load(args.samples, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    required = {
        "voxel_samples",
        "voxel_truth",
        "patch_core_id",
        "redshift_shell",
        "angular_response",
        "boundary_distance_mpc",
        "true_environment_delta_r7",
        "lambda_samples",
        "lambda_truth",
    }
    if not required.issubset(arrays):
        raise RuntimeError(f"missing arrays: {sorted(required - set(arrays))}")
    if "ph001" in str(args.samples):
        raise PermissionError("ph001 artifact rejected")
    block = arrays["patch_core_id"]
    voxel_hits = central_hits(arrays["voxel_samples"], arrays["voxel_truth"])
    strata = {
        "overall": None,
        "redshift_shell": arrays["redshift_shell"],
        "angular_response_quartile": quantile_labels(arrays["angular_response"]),
        "boundary_distance_quartile": quantile_labels(
            arrays["boundary_distance_mpc"]
        ),
        "true_environment_quartile": quantile_labels(
            arrays["true_environment_delta_r7"]
        ),
    }
    report = {
        "schema_version": "p12f-patch-block-bootstrap-coverage-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_samples": str(args.samples.resolve()),
        "source_samples_sha256": sha256(args.samples),
        "nominal_coverage": 0.68,
        "bootstrap_unit": "ph006 authoritative patch core",
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "unique_blocks": int(len(np.unique(block))),
        "voxel": {},
        "derived_local_tidal_eigenvalues": {},
        "ph001_opened": False,
    }
    for offset, (name, labels) in enumerate(strata.items()):
        report["voxel"][name] = blocked_fraction_interval(
            voxel_hits,
            block,
            strata=labels,
            replicates=args.bootstrap_replicates,
            seed=args.seed + offset,
        )
    for index in range(3):
        hits = central_hits(
            arrays["lambda_samples"][..., index], arrays["lambda_truth"][..., index]
        )
        report["derived_local_tidal_eigenvalues"][f"lambda{index + 1}"] = (
            blocked_fraction_interval(
                hits,
                block,
                strata=None,
                replicates=args.bootstrap_replicates,
                seed=args.seed + 20 + index,
            )["0"]
        )
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
