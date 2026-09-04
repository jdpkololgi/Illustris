#!/usr/bin/env python3
"""Frozen FMPE-versus-Gaussian log-score comparison on opened ph001 audit rows."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.sbi.p12_prepare_base_response_dataset import softplus_coordinates
from workflows.sbi.p12_train_base_response_fmpe import paired_posterior_log_prob
from workflows.sbi.p12a_blind_inference import reconstruct_fmpe
from workflows.sbi.p12a_evaluate_blind import (
    load_prediction_arrays,
    validate_open_state,
)


SCHEMA = "p12a-ph001-proper-score-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def align_positions(reference: np.ndarray, requested: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.int64)
    requested = np.asarray(requested, dtype=np.int64)
    order = np.argsort(reference)
    position = np.searchsorted(reference[order], requested)
    if np.any(position >= len(reference)) or not np.array_equal(reference[order][position], requested):
        raise RuntimeError("requested parent IDs are absent from the frozen reference")
    return order[position]


def gaussian_physical_log_prob(
    truth: np.ndarray,
    base: np.ndarray,
    shell: np.ndarray,
    cap: np.ndarray,
    gaussian: dict,
) -> np.ndarray:
    truth = np.asarray(truth, dtype=np.float64)
    residual = truth - np.asarray(base, dtype=np.float64)
    result = np.full(len(truth), np.nan, dtype=np.float64)
    constant = 3.0 * np.log(2.0 * np.pi)
    for shell_value in range(4):
        for cap_value in (0, 1):
            chosen = (shell == shell_value) & (cap == cap_value)
            group = gaussian.get("groups", {}).get(f"shell{shell_value}_cap{cap_value}")
            if not np.any(chosen) or group is None:
                continue
            mean = np.asarray(group["mean"], dtype=np.float64)
            covariance = np.asarray(group["covariance"], dtype=np.float64)
            sign, logdet = np.linalg.slogdet(covariance)
            if sign <= 0:
                raise RuntimeError("Gaussian baseline covariance is not positive definite")
            centered = residual[chosen] - mean
            quadratic = np.einsum(
                "ni,ij,nj->n", centered, np.linalg.inv(covariance), centered
            )
            result[chosen] = -0.5 * (constant + logdet + quadratic)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("Gaussian baseline did not score every audit row")
    return result


def clustered_mean_interval(
    values: np.ndarray, groups: np.ndarray, *, repeats: int, seed: int
) -> dict:
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups, dtype=np.int64)
    unique, inverse = np.unique(groups, return_inverse=True)
    count = np.bincount(inverse)
    total = np.bincount(inverse, weights=values)
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        chosen = rng.integers(0, len(unique), size=len(unique))
        draws[index] = total[chosen].sum() / count[chosen].sum()
    return {
        "mean": float(values.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "bootstrap_repeats": int(repeats),
        "bootstrap_unit": "authoritative core",
        "spatial_blocks": int(len(unique)),
    }


def fmpe_physical_log_prob(
    posterior: object,
    checkpoint: dict,
    truth: np.ndarray,
    context: np.ndarray,
    *,
    device: str,
    chunk: int,
) -> np.ndarray:
    theta = softplus_coordinates(truth)
    theta_mean = np.asarray(checkpoint["theta_mean"], dtype=np.float64)
    theta_std = np.asarray(checkpoint["theta_std"], dtype=np.float64)
    context_mean = np.asarray(checkpoint["context_mean"], dtype=np.float64)
    context_std = np.asarray(checkpoint["context_std"], dtype=np.float64)
    scaled_theta = ((theta - theta_mean) / theta_std).astype(np.float32)
    scaled_context = ((context - context_mean) / context_std).astype(np.float32)
    pieces = []
    for start in range(0, len(truth), chunk):
        stop = min(start + chunk, len(truth))
        y = torch.as_tensor(scaled_theta[start:stop], dtype=torch.float32, device=device)
        x = torch.as_tensor(scaled_context[start:stop], dtype=torch.float32, device=device)
        with torch.inference_mode():
            value = paired_posterior_log_prob(posterior, y, x)
        pieces.append(np.asarray(value.detach().cpu(), dtype=np.float64).reshape(-1))
    log_q_scaled = np.concatenate(pieces)
    gaps = np.maximum(np.diff(np.asarray(truth, dtype=np.float64), axis=1), 1.0e-12)
    log_jacobian = (
        -float(np.log(theta_std).sum())
        - np.log1p(-np.exp(-gaps[:, 0]))
        - np.log1p(-np.exp(-gaps[:, 1]))
    )
    result = log_q_scaled + log_jacobian
    if not np.all(np.isfinite(result)):
        raise RuntimeError("FMPE physical log score is non-finite on ph001")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-predictions", type=Path, required=True)
    parser.add_argument("--opened-marker", type=Path, required=True)
    parser.add_argument("--evaluation-contract", type=Path, required=True)
    parser.add_argument("--truth-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite proper-score result: {args.output}")
    frozen, opened, contract, truth_manifest = validate_open_state(
        frozen_path=args.frozen_predictions,
        opened_path=args.opened_marker,
        contract_path=args.evaluation_contract,
        truth_manifest_path=args.truth_manifest,
    )
    summary, audit = load_prediction_arrays(frozen)
    truth_path = Path(truth_manifest["array"]["path"])
    with np.load(truth_path, mmap_mode="r") as truth_archive:
        truth_parent = np.asarray(truth_archive["parent_node_id"], dtype=np.int64)
        if not np.array_equal(truth_parent, summary["parent_node_id"]):
            raise RuntimeError("truth package differs from frozen posterior rows")
        audit_position = align_positions(truth_parent, audit["parent_node_id"])
        truth = np.asarray(truth_archive["eigenvalues"][audit_position], dtype=np.float64)
    manifests = _prediction_manifests(frozen)
    context_path = Path(manifests["p12a-blind-base-context-v1"]["array"])
    with np.load(context_path, mmap_mode="r") as context_archive:
        context_position = align_positions(context_archive["parent_node_id"], audit["parent_node_id"])
        context = np.asarray(context_archive["context"][context_position], dtype=np.float32)
        core = np.asarray(context_archive["core_id"][context_position], dtype=np.int64)
    candidate_path = Path(contract["candidate"]["path"])
    if sha256(candidate_path) != contract["candidate"]["sha256"]:
        raise RuntimeError("P12-A candidate changed after evaluation-contract freeze")
    candidate = json.loads(candidate_path.read_text())
    checkpoint_path = Path(candidate["artifacts"]["checkpoint"]["path"])
    if sha256(checkpoint_path) != candidate["artifacts"]["checkpoint"]["sha256"]:
        raise RuntimeError("P12-A checkpoint changed")
    posterior, checkpoint = reconstruct_fmpe(checkpoint_path, args.device)
    fmpe = fmpe_physical_log_prob(
        posterior, checkpoint, truth, context, device=args.device, chunk=args.chunk
    )
    gaussian_path = Path(contract["gaussian_baseline"]["path"])
    if sha256(gaussian_path) != contract["gaussian_baseline"]["sha256"]:
        raise RuntimeError("Gaussian baseline changed after evaluation-contract freeze")
    gaussian = json.loads(gaussian_path.read_text())
    gaussian_score = gaussian_physical_log_prob(
        truth,
        summary["base_prediction_eigenvalues"][audit_position],
        summary["shell"][audit_position],
        summary["cap"][audit_position],
        gaussian,
    )
    comparison = clustered_mean_interval(
        fmpe - gaussian_score, core, repeats=args.bootstrap, seed=args.seed
    )
    report = {
        "schema_version": SCHEMA,
        "created_utc": utc_now(),
        "git_revision": git_revision(),
        "phase": "ph001",
        "rows": int(len(truth)),
        "fmpe_mean_physical_log_prob": float(fmpe.mean()),
        "gaussian_mean_physical_log_prob": float(gaussian_score.mean()),
        **comparison,
        "higher_is_better": True,
        "frozen_predictions": {
            "path": str(args.frozen_predictions.resolve()),
            "sha256": sha256(args.frozen_predictions),
        },
        "opened_marker": {
            "path": str(args.opened_marker.resolve()),
            "sha256": sha256(args.opened_marker),
        },
        "evaluation_contract": {
            "path": str(args.evaluation_contract.resolve()),
            "sha256": sha256(args.evaluation_contract),
        },
        "truth_manifest": {
            "path": str(args.truth_manifest.resolve()),
            "sha256": sha256(args.truth_manifest),
        },
        "post_open_refit_performed": False,
        "post_open_tuning_allowed": False,
        "truth_files_read": opened["truth_files_read"],
        "open_count": 1,
        "sealed_phase_opened": True,
        "pass": bool(comparison["ci95"][0] > 0.0),
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


def _prediction_manifests(frozen: dict) -> dict[str, dict]:
    result = {}
    for record in frozen["prediction_manifests"]:
        path = Path(record["path"])
        if sha256(path) != record["sha256"]:
            raise RuntimeError("prediction manifest changed after blind freeze")
        payload = json.loads(path.read_text())
        result[payload["schema_version"]] = payload
    return result


if __name__ == "__main__":
    main()
