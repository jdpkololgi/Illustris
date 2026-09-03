#!/usr/bin/env python3
"""Audit the five low-k traceless-tidal components in P12-F3-L2 archives.

This is deliberately distinct from checking scalar density power.  A sampler
can reproduce P(k) while assigning power to the wrong Fourier directions, and
therefore produce the wrong tidal shear and eigengap dependence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from workflows.abacus_tweb.p6_field_patch_utils import trilinear_sample
from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.sbi.p12f3_export_hybrid_archive import lowpass_numpy
from workflows.sbi.p12f_common_evaluator import load_core_record
from workflows.sbi.p12f_dependency_rescue_evaluator import tarp_curve
from workflows.sbi.p12f_field_posterior_diagnostics import (
    fixed_tidal_tensor,
    scalar_posterior_report,
)


COMPONENTS = ("Sxx", "Syy", "Sxy", "Sxz", "Syz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--draw-batch", type=int, default=8)
    parser.add_argument("--maximum-k", type=float, default=0.1813799364234218)
    return parser.parse_args()


def traceless_components(tensor: torch.Tensor) -> torch.Tensor:
    """Return [Sxx,Syy,Sxy,Sxz,Syz] from a symmetric 3x3 tensor field."""
    if tensor.shape[-2:] != (3, 3):
        raise ValueError("tensor must end in a 3x3 matrix")
    trace_third = torch.diagonal(tensor, dim1=-2, dim2=-1).sum(dim=-1) / 3.0
    return torch.stack(
        (
            tensor[..., 0, 0] - trace_third,
            tensor[..., 1, 1] - trace_third,
            tensor[..., 0, 1],
            tensor[..., 0, 2],
            tensor[..., 1, 2],
        ),
        dim=-4,
    )


def sample_shear_at_galaxies(
    fields: np.ndarray,
    coordinates: np.ndarray,
    *,
    device: str,
    batch: int,
) -> np.ndarray:
    values = np.asarray(fields, dtype=np.float32)
    if values.ndim == 3:
        values = values[None]
    pieces = []
    for start in range(0, len(values), batch):
        field = torch.from_numpy(values[start : start + batch]).to(device)
        shear = traceless_components(fixed_tidal_tensor(field))
        # [draw,5,x,y,z] -> channels understood by trilinear_sample.
        channels = shear.reshape(-1, *shear.shape[-3:])
        sampled = trilinear_sample(channels, coordinates)
        sampled = sampled.reshape(len(coordinates), len(field), 5).permute(1, 0, 2)
        pieces.append(sampled.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(pieces, axis=0)


def covariance_report(draws: np.ndarray, truth: np.ndarray) -> dict:
    draw_centered = draws - draws.mean(axis=0, keepdims=True)
    truth_centered = truth - truth.mean(axis=0, keepdims=True)
    posterior = np.einsum("dnc,dne->ce", draw_centered, draw_centered)
    posterior /= max(draw_centered.shape[0] * draw_centered.shape[1] - 1, 1)
    target = truth_centered.T @ truth_centered / max(len(truth_centered) - 1, 1)
    difference = posterior - target
    return {
        "posterior": posterior.tolist(),
        "truth_innovation": target.tolist(),
        "relative_frobenius_error": float(
            np.linalg.norm(difference) / max(np.linalg.norm(target), 1e-12)
        ),
        "component_std_ratio": np.sqrt(
            np.divide(
                np.diag(posterior),
                np.diag(target),
                out=np.full(5, np.nan),
                where=np.diag(target) > 0,
            )
        ).tolist(),
    }


def audit_archive(
    manifest_path: Path,
    *,
    device: str,
    draw_batch: int,
    maximum_k: float,
) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != "p12f-sample-archive-v1"
        or manifest.get("phase") != "ph006"
        or manifest.get("ph001_opened")
        or manifest.get("truth_files_read") not in (["ph006"], ["ph006 density/T-web"])
    ):
        raise RuntimeError("unsafe P12-F3-L2 shear input")
    all_draws, all_truth, core_ids = [], [], []
    for ordinal, entry in enumerate(manifest["entries"]):
        path = Path(entry["path"])
        if sha256(path) != entry["sha256"]:
            raise RuntimeError("P12-F3-L2 shear archive changed")
        record = load_core_record(entry, int(manifest["draws"]))
        coordinates = np.asarray(record["galaxy_frac_index_local"], dtype=np.float64)
        if not len(coordinates):
            continue
        samples = np.asarray(record["delta_samples"], dtype=np.float32)
        truth = np.asarray(record["delta_truth"], dtype=np.float32)
        posterior_mean = samples.mean(axis=0)
        residual = samples - posterior_mean[None]
        innovation = truth - posterior_mean
        low_draws = lowpass_numpy(
            residual, voxel_mpc_h=5.0, maximum_k=maximum_k
        ).astype(np.float32)
        low_truth = lowpass_numpy(
            innovation[None], voxel_mpc_h=5.0, maximum_k=maximum_k
        )[0].astype(np.float32)
        draw_shear = sample_shear_at_galaxies(
            low_draws, coordinates, device=device, batch=draw_batch
        )
        truth_shear = sample_shear_at_galaxies(
            low_truth, coordinates, device=device, batch=1
        )[0]
        all_draws.append(draw_shear)
        all_truth.append(truth_shear)
        core_ids.append(np.full(len(coordinates), int(entry["core_id"]), dtype=np.int64))
        print(
            json.dumps(
                {
                    "method": manifest["method"],
                    "core": ordinal + 1,
                    "total": len(manifest["entries"]),
                }
            ),
            flush=True,
        )
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    draws = np.concatenate(all_draws, axis=1)
    truth = np.concatenate(all_truth, axis=0)
    groups = np.concatenate(core_ids)
    marginal = {
        name: scalar_posterior_report(draws[..., index], truth[..., index], seed=71 + index)
        for index, name in enumerate(COMPONENTS)
    }
    joint = tarp_curve(draws, truth, seed=79)
    maximum_coverage_error = max(
        float(row["coverage"][level]["absolute_error"])
        for row in marginal.values()
        for level in ("0.68", "0.90")
    )
    return {
        "schema_version": "p12f3l2-lowmode-shear-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": manifest["method"],
        "phase": "ph006",
        "cores": len(manifest["entries"]),
        "galaxies": int(len(truth)),
        "draws": int(draws.shape[0]),
        "components": list(COMPONENTS),
        "maximum_k_h_mpc": float(maximum_k),
        "marginal": marginal,
        "joint_tarp": joint,
        "maximum_marginal_coverage_error": maximum_coverage_error,
        "covariance": covariance_report(draws, truth),
        "resampling_note": "TARP visualization is pooled; uncertainty decisions remain patch-blocked",
        "core_id_sha256": __import__("hashlib").sha256(groups.tobytes()).hexdigest(),
        "archive_manifest": str(manifest_path.resolve()),
        "archive_manifest_sha256": sha256(manifest_path),
        "truth_files_read": ["ph006 density/T-web"],
        "ph001_opened": False,
    }


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("P12-F3-L2 shear audit requires CUDA")
    report = audit_archive(
        args.archive_manifest,
        device=args.device,
        draw_batch=args.draw_batch,
        maximum_k=args.maximum_k,
    )
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
