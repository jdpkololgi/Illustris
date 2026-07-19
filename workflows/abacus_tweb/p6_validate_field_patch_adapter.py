#!/usr/bin/env python3
"""Validate P6 channel, coordinate, interpolation, and context-view parity."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np

from p6_field_patch_utils import (
    CAP_NAME,
    CanonicalFieldPatchAdapter,
    sample_patch,
    trilinear_sample,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter-root", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p6_unet_patch_adapter"))
    ap.add_argument("--halo-small", type=int, default=4)
    ap.add_argument("--halo-large", type=int, default=8)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    manifest_path = args.adapter_root / "adapter_manifest.json"
    records = []
    with CanonicalFieldPatchAdapter(args.adapter_root) as adapter:
        counts = np.diff(np.asarray(adapter.core_offsets))
        chosen = []
        for cap in (0, 1):
            for fold in range(5):
                candidates = np.flatnonzero(
                    (adapter.core_cap == cap) & (adapter.core_fold == fold) & (counts > 0)
                )
                if not len(candidates):
                    raise RuntimeError(f"no non-empty core for cap={cap}, fold={fold}")
                chosen.append(int(candidates[np.argmax(counts[candidates])]))

        for core_id in chosen:
            small = adapter.extract(core_id, args.halo_small)
            large = adapter.extract(core_id, args.halo_large)
            handle = adapter._handle(small.cap)
            selection = tuple(
                slice(int(left), int(right))
                for left, right in zip(small.context_start, small.context_stop)
            )
            identity = all(
                np.array_equal(
                    small.values[index],
                    np.asarray(handle[name][selection], dtype=np.float32),
                )
                for index, name in enumerate(small.channel_names)
            )
            sample_small = sample_patch(small)
            sample_large = sample_patch(large)
            context_diff = (
                float(np.max(np.abs(sample_small - sample_large)))
                if len(sample_small) else 0.0
            )
            inside_nominal_core = np.all(
                (small.authoritative_frac_index_global >= small.core_start)
                & (small.authoritative_frac_index_global < small.core_stop),
                axis=1,
            )
            inside_context = np.all(
                (small.authoritative_frac_index_global >= small.context_start)
                & (small.authoritative_frac_index_global <= small.context_stop - 1),
                axis=1,
            )
            local_large = (
                small.authoritative_frac_index_global - large.context_start[None, :]
            )
            direct_large = trilinear_sample(large.values, local_large)
            coordinate_diff = (
                float(np.max(np.abs(direct_large - sample_large)))
                if len(sample_large) else 0.0
            )
            order = np.arange(len(sample_small))[::-1]
            order_diff = (
                float(np.max(np.abs(
                    trilinear_sample(
                        small.values,
                        small.authoritative_frac_index_local[order],
                    )[::-1] - sample_small
                ))) if len(order) else 0.0
            )
            unsupported_fraction = float(np.mean(small.unsupported_mask))
            records.append({
                "core_id": core_id,
                "cap": CAP_NAME[small.cap],
                "fold": small.fold,
                "authoritative_galaxies": len(small.authoritative_parent_id),
                "context_shape_small": list(small.values.shape[1:]),
                "context_shape_large": list(large.values.shape[1:]),
                "channel_identity": bool(identity),
                "nominal_voxel_core_fraction": float(np.mean(inside_nominal_core)),
                "all_authoritative_have_context_interpolation_support": bool(
                    np.all(inside_context)),
                "context_growth_interpolation_max_abs": context_diff,
                "coordinate_reexpression_max_abs": coordinate_diff,
                "galaxy_order_max_abs": order_diff,
                "unsupported_context_fraction": unsupported_fraction,
                "available_halo_low": small.available_halo_low.tolist(),
                "available_halo_high": small.available_halo_high.tolist(),
            })

    gates = {
        "canonical_channel_identity": all(row["channel_identity"] for row in records),
        "authoritative_coordinates_inside_context": all(
            row["all_authoritative_have_context_interpolation_support"] for row in records),
        "interpolation_invariant_to_context_growth": all(
            row["context_growth_interpolation_max_abs"] == 0.0 for row in records),
        "global_to_local_coordinate_parity": all(
            row["coordinate_reexpression_max_abs"] == 0.0 for row in records),
        "galaxy_order_parity": all(row["galaxy_order_max_abs"] == 0.0 for row in records),
        "both_caps_all_folds_represented": {
            (row["cap"], row["fold"]) for row in records
        } == {(cap, fold) for cap in ("NGC", "SGC") for fold in range(5)},
    }
    manifest = json.loads(manifest_path.read_text())
    report = {
        "schema_version": 1,
        "stage": "P6 structural field-patch parity",
        "adapter_manifest": str(manifest_path),
        "adapter_manifest_sha256": sha256(manifest_path),
        "halo_small_voxels": args.halo_small,
        "halo_large_voxels": args.halo_large,
        "cell_mpc": {
            name: cap["cell_mpc"] for name, cap in manifest["caps"].items()
        },
        "records": records,
        "gates": gates,
        "pass": all(gates.values()),
        "model_level_pending": [
            "fit and freeze per-rotation normalization on all training-core voxels",
            "refit or validate cap-specific selection channels",
            "run frozen U-Net prediction convergence versus context and boundary",
        ],
        "elapsed_seconds": time.time() - started,
    }
    report_path = args.adapter_root / "structural_parity_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["pass"]:
        raise RuntimeError(f"P6 structural parity failed: {gates}")
    (args.adapter_root / "FIELD_PATCH_PARITY_READY").write_text(
        f"adapter_manifest_sha256={sha256(manifest_path)}\n"
        f"structural_parity_report_sha256={sha256(report_path)}\n"
        "unet_patch_ready=false\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
