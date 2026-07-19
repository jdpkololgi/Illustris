#!/usr/bin/env python3
"""Build the compact P6 field-patch index from immutable P3/P4 products."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from p6_field_patch_utils import CAP_NAME, fractional_cell_index


def sha256(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p3-manifest", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p3_full_footprint/field_manifest.json"))
    ap.add_argument("--p4-root", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest"))
    ap.add_argument("--output-root", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p6_unet_patch_adapter"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    args.output_root.mkdir(parents=True, exist_ok=True)
    p3 = json.loads(args.p3_manifest.read_text())
    cores_path = args.p4_root / "cores.npz"
    active_path = args.p4_root / "active_assignment.npz"
    with np.load(cores_path) as cores:
        core_start = np.asarray(cores["voxel_start"], dtype=np.int32)
        core_stop = np.asarray(cores["voxel_stop"], dtype=np.int32)
        core_fold = np.asarray(cores["fold"], dtype=np.int8)
        core_cap = np.asarray(cores["cap"], dtype=np.int8)
    n_core = len(core_fold)

    with np.load(active_path) as active:
        eligible = np.asarray(active["supervised_eligible"], dtype=bool)
        parent = np.asarray(active["parent_node_id"][eligible], dtype=np.int64)
        core_id = np.asarray(active["core_id"][eligible], dtype=np.int32)
        active_cap = np.asarray(active["cap"][eligible], dtype=np.int8)
    order = np.lexsort((parent, core_id))
    parent = parent[order]
    core_id = core_id[order]
    active_cap = active_cap[order]
    counts = np.bincount(core_id, minlength=n_core).astype(np.int64)
    offsets = np.empty(n_core + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])

    points_path = Path(p3["points"])
    points = np.load(points_path, mmap_mode="r")
    xyz = np.asarray(points[parent, :3], dtype=np.float64)
    frac = np.empty_like(xyz, dtype=np.float32)
    cap_records = {}
    selection_failures = []
    for cap in (0, 1):
        name = CAP_NAME[cap]
        component = p3["components"][name]
        grid = component["grid"]
        mask = active_cap == cap
        frac[mask] = fractional_cell_index(
            xyz[mask], np.asarray(grid["origin_mpc"]), float(grid["cell_mpc"])
        ).astype(np.float32)
        shell_ratios = {}
        for shell, record in component["support_atlas"].items():
            ratio = float(record["expected_count_sum"]) / float(record["input_galaxies"])
            shell_ratios[shell] = ratio
            if abs(ratio - 1.0) > 0.05:
                selection_failures.append(
                    {"cap": name, "shell": shell, "expected_to_input_ratio": ratio}
                )
        cap_records[name] = {
            "cap_id": cap,
            "field_path": component["file"],
            "field_sha256": component["file_sha256"],
            "shape": grid["shape"],
            "origin_mpc": grid["origin_mpc"],
            "cell_mpc": grid["cell_mpc"],
            "selection_expected_to_input_by_shell": shell_ratios,
        }

    arrays = {
        "core_voxel_start.npy": core_start,
        "core_voxel_stop.npy": core_stop,
        "core_fold.npy": core_fold,
        "core_cap.npy": core_cap,
        "core_active_offsets.npy": offsets,
        "core_active_parent.npy": parent,
        "core_active_frac_index.npy": frac,
    }
    for name, array in arrays.items():
        np.save(args.output_root / name, array)

    gates = {
        "all_authoritative_rows_indexed": int(offsets[-1]) == len(parent),
        "core_ids_in_range": bool(
            len(core_id) == 0 or (int(core_id.min()) >= 0 and int(core_id.max()) < n_core)
        ),
        "cap_identity_matches_core": bool(np.all(active_cap == core_cap[core_id])),
        "fractional_indices_finite": bool(np.isfinite(frac).all()),
        "two_cap_fields_present": set(cap_records) == {"NGC", "SGC"},
        "p3_unit_audit_passed": bool(p3["gates"]["unit_audit_pass"]),
    }
    manifest = {
        "schema_version": 1,
        "stage": "P6 canonical field-patch index",
        "git_sha": git_sha(),
        "p3_manifest": str(args.p3_manifest),
        "p3_manifest_sha256": sha256(args.p3_manifest),
        "p4_cores": str(cores_path),
        "p4_cores_sha256": sha256(cores_path),
        "p4_active_assignment": str(active_path),
        "p4_active_assignment_sha256": sha256(active_path),
        "points": str(points_path),
        "channel_order": p3["channel_order"],
        "axis_order": "ix,iy,iz",
        "n_cores": n_core,
        "n_authoritative": len(parent),
        "caps": cap_records,
        "normalization_contract": {
            "fit_population": "training-fold output-core voxels only",
            "application": "frozen per rotation; never patch local",
            "counts": "log1p then zscore",
            "expected_counts": "log1p then zscore",
            "ntilde_mpc3": "log then zscore",
            "log_count_ratio": "zscore",
            "exposure_binary": "identity",
            "exposure_apodized": "identity",
            "los_x": "identity",
            "los_y": "identity",
            "los_z": "identity",
        },
        "selection_channel_status": {
            "ready_for_u_patch_training": not selection_failures,
            "reason": (
                "P3 radial spline was frozen on a wedge and must be refit or "
                "validated per cap before expected-count channels are production inputs"
            ),
            "failures_over_5pct": selection_failures,
        },
        "gates": gates,
        "pass": all(gates.values()),
        "elapsed_seconds": time.time() - started,
    }
    manifest_path = args.output_root / "adapter_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if not manifest["pass"]:
        raise RuntimeError(f"P6 index gates failed: {gates}")
    (args.output_root / "FIELD_PATCH_INDEX_READY").write_text(
        f"adapter_manifest_sha256={sha256(manifest_path)}\n"
        f"authoritative_rows={len(parent)}\n"
        f"selection_channels_ready={not selection_failures}\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
