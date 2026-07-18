#!/usr/bin/env python3
"""Create the compact P1b canonical index over the full NGC+SGC path1 parent."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import fitsio
import numpy as np


SHELLS = ((0.15, 0.25), (0.25, 0.35), (0.35, 0.45), (0.45, 0.55))
SENTINEL = (0.585, 0.595)


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--points", type=Path, required=True)
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--parent-sha256", required=True)
    args = ap.parse_args()

    audit = json.loads(args.audit.read_text())
    if not audit.get("pass"):
        raise RuntimeError("P1b/P2b audit has not passed")
    columns = [
        "TARGETID", "Z", "FILE_NUM", "BOX_INDEX", "HALO_INDEX",
        "LAMBDA1", "LAMBDA2", "LAMBDA3",
    ]
    table = fitsio.read(str(args.parent), columns=columns)
    points = np.load(args.points, mmap_mode="r")
    n = len(table)
    if points.shape != (n, 4):
        raise RuntimeError(f"parent/points row mismatch: {n} versus {points.shape}")

    parent_node_id = np.arange(n, dtype=np.int64)
    cap = np.asarray(points[:, 3], dtype=np.uint8)
    z = np.asarray(table["Z"], dtype=np.float64)
    eig = np.column_stack([table["LAMBDA1"], table["LAMBDA2"], table["LAMBDA3"]])
    valid_target = (np.asarray(table["BOX_INDEX"]) >= 0) & np.isfinite(eig).all(axis=1)
    shell = np.full(n, -1, dtype=np.int8)
    for shell_id, (lo, hi) in enumerate(SHELLS):
        shell[(z >= lo) & (z < hi)] = shell_id
    active = valid_target & (shell >= 0)
    sentinel = (z >= SENTINEL[0]) & (z < SENTINEL[1])
    context = (z >= 0.10) & (z < 0.60) & ~sentinel

    if len(np.unique(table["TARGETID"])) != n:
        raise RuntimeError("TARGETID is not unique")
    if (cap > 1).any():
        raise RuntimeError("cap labels are not binary")
    if not active.any() or not (active & (cap == 0)).any() or not (active & (cap == 1)).any():
        raise RuntimeError("active sample does not cover both caps")
    if not np.array_equal(parent_node_id, np.arange(n, dtype=np.int64)):
        raise RuntimeError("parent node indexing is not identity")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.out_dir / "canonical_index.npz"
    np.savez_compressed(
        index_path,
        parent_node_id=parent_node_id,
        targetid=np.asarray(table["TARGETID"], dtype=np.int64),
        cap=cap,
        shell=shell,
        active=active,
        context=context,
        valid_target=valid_target,
    )
    index_sha = sha256(index_path)
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    shell_counts = {
        f"{lo:.2f}_{hi:.2f}": {
            "all": int((active & (shell == shell_id)).sum()),
            "NGC": int((active & (shell == shell_id) & (cap == 1)).sum()),
            "SGC": int((active & (shell == shell_id) & (cap == 0)).sum()),
        }
        for shell_id, (lo, hi) in enumerate(SHELLS)
    }
    manifest = {
        "schema_version": "1.0",
        "stage": "P1b",
        "catalogue_id": "ph000_path1_full_ngc_sgc_v1",
        "phase": "ph000",
        "observer": "path1_fiberassign",
        "parent": str(args.parent),
        "parent_sha256": args.parent_sha256,
        "parent_rows_are_canonical_rows": True,
        "index": str(index_path),
        "index_sha256": index_sha,
        "points": str(args.points),
        "audit": str(args.audit),
        "git_sha": git_sha,
        "scope": {
            "footprint": "full usable path1 BGS NGC+SGC",
            "components": {"0": "SGC", "1": "NGC"},
            "z_core": [0.15, 0.55],
            "z_context": [0.10, 0.60],
            "sentinel_excluded_from_context": list(SENTINEL),
        },
        "redshift_policy": {
            "geometry": "parent observed Z used by the pre-existing full graph",
            "additional_measurement_error": "none; do not perturb after graph construction",
            "reason": "preserve exact catalogue/graph/metric alignment",
        },
        "target_convention": {
            "labels": "tidal eigenvalues ascending, rs7=7 Mpc/h Gaussian, ngrid2048, halo_xcom",
            "epoch": "z=0.2 snapshot",
        },
        "counts": {
            "total": n,
            "NGC": int((cap == 1).sum()),
            "SGC": int((cap == 0).sum()),
            "context": int(context.sum()),
            "active": int(active.sum()),
            "valid_target": int(valid_target.sum()),
            "by_shell": shell_counts,
        },
        "mapping_contract": {
            "graph_node_id": "PARENT_NODE_ID == parent FITS row == full graph row",
            "galaxy_id": "TARGETID",
            "halo_group": ["FILE_NUM", "BOX_INDEX", "HALO_INDEX"],
            "p4_rule": "no repeated TARGETID or underlying halo group may cross supervised folds",
        },
        "no_train_fitted_normalisation": True,
        "no_split_filtering": True,
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    complete_path = args.out_dir / "CATALOGUE_COMPLETE"
    complete_path.write_text(
        f"P1b {manifest['catalogue_id']} rows={n} active={int(active.sum())} "
        f"index_sha256={index_sha}\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
