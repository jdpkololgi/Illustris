#!/usr/bin/env python3
"""Cache parent-node exposure values needed by the P8 covariate audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p6_field_patch_utils import fractional_cell_index
from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256


P8_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p8_deterministic_v1")
P6_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p6_unet_patch_adapter")
POINTS = Path(
    "/pscratch/sd/d/dkololgi/abacus/graph_constructions/"
    "path1_fiberassign_mock_bgs_maglim_rs7_points.npy"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p8-root", type=Path, default=P8_ROOT)
    parser.add_argument("--p6-root", type=Path, default=P6_ROOT)
    parser.add_argument("--points", type=Path, default=POINTS)
    args = parser.parse_args()
    manifest_path = args.p6_root / "adapter_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    points = np.load(args.points, mmap_mode="r")
    output_path = args.p8_root / "parent_exposure_apodized.npy"
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float32, shape=(len(points),)
    )
    records = []
    for cap_name in ("SGC", "NGC"):
        spec = manifest["caps"][cap_name]
        cap_id = int(spec["cap_id"])
        parent = np.flatnonzero(np.asarray(points[:, 3], dtype=np.int8) == cap_id)
        frac = fractional_cell_index(
            np.asarray(points[parent, :3], dtype=np.float64),
            np.asarray(spec["origin_mpc"], dtype=np.float64),
            float(spec["cell_mpc"]),
        )
        index = np.rint(frac).astype(np.int64)
        shape = np.asarray(spec["shape"], dtype=np.int64)
        index = np.clip(index, 0, shape - 1)
        with h5py.File(spec["field_path"], "r") as handle:
            # One sequential dataset read is substantially safer than millions of
            # random HDF5 accesses and fits comfortably on a Perlmutter CPU node.
            exposure = np.asarray(handle["exposure_apodized"], dtype=np.float32)
        output[parent] = exposure[index[:, 0], index[:, 1], index[:, 2]]
        records.append({
            "cap": cap_name,
            "parent_nodes": int(len(parent)),
            "minimum": float(output[parent].min()),
            "maximum": float(output[parent].max()),
            "mean": float(output[parent].mean()),
            "sampling": "nearest canonical P3 voxel centre",
        })
        del exposure, frac, index, parent
    output.flush()
    del output
    payload = {
        "schema_version": 1,
        "stage": "P8 parent observational covariate cache",
        "output": str(output_path),
        "output_sha256": sha256(output_path),
        "points": str(args.points),
        "points_sha256": sha256(args.points),
        "p6_adapter": str(manifest_path),
        "p6_adapter_sha256": sha256(manifest_path),
        "records": records,
    }
    atomic_json(args.p8_root / "parent_diagnostics_manifest.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
