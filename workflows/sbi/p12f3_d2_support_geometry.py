#!/usr/bin/env python3
"""Read-only D2 patch/support geometry and independent mask-construction audit."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import healpy as hp
import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from workflows.sbi.p12f3_d2_recovery_audit import sha


def edge_distance(points, low, high):
    """Signed distance to nearest axis-aligned cell face, in voxel units."""
    return np.minimum(points - np.asarray(low), np.asarray(high) - points).min(axis=1)


def summary(values):
    x = np.asarray(values, dtype=float)
    return dict(n=len(x), quantiles=dict(zip(
        ("min", "p05", "p25", "p50", "p75", "p95", "max"),
        np.quantile(x, [0, .05, .25, .5, .75, .95, 1]).tolist()))) if len(x) else dict(n=0)


def native_support(xyz, cap_id, support, domain, selection):
    radius = np.linalg.norm(xyz, axis=1)
    pixel = hp.vec2pix(256, *xyz.T, nest=False)
    z = np.interp(radius, selection["cosmology"]["radius_grid_mpc"],
                  selection["cosmology"]["redshift_grid"])
    angular = support[pixel] & (domain[pixel] // 2 == cap_id)
    radial = (z >= .10) & (z < .60) & ~((z >= .585) & (z < .595))
    return angular & radial, angular, radial, pixel, z


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("compute allocation required")
    if args.output.exists():
        raise FileExistsError("refuse to replace completed geometry audit")
    phase_root = args.root.parents[1]
    adapter = phase_root / "training_contract_r1_random/adapters/ph006/field"
    manifest_path = adapter / "adapter_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    selection_path = phase_root / "training_contract/transforms/field/selection_manifest.json"
    selection = json.loads(selection_path.read_text())
    decision_path = phase_root / "training_contract/P3BR_RANDOM_DENSITY_DECISION.json"
    decision = json.loads(decision_path.read_text())
    angular_path = phase_root / "ph006/p3b_random_response_v1/angular" / f'randoms_n{decision["selected_realisation_count"]}.npz'
    with np.load(angular_path, allow_pickle=False) as data:
        angular_support = data["support"].astype(bool)
        domain = data["domain"]
    arrays = {key: np.load(adapter / (key+".npy"), mmap_mode="r") for key in
              ("core_voxel_start", "core_active_offsets", "core_active_parent", "core_active_frac_index", "core_cap")}
    archive_path = args.root / "evaluation/seed42_v1/d2_modern_base4_nfe50/P12F_SAMPLE_ARCHIVE.json"
    archive = json.loads(archive_path.read_text())
    if archive["phase"] != "ph006" or not archive["pass"]:
        raise RuntimeError("invalid diagnostic archive")
    all_rows, bad_rows, mismatches = [], [], 0
    for n, entry in enumerate(archive["entries"]):
        if sha(entry["path"]) != entry["sha256"]:
            raise RuntimeError("archive changed")
        core = int(entry["core_id"])
        with np.load(entry["path"], allow_pickle=False) as data:
            mask = data["support"].astype(bool)
            coords = data["galaxy_frac_index_local"].astype(float)
            bounds = data["core_bounds"]
        start, stop = arrays["core_active_offsets"][core:core+2]
        parents = arrays["core_active_parent"][int(start):int(stop)]
        global_coords = arrays["core_active_frac_index"][int(start):int(stop)]
        context_start = arrays["core_voxel_start"][core] - bounds[0]
        if not np.array_equal((global_coords-context_start).astype(np.float32), coords.astype(np.float32)):
            raise RuntimeError("coordinate identity changed")
        nearest = np.rint(coords).astype(int)
        supported = mask[tuple(nearest.T)]
        cap_id = int(arrays["core_cap"][core])
        cap = "NGC" if cap_id == 1 else "SGC"
        geo = manifest["caps"][cap]
        cell = float(geo["cell_mpc"])
        xyz = np.asarray(geo["origin_mpc"]) + cell*(global_coords+.5)
        centre_xyz = np.asarray(geo["origin_mpc"]) + cell*(nearest+context_start+.5)
        native, angular, radial, pixel, z = native_support(xyz,cap_id,angular_support,domain,selection)
        centre, centre_angular, centre_radial, centre_pixel, _ = native_support(centre_xyz,cap_id,angular_support,domain,selection)
        mismatches += int(np.count_nonzero(supported != centre))
        context_edge = edge_distance(coords, -.5*np.ones(3), np.asarray(mask.shape)-.5)
        core_edge = edge_distance(coords, bounds[0]-.5, bounds[1]-.5)
        for i in range(len(coords)):
            all_rows.append([int(supported[i]), float(core_edge[i]), float(context_edge[i])])
        bad = np.flatnonzero(~supported)
        if len(bad):
            tree = cKDTree(np.argwhere(mask))
            distance, near_supported = tree.query(coords[bad])
            supported_voxels = tree.data[near_supported].astype(int)
            for j, i in enumerate(bad):
                bad_rows.append(dict(core=core, cap=cap, parent=int(parents[i]),
                    local_coordinate=coords[i].tolist(), xyz_mpc=xyz[i].tolist(),
                    core_edge_voxels=float(core_edge[i]), context_edge_voxels=float(context_edge[i]),
                    nearest_supported_centre_voxels=float(distance[j]),
                    nearest_supported_local_index=supported_voxels[j].tolist(),
                    cell_mpc=cell, galaxy_native_support=bool(native[i]),
                    galaxy_angular_support=bool(angular[i]), galaxy_radial_support=bool(radial[i]),
                    centre_angular_support=bool(centre_angular[i]), centre_radial_support=bool(centre_radial[i]),
                    galaxy_healpix=int(pixel[i]), centre_healpix=int(centre_pixel[i]), redshift=float(z[i])))
        if (n+1)%32 == 0:
            print(json.dumps(dict(cores=n+1, unsupported=len(bad_rows), centre_mismatches=mismatches)),flush=True)
    all_rows = np.asarray(all_rows)
    if len(bad_rows) != 736 or len(all_rows) != 133698:
        raise RuntimeError("audited panel counts changed")
    output = dict(schema_version="d2-support-geometry-v1", created_utc=datetime.now(timezone.utc).isoformat(),
        allocation=os.environ["SLURM_JOB_ID"], ph001_access=False, evaluation_amended=False,
        source_sha256=sha(__file__), archive_sha256=sha(archive_path),
        angular_map_sha256=sha(angular_path), selection_sha256=sha(selection_path),
        manifest_sha256=sha(manifest_path), centre_mask_mismatches=mismatches,
        galaxies=len(all_rows), unsupported=len(bad_rows),
        distance_units="voxel units; each voxel is 5 Mpc (not 5 Mpc/h)",
        unsupported_summary={key:summary([row[key] for row in bad_rows]) for key in
            ("core_edge_voxels","context_edge_voxels","nearest_supported_centre_voxels","redshift")},
        supported_summary={"core_edge_voxels":summary(all_rows[all_rows[:,0]==1,1]),
                           "context_edge_voxels":summary(all_rows[all_rows[:,0]==1,2])},
        unsupported_counts={
            "within_one_voxel_of_core_face":sum(row["core_edge_voxels"]<=1 for row in bad_rows),
            "within_one_voxel_of_context_face":sum(row["context_edge_voxels"]<=1 for row in bad_rows),
            "over_four_voxels_from_core_face":sum(row["core_edge_voxels"]>4 for row in bad_rows),
            "within_one_voxel_of_supported_centre":sum(row["nearest_supported_centre_voxels"]<=1 for row in bad_rows),
            "within_two_voxels_of_supported_centre":sum(row["nearest_supported_centre_voxels"]<=2 for row in bad_rows),
            "galaxy_native_supported_but_nearest_voxel_unsupported":sum(row["galaxy_native_support"] for row in bad_rows),
            "galaxy_native_angular_unsupported":sum(not row["galaxy_angular_support"] for row in bad_rows),
            "galaxy_native_radial_unsupported":sum(not row["galaxy_radial_support"] for row in bad_rows)},
        rows=bad_rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(output,stream,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps({k:v for k,v in output.items() if k!="rows"},indent=2),flush=True)


if __name__ == "__main__":
    main()
