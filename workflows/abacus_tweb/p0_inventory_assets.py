#!/usr/bin/env python3
"""Inventory current GraphWeb mock, target, graph, field, and model assets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PHASE_RE = re.compile(r"AbacusSummit_base_c000_(ph\d{3})")


def fingerprint(path: Path, full_limit: int = 256 << 20, chunk: int = 4 << 20) -> dict:
    stat = path.stat()
    if path.is_dir():
        return {"path": str(path), "exists": True, "kind": "directory",
                "size_bytes": int(stat.st_size),
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "fingerprint_mode": "directory_stat_only", "fingerprint": None}
    h = hashlib.sha256()
    if stat.st_size <= full_limit:
        mode = "sha256_full"
        with path.open("rb") as f:
            while data := f.read(chunk):
                h.update(data)
    else:
        mode = "sha256_head_tail_plus_size"
        with path.open("rb") as f:
            h.update(f.read(chunk))
            f.seek(max(0, stat.st_size - chunk))
            h.update(f.read(chunk))
        h.update(str(stat.st_size).encode())
    return {"path": str(path), "exists": True, "kind": "directory" if path.is_dir() else "file",
            "size_bytes": int(stat.st_size), "mtime_utc": datetime.fromtimestamp(
                stat.st_mtime, timezone.utc).isoformat(), "fingerprint_mode": mode,
            "fingerprint": h.hexdigest() if path.is_file() else None}


def record(path: Path) -> dict:
    return fingerprint(path) if path.exists() else {"path": str(path), "exists": False}


def glob_records(pattern: str) -> list[dict]:
    from glob import glob
    return [record(Path(p)) for p in sorted(glob(pattern))]


def canonical_summary(cache_path: Path) -> dict:
    with cache_path.open("rb") as f:
        cache = pickle.load(f)
    eig = np.asarray(cache["eigenvalues_raw"])
    masks = tuple(np.asarray(m, bool) for m in cache["masks"])
    tid = np.asarray(cache["tid"], np.int64)
    return {"n_rows": int(len(eig)), "target_shape": list(eig.shape),
            "finite_targets": bool(np.isfinite(eig[np.logical_or.reduce(masks)]).all()),
            "split_counts": {name: int(mask.sum()) for name, mask in zip(
                ("train", "validation", "test"), masks)},
            "targetid_unique": bool(len(np.unique(tid)) == len(tid)),
            "shell_counts": {str(k): int(v) for k, v in zip(*np.unique(cache["shell"], return_counts=True))},
            "provenance": cache.get("provenance", {})}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("/pscratch/sd/d/dkololgi/abacus"))
    ap.add_argument("--canonical-cache", type=Path, required=True)
    ap.add_argument("--canonical-points", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cutsky = Path("/global/cfs/cdirs/desi/cosmosim/SecondGenMocks/AbacusSummit/CutSky/BGS/v0.1/z0.200")
    halo_root = Path("/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit/AbacusSummit_base_c000_ph000/halos/z0.200")
    density = Path("/pscratch/sd/d/dkololgi/AbacusSummit_densities/density_fields/AbacusSummit_base_c000_ph000_z0.200_ngrid_2048_10pc_density_field.npy")
    tweb = Path("/pscratch/sd/d/dkololgi/AbacusSummit_densities/tweb_rank_outputs_fullgrid_v3/dens_AbacusSummit_base_c000_ph000_z0.200_ngrid2048_box2000_thr0p2/backend_optimized_ngrid_2048_rsmooth_7")
    tiled = args.root / "sbi_caches/s3b_tiled_valid_v2"

    groups = {
        "source_cutsky_catalogues": glob_records(str(cutsky / "*ph*.fits")),
        "labelled_mock_catalogues": glob_records(str(args.root / "mocks_with_eigs*/*.fits")),
        "full_range_shell_catalogues": glob_records(str(args.root / "s2_shells/*.fits")),
        "canonical_row_index": [record(args.canonical_cache), record(args.canonical_points)],
        "tiled_graph_cache": glob_records(str(tiled / "manifest.json")) + glob_records(str(tiled / "tile_*.pkl")),
        "existing_graph_artifacts": glob_records(str(args.root / "graph_constructions/wedges/path1_fiberassign/*")),
        "density_and_tweb": [record(density), record(tweb)],
        "halo_and_particle_sources": [record(halo_root), record(halo_root / "halo_info")],
        "frozen_models": glob_records(str(args.root / "A1_sqrt/sbi_output/*bestL1*.pkl"))
                         + glob_records(str(args.root / "R0_valid_corrected/sbi_output/*bestL1*.pkl"))
                         + glob_records(str(args.root / "C_unet_fullrange/scores.pred_eigs.npy")),
        "classical_predictions": glob_records(str(args.root / "classical_baseline/fullrange_holdout/*")),
    }
    all_paths = [x["path"] for rows in groups.values() for x in rows]
    phases = sorted({m.group(1) for p in all_paths if (m := PHASE_RE.search(p))})
    source_by_phase = {
        phase: [x["path"] for x in groups["source_cutsky_catalogues"] if phase in x["path"]]
        for phase in phases
    }
    labelled_by_phase = {
        phase: [x["path"] for x in groups["labelled_mock_catalogues"] if phase in x["path"]]
        for phase in phases
    }
    phase_inventory = {
        phase: {
            "source_cutsky_catalogues": source_by_phase[phase],
            "hod_ids": sorted({
                (m.group(1) if (m := re.search(r"_hod_sample_(\d+)", path)) else "default")
                for path in source_by_phase[phase]
            }),
            "labelled_tweb_catalogues": labelled_by_phase[phase],
            "density_field_ready": phase == "ph000" and density.exists(),
            "tweb_grid_ready": phase == "ph000" and tweb.exists(),
            "canonical_graph_ready": phase == "ph000" and (tiled / "manifest.json").exists(),
        } for phase in phases
    }
    canonical = canonical_summary(args.canonical_cache)
    point_shape = list(np.load(args.canonical_points, mmap_mode="r").shape)
    required = {
        "canonical_cache_exists": args.canonical_cache.exists(),
        "canonical_points_exist": args.canonical_points.exists(),
        "canonical_points_shape_matches": point_shape == [canonical["n_rows"], 3],
        "active_targets_finite": canonical["finite_targets"],
        "targetids_unique": canonical["targetid_unique"],
        "ph000_density_exists": density.exists(),
        "ph000_tweb_exists": tweb.exists(),
        "tiled_manifest_exists": (tiled / "manifest.json").exists(),
        "independent_phase_labelled_catalogue_exists": any(
            phase != "ph000" and labelled_by_phase[phase] for phase in phases),
    }
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_convention": {
            "simulation": "AbacusSummit_base_c000_ph000",
            "hod_catalogue_family": "SecondGenMocks CutSky BGS v0.1",
            "observer_id": "default CutSky observer; not encoded separately in current filenames",
            "target_epoch_z": 0.2, "smoothing_mpc_h": 7.0, "threshold": 0.2,
            "eigenvalue_order": "lambda1<=lambda2<=lambda3",
            "input_coordinates": {"RA": "degrees", "DEC": "degrees", "Z": "observed redshift with RSD",
                                  "XYZ": "observer-frame comoving Mpc using Planck18 in s3c"},
            "target_coordinates": "real-space periodic Abacus box, host-halo index joined to CutSky",
        },
        "discovered_phases": phases,
        "phase_inventory": phase_inventory,
        "canonical_catalogue": canonical | {"points_shape": point_shape},
        "mapping_contract": {"galaxy_id": "TARGETID", "host_halo": ["FILE_NUM", "BOX_INDEX", "HALO_INDEX"],
                             "invalid_label_rule": "BOX_INDEX<0 excluded from every active mask",
                             "current_cache_limitation": "tile_v2 does not persist TARGETID/RA/halo per node; P1 must"},
        "assets": groups,
        "readiness": required,
        "gate": {"inventory_complete": True,
                 "p0_current_phase_assets_ready": bool(all(v for k, v in required.items()
                                                            if k != "independent_phase_labelled_catalogue_exists")),
                 "multi_phase_blind_test_ready": required["independent_phase_labelled_catalogue_exists"],
                 "blocking_findings": (["no labelled independent phase discovered"]
                                       if not required["independent_phase_labelled_catalogue_exists"] else [])},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    print(json.dumps(payload["gate"], indent=2))


if __name__ == "__main__":
    main()
