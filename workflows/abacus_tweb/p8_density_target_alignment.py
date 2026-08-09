#!/usr/bin/env python3
"""Preflight observer-frame truth alignment for P8.9 U-DENSITY-PHYS-v1.

The density-first model requires a voxelwise true matter field on the canonical
observer-frame lattices.  Before generating those large fields, this audit asks
whether a simple sky-coordinate mapping back into the periodic ph000 cube
reproduces the already-frozen R=7 T-web labels.  It compares observed-redshift
and cosmological-redshift coordinates at the plausible observer origins against
an explicit host-halo x_com reference.

No learned model is run and no validation score is used for model selection.
Failure is a target-coordinate blocker: it means the cut-sky replication mapping
must be recovered explicitly before density supervision can begin.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from astropy.cosmology import Planck18
import fitsio
import numpy as np

from workflows.abacus_tweb.p8_deterministic_common import (
    atomic_json,
    authoritative_mask,
    sha256,
)
from workflows.abacus_tweb.validate_cutsky_eigs_boxindex_vs_halo_xcom import (
    assign_eigs_from_slabs,
    build_slab_maps,
    discover_slabs,
    load_halo_positions_xcom,
)


ROOT = Path("/pscratch/sd/d/dkololgi/abacus")
CATALOGUE = ROOT / (
    "mocks_with_eigs_05062026_rsmooth_7/"
    "mock_bgs_maglim_path1_fiberassign_graph_ready_with_tweb_eigs_rs7_"
    "ngrid2048_thr0p2_halo_xcom.fits"
)
TARGET_INPUT = ROOT / (
    "SecondGen_Mocks/ph000/stage_3/path1_fiberassign_20260604_083322/inputs/targ.fits"
)
ASSIGNMENT = ROOT / "p4_spatial_manifest/active_assignment.npz"
TWEB = Path(
    "/pscratch/sd/d/dkololgi/AbacusSummit_densities/"
    "tweb_rank_outputs_fullgrid_v3/"
    "dens_AbacusSummit_base_c000_ph000_z0.200_ngrid2048_box2000_thr0p2/"
    "backend_optimized_ngrid_2048_rsmooth_7"
)
HALO_INFO = Path(
    "/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit/"
    "AbacusSummit_base_c000_ph000/halos/z0.200/halo_info"
)
OUTPUT = ROOT / "p8_density_phys_v1/preflight/coordinate_alignment.json"
SHELLS = ("0p15_0p25", "0p25_0p35", "0p35_0p45", "0p45_0p55")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, default=CATALOGUE)
    parser.add_argument("--target-input", type=Path, default=TARGET_INPUT)
    parser.add_argument("--assignment", type=Path, default=ASSIGNMENT)
    parser.add_argument("--tweb-dir", type=Path, default=TWEB)
    parser.add_argument("--halo-info", type=Path, default=HALO_INFO)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--sample-per-cap-shell", type=int, default=2_000)
    parser.add_argument("--target-chunk", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--minimum-sky-r2", type=float, default=0.90)
    parser.add_argument("--minimum-host-r2", type=float, default=0.999)
    return parser.parse_args()


def read_rows(path: Path, rows: np.ndarray, columns: list[str]) -> np.ndarray:
    """Read arbitrary FITS rows while preserving caller order."""
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows, kind="mergesort")
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    with fitsio.FITS(str(path), "r") as handle:
        table = handle[1].read(rows=rows[order], columns=columns)
    return table[inverse]


def choose_sample(assignment: np.lib.npyio.NpzFile, per_stratum: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eligible = authoritative_mask(assignment) & np.asarray(
        assignment["supervised_eligible"], dtype=bool
    )
    cap = np.asarray(assignment["cap"], dtype=np.int8)
    shell = np.asarray(assignment["shell"], dtype=np.int8)
    selected: list[np.ndarray] = []
    for cap_id in (0, 1):
        for shell_id in range(4):
            rows = np.flatnonzero(eligible & (cap == cap_id) & (shell == shell_id))
            if len(rows) < per_stratum:
                raise RuntimeError(
                    f"cap={cap_id} shell={shell_id} has only {len(rows)} eligible rows"
                )
            selected.append(np.sort(rng.choice(rows, per_stratum, replace=False)))
    return np.concatenate(selected)


def join_target_truth(
    path: Path,
    targetid: np.ndarray,
    *,
    chunk_rows: int,
) -> dict[str, np.ndarray]:
    """Recover Z_COSMO/RSDZ for sampled TARGETIDs from the immutable target table."""
    targetid = np.asarray(targetid, dtype=np.int64)
    if len(np.unique(targetid)) != len(targetid):
        raise RuntimeError("sampled Bright TARGETIDs are not unique")
    order = np.argsort(targetid)
    sorted_id = targetid[order]
    columns = ("Z_COSMO", "RSDZ", "RA", "DEC")
    output = {name: np.full(len(targetid), np.nan, dtype=np.float64) for name in columns}
    matches = np.zeros(len(targetid), dtype=np.int8)
    with fitsio.FITS(str(path), "r") as handle:
        hdu = handle[1]
        nrows = int(hdu.get_nrows())
        for start in range(0, nrows, chunk_rows):
            stop = min(start + chunk_rows, nrows)
            block = hdu[start:stop][["TARGETID", *columns]]
            block_id = np.asarray(block["TARGETID"], dtype=np.int64)
            position = np.searchsorted(sorted_id, block_id)
            valid = position < len(sorted_id)
            valid &= sorted_id[np.minimum(position, len(sorted_id) - 1)] == block_id
            if not np.any(valid):
                continue
            destination = order[position[valid]]
            if np.any(matches[destination]):
                raise RuntimeError("TARGETID appears more than once in immutable target input")
            matches[destination] = 1
            for name in columns:
                output[name][destination] = np.asarray(block[name][valid], dtype=np.float64)
    if not np.all(matches == 1):
        raise RuntimeError(f"only {int(matches.sum())}/{len(matches)} TARGETIDs matched")
    return output


def sky_to_periodic(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    redshift: np.ndarray,
    *,
    origin_mpc_h: float,
    boxsize_mpc_h: float,
) -> np.ndarray:
    distance = Planck18.comoving_distance(redshift).value * Planck18.h
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    xyz = np.column_stack(
        [
            distance * np.cos(dec) * np.cos(ra),
            distance * np.cos(dec) * np.sin(ra),
            distance * np.sin(dec),
        ]
    )
    return np.mod(xyz + float(origin_mpc_h), boxsize_mpc_h)


def score(prediction: np.ndarray, truth: np.ndarray) -> dict:
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    residual = prediction - truth
    output = {"n": int(len(truth)), "eigenvalues": {}}
    for index in range(3):
        variance = float(np.sum((truth[:, index] - truth[:, index].mean()) ** 2))
        r2 = 1.0 - float(np.sum(residual[:, index] ** 2)) / variance
        correlation = float(np.corrcoef(prediction[:, index], truth[:, index])[0, 1])
        output["eigenvalues"][f"lambda{index + 1}"] = {
            "r2": r2,
            "pearson": correlation,
            "mae": float(np.mean(np.abs(residual[:, index]))),
            "rmse": float(np.sqrt(np.mean(residual[:, index] ** 2))),
            "max_abs": float(np.max(np.abs(residual[:, index]))),
        }
    output["minimum_r2"] = min(
        row["r2"] for row in output["eigenvalues"].values()
    )
    output["mean_r2"] = float(
        np.mean([row["r2"] for row in output["eigenvalues"].values()])
    )
    return output


def main() -> None:
    args = parse_args()
    started = time.time()
    assignment = np.load(args.assignment, mmap_mode="r")
    sample_rows = choose_sample(assignment, args.sample_per_cap_shell, args.seed)
    parent_rows = np.asarray(assignment["parent_node_id"][sample_rows], dtype=np.int64)
    sample_cap = np.asarray(assignment["cap"][sample_rows], dtype=np.int8)
    sample_shell = np.asarray(assignment["shell"][sample_rows], dtype=np.int8)
    catalogue = read_rows(
        args.catalogue,
        parent_rows,
        [
            "TARGETID",
            "RA",
            "DEC",
            "Z",
            "FILE_NUM",
            "HALO_INDEX",
            "LAMBDA1",
            "LAMBDA2",
            "LAMBDA3",
        ],
    )
    target = join_target_truth(
        args.target_input,
        np.asarray(catalogue["TARGETID"], dtype=np.int64),
        chunk_rows=args.target_chunk,
    )
    truth = np.column_stack(
        [np.asarray(catalogue[f"LAMBDA{i}"], dtype=np.float64) for i in (1, 2, 3)]
    )

    slabs = discover_slabs(args.tweb_dir)
    ix_to_slab, slab_xstart, ngrid, boxsize = build_slab_maps(slabs)
    host = load_halo_positions_xcom(
        halo_info_dir=args.halo_info,
        file_nums=np.asarray(catalogue["FILE_NUM"]),
        halo_indices=np.asarray(catalogue["HALO_INDEX"]),
    )
    host_prediction = assign_eigs_from_slabs(
        np.mod(host, boxsize),
        slabs=slabs,
        ix_to_slab=ix_to_slab,
        slab_xstart=slab_xstart,
        ngrid=ngrid,
        boxsize=boxsize,
    )

    variants: dict[str, dict] = {"host_xcom_periodic": score(host_prediction, truth)}
    for redshift_name, redshift in (
        ("z_observed", np.asarray(catalogue["Z"], dtype=np.float64)),
        ("z_cosmo", target["Z_COSMO"]),
    ):
        for origin in (-1000.0, -990.0, 0.0):
            name = f"{redshift_name}_origin_{str(origin).replace('-', 'm').replace('.', 'p')}"
            xyz = sky_to_periodic(
                np.asarray(catalogue["RA"], dtype=np.float64),
                np.asarray(catalogue["DEC"], dtype=np.float64),
                redshift,
                origin_mpc_h=origin,
                boxsize_mpc_h=boxsize,
            )
            prediction = assign_eigs_from_slabs(
                xyz,
                slabs=slabs,
                ix_to_slab=ix_to_slab,
                slab_xstart=slab_xstart,
                ngrid=ngrid,
                boxsize=boxsize,
            )
            variants[name] = score(prediction, truth)

    sky_names = [name for name in variants if name != "host_xcom_periodic"]
    best_sky = max(sky_names, key=lambda name: variants[name]["minimum_r2"])
    host_pass = variants["host_xcom_periodic"]["minimum_r2"] >= args.minimum_host_r2
    sky_pass = variants[best_sky]["minimum_r2"] >= args.minimum_sky_r2
    rsdz_delta = np.asarray(catalogue["Z"], dtype=np.float64) - target["RSDZ"]
    payload = {
        "schema_version": "p8-density-target-alignment-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "P8.9 D0 target-coordinate preflight",
        "inputs": {
            "catalogue": str(args.catalogue),
            "catalogue_sha256": sha256(args.catalogue),
            "target_input": str(args.target_input),
            "assignment": str(args.assignment),
            "assignment_sha256": sha256(args.assignment),
            "tweb_dir": str(args.tweb_dir),
            "halo_info": str(args.halo_info),
            "tweb": {
                "ngrid": int(ngrid),
                "boxsize_mpc_h": float(boxsize),
                "rsmooth_mpc_h": float(slabs[0].path.parent.name.rsplit("_", 1)[-1]),
                "n_slabs": len(slabs),
            },
        },
        "sample": {
            "n": int(len(sample_rows)),
            "per_cap_shell": int(args.sample_per_cap_shell),
            "seed": int(args.seed),
            "cap_counts": {str(i): int(np.sum(sample_cap == i)) for i in (0, 1)},
            "shell_counts": {SHELLS[i]: int(np.sum(sample_shell == i)) for i in range(4)},
        },
        "target_join": {
            "all_targetids_matched": True,
            "max_abs_ra_difference_deg": float(
                np.max(np.abs(np.asarray(catalogue["RA"], dtype=np.float64) - target["RA"]))
            ),
            "max_abs_dec_difference_deg": float(
                np.max(np.abs(np.asarray(catalogue["DEC"], dtype=np.float64) - target["DEC"]))
            ),
            "max_abs_observed_z_minus_rsdz": float(np.max(np.abs(rsdz_delta))),
        },
        "variants": variants,
        "best_sky_variant": best_sky,
        "gates": {
            "host_xcom_reproduces_frozen_labels": bool(host_pass),
            "simple_observer_frame_mapping_is_density_target_ready": bool(sky_pass),
        },
        "pass": bool(host_pass and sky_pass),
        "interpretation": (
            "PASS authorizes large observer-frame delta_R7 target generation with the "
            "recorded sky mapping. FAIL with host PASS isolates the blocker to cut-sky "
            "replica/RSD coordinates; no density model may train until that mapping is fixed."
        ),
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
