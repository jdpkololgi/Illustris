#!/usr/bin/env python3
"""Validate corrected P8.9 delta_R7 trace targets at authoritative galaxies."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

from astropy.cosmology import Planck18
import fitsio
import h5py
import numpy as np

from workflows.abacus_tweb.p6_field_patch_utils import (
    fractional_cell_index,
    trilinear_sample,
)
from workflows.abacus_tweb.p8_density_target_alignment import (
    CATALOGUE,
    HALO_INFO,
    TARGET_INPUT,
    TWEB,
    choose_sample,
    join_target_truth,
    read_rows,
)
from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.abacus_tweb.validate_cutsky_eigs_boxindex_vs_halo_xcom import (
    assign_eigs_from_slabs,
    build_slab_maps,
    discover_slabs,
    load_halo_positions_xcom,
)


ROOT = Path("/pscratch/sd/d/dkololgi/abacus")
ASSIGNMENT = ROOT / "p4_spatial_manifest/active_assignment.npz"
TARGET_MANIFEST = ROOT / "p8_density_phys_v1/targets/target_manifest.json"
OUTPUT = ROOT / "p8_density_phys_v1/target_closure/trace_closure.json"
CAP_NAME = {0: "SGC", 1: "NGC"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, default=CATALOGUE)
    parser.add_argument("--target-input", type=Path, default=TARGET_INPUT)
    parser.add_argument("--assignment", type=Path, default=ASSIGNMENT)
    parser.add_argument("--target-manifest", type=Path, default=TARGET_MANIFEST)
    parser.add_argument("--tweb-dir", type=Path, default=TWEB)
    parser.add_argument("--halo-info", type=Path, default=HALO_INFO)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--sample-per-cap-shell", type=int, default=2_000)
    parser.add_argument("--target-chunk", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--minimum-oracle-r2", type=float, default=0.95)
    parser.add_argument("--minimum-host-r2", type=float, default=0.999)
    return parser.parse_args()


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def sky_to_observer_mpc(
    ra_deg: np.ndarray, dec_deg: np.ndarray, redshift: np.ndarray
) -> np.ndarray:
    distance = Planck18.comoving_distance(redshift).value
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    return np.column_stack([
        distance * np.cos(dec) * np.cos(ra),
        distance * np.cos(dec) * np.sin(ra),
        distance * np.sin(dec),
    ])


def scalar_score(prediction: np.ndarray, truth: np.ndarray) -> dict:
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    valid = np.isfinite(prediction) & np.isfinite(truth)
    prediction = prediction[valid]
    truth = truth[valid]
    residual = prediction - truth
    denominator = float(np.sum((truth - np.mean(truth)) ** 2))
    return {
        "n": int(len(truth)),
        "r2": 1.0 - float(np.sum(residual**2)) / denominator,
        "pearson": float(np.corrcoef(prediction, truth)[0, 1]),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "bias": float(np.mean(residual)),
        "max_abs": float(np.max(np.abs(residual))),
    }


def sample_cap_fields(
    manifest: dict,
    cap: np.ndarray,
    xyz_mpc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    values = np.full(len(cap), np.nan, dtype=np.float32)
    supported = np.zeros(len(cap), dtype=bool)
    diagnostics = {}
    for cap_id, cap_name in CAP_NAME.items():
        selected = np.flatnonzero(cap == cap_id)
        component = manifest["components"][cap_name]
        grid = component["grid"]
        origin = np.asarray(grid["origin_mpc"], dtype=np.float64)
        cell = float(grid["cell_mpc"])
        shape = np.asarray(grid["shape"], dtype=np.int64)
        frac = fractional_cell_index(xyz_mpc[selected], origin, cell)
        inside = np.all((frac >= 0.0) & (frac <= shape - 1.0), axis=1)
        if not np.all(inside):
            raise RuntimeError(f"{cap_name}: {int(np.sum(~inside))} sample points outside P3 grid")
        with h5py.File(component["file"], "r") as handle:
            values[selected] = trilinear_sample(handle["delta_r7"][:], frac)
            nearest = np.rint(frac).astype(np.int64)
            support = np.asarray(handle["science_support"], dtype=bool)
            supported[selected] = np.asarray(
                support[nearest[:, 0], nearest[:, 1], nearest[:, 2]], dtype=bool
            )
        diagnostics[cap_name] = {
            "n": int(len(selected)),
            "inside_grid": int(np.sum(inside)),
            "science_supported_nearest_cell": int(np.sum(supported[selected])),
        }
    return values, supported, diagnostics


def main() -> None:
    args = parse_args()
    started = time.time()
    target_manifest = json.loads(args.target_manifest.read_text())
    contract = target_manifest["contract"]
    if contract.get("observer_coordinate_units") != "Mpc":
        raise RuntimeError("target manifest lacks corrected observer-Mpc contract")
    if abs(float(contract.get("observer_coordinate_h", -1.0)) - float(Planck18.h)) > 1e-12:
        raise RuntimeError("target manifest h does not match Planck18")

    assignment = np.load(args.assignment, mmap_mode="r")
    sample_rows = choose_sample(assignment, args.sample_per_cap_shell, args.seed)
    parent_rows = np.asarray(assignment["parent_node_id"][sample_rows], dtype=np.int64)
    cap = np.asarray(assignment["cap"][sample_rows], dtype=np.int8)
    shell = np.asarray(assignment["shell"][sample_rows], dtype=np.int8)
    catalogue = read_rows(
        args.catalogue,
        parent_rows,
        ["TARGETID", "RA", "DEC", "Z", "FILE_NUM", "HALO_INDEX",
         "LAMBDA1", "LAMBDA2", "LAMBDA3"],
    )
    target_rows = join_target_truth(
        args.target_input,
        np.asarray(catalogue["TARGETID"], dtype=np.int64),
        chunk_rows=args.target_chunk,
    )
    truth_eigenvalues = np.column_stack([
        np.asarray(catalogue[f"LAMBDA{axis}"], dtype=np.float64) for axis in (1, 2, 3)
    ])
    truth_trace = np.sum(truth_eigenvalues, axis=1)

    ra = np.asarray(catalogue["RA"], dtype=np.float64)
    dec = np.asarray(catalogue["DEC"], dtype=np.float64)
    xyz_cosmo = sky_to_observer_mpc(ra, dec, target_rows["Z_COSMO"])
    xyz_observed = sky_to_observer_mpc(
        ra, dec, np.asarray(catalogue["Z"], dtype=np.float64)
    )
    cosmo_prediction, cosmo_support, cosmo_diag = sample_cap_fields(
        target_manifest, cap, xyz_cosmo
    )
    observed_prediction, observed_support, observed_diag = sample_cap_fields(
        target_manifest, cap, xyz_observed
    )

    slabs = discover_slabs(args.tweb_dir)
    ix_to_slab, slab_xstart, ngrid, boxsize = build_slab_maps(slabs)
    host = load_halo_positions_xcom(
        halo_info_dir=args.halo_info,
        file_nums=np.asarray(catalogue["FILE_NUM"]),
        halo_indices=np.asarray(catalogue["HALO_INDEX"]),
    )
    host_eigenvalues = assign_eigs_from_slabs(
        np.mod(host, boxsize), slabs=slabs, ix_to_slab=ix_to_slab,
        slab_xstart=slab_xstart, ngrid=ngrid, boxsize=boxsize,
    )
    host_trace = np.sum(host_eigenvalues, axis=1)

    rows = {
        "host_xcom_direct_slab": scalar_score(host_trace, truth_trace),
        "z_cosmo_cap_trace_all": scalar_score(cosmo_prediction, truth_trace),
        "z_cosmo_cap_trace_supported": scalar_score(
            cosmo_prediction[cosmo_support], truth_trace[cosmo_support]
        ),
        "z_observed_cap_trace_all": scalar_score(observed_prediction, truth_trace),
        "z_observed_cap_trace_supported": scalar_score(
            observed_prediction[observed_support], truth_trace[observed_support]
        ),
    }
    host_pass = rows["host_xcom_direct_slab"]["r2"] >= args.minimum_host_r2
    oracle_pass = rows["z_cosmo_cap_trace_supported"]["r2"] >= args.minimum_oracle_r2
    payload = {
        "schema_version": "p8-density-target-trace-closure-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "stage": "P8.9 corrected target trace closure",
        "inputs": {
            "catalogue": str(args.catalogue),
            "catalogue_sha256": sha256(args.catalogue),
            "assignment": str(args.assignment),
            "assignment_sha256": sha256(args.assignment),
            "target_manifest": str(args.target_manifest),
            "target_manifest_sha256": sha256(args.target_manifest),
            "target_input": str(args.target_input),
            "tweb_dir": str(args.tweb_dir),
        },
        "sample": {
            "n": int(len(sample_rows)),
            "per_cap_shell": int(args.sample_per_cap_shell),
            "seed": int(args.seed),
            "cap_counts": {str(i): int(np.sum(cap == i)) for i in (0, 1)},
            "shell_counts": {str(i): int(np.sum(shell == i)) for i in range(4)},
        },
        "sampling": {
            "interpolation": "trilinear on P3 cell-centre lattice with border padding",
            "z_cosmo": cosmo_diag,
            "z_observed": observed_diag,
        },
        "rows": rows,
        "gates": {
            "host_xcom_direct_trace_reproduces_labels": bool(host_pass),
            "z_cosmo_supported_cap_trace_r2_at_least_minimum": bool(oracle_pass),
            "observed_z_is_diagnostic_not_oracle_gate": True,
        },
        "thresholds": {
            "minimum_host_r2": float(args.minimum_host_r2),
            "minimum_oracle_r2": float(args.minimum_oracle_r2),
        },
        "pass": bool(host_pass and oracle_pass),
        "interpretation": (
            "PASS validates target trace construction and P3 interpolation only. The "
            "global cap FFT tensor must pass separately before D0 training. Z_COSMO is "
            "an oracle row and may never be quoted as deployable DESI performance."
        ),
        "elapsed_seconds": float(time.time() - started),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
