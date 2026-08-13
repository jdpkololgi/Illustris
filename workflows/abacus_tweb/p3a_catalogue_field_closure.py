#!/usr/bin/env python3
"""Catalogue/field/target closure audit for the completed P3a products.

This is deliberately stronger than a catalogue row-count check.  It tests three
different links in the P1b -> P3a chain:

1. an independent CIC redeposition of the canonical context coordinates must
   reproduce the stored count field voxel by voxel;
2. repeated rows carrying the same (FILE_NUM, BOX_INDEX, HALO_INDEX) host key
   must carry identical tidal labels, and CWEB must agree with the eigenvalues;
3. the T-Web trace must correlate positively with the selection-aware galaxy
   count field sampled at the exact catalogue coordinates, well above a
   within-cap/shell shuffled-label control.

The final test is a spatial-coherence/scrambling diagnostic, not a proof that a
particular halo matcher is physically unique.  Galaxy counts are a biased,
shot-noisy tracer, so their correlation with the smoothed matter trace is not
expected to be one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import fitsio
import h5py
import numpy as np
from scipy.stats import spearmanr

from p10_target_contract import stored_class_consistency


CAPS = ((0, "SGC"), (1, "NGC"))
SHELL_NAMES = ("0.15_0.25", "0.25_0.35", "0.35_0.45", "0.45_0.55")


def sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def iter_chunks(shape: tuple[int, int, int], chunk: tuple[int, int, int]):
    for i in range(0, shape[0], chunk[0]):
        for j in range(0, shape[1], chunk[1]):
            for k in range(0, shape[2], chunk[2]):
                yield (
                    slice(i, min(i + chunk[0], shape[0])),
                    slice(j, min(j + chunk[1], shape[1])),
                    slice(k, min(k + chunk[2], shape[2])),
                )


def independent_cic(xyz: np.ndarray, origin: np.ndarray, cell: float,
                    shape: tuple[int, int, int]) -> tuple[np.ndarray, float]:
    """Independent mass-conserving CIC implementation in ix,iy,iz order."""
    out = np.zeros(shape, dtype=np.float32)
    u = (np.asarray(xyz, dtype=np.float64) - origin) / cell - 0.5
    base = np.floor(u).astype(np.int64)
    frac = u - base
    extent = np.asarray(shape, dtype=np.int64)
    lost = 0.0
    for dx in (0, 1):
        wx = frac[:, 0] if dx else 1.0 - frac[:, 0]
        for dy in (0, 1):
            wy = frac[:, 1] if dy else 1.0 - frac[:, 1]
            for dz in (0, 1):
                wz = frac[:, 2] if dz else 1.0 - frac[:, 2]
                idx = base + np.array((dx, dy, dz), dtype=np.int64)
                weight = wx * wy * wz
                valid = np.all((idx >= 0) & (idx < extent), axis=1)
                np.add.at(
                    out,
                    (idx[valid, 0], idx[valid, 1], idx[valid, 2]),
                    weight[valid].astype(np.float32),
                )
                lost += float(np.sum(weight[~valid], dtype=np.float64))
    return out, lost


def trilinear_sample(field: np.ndarray, xyz: np.ndarray, origin: np.ndarray,
                     cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Sample a cell-centred canonical field at observer-frame coordinates."""
    u = (np.asarray(xyz, dtype=np.float64) - origin) / cell - 0.5
    base = np.floor(u).astype(np.int64)
    frac = u - base
    shape = np.asarray(field.shape, dtype=np.int64)
    valid = np.all((base >= 0) & ((base + 1) < shape), axis=1)
    value = np.zeros(len(xyz), dtype=np.float64)
    for dx in (0, 1):
        wx = frac[:, 0] if dx else 1.0 - frac[:, 0]
        for dy in (0, 1):
            wy = frac[:, 1] if dy else 1.0 - frac[:, 1]
            for dz in (0, 1):
                wz = frac[:, 2] if dz else 1.0 - frac[:, 2]
                idx = base[valid] + np.array((dx, dy, dz), dtype=np.int64)
                value[valid] += (
                    wx[valid] * wy[valid] * wz[valid]
                    * field[idx[:, 0], idx[:, 1], idx[:, 2]]
                )
    value[~valid] = np.nan
    return value, valid


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    keep = np.isfinite(x) & np.isfinite(y)
    if int(keep.sum()) < 3 or np.ptp(x[keep]) == 0 or np.ptp(y[keep]) == 0:
        return float("nan")
    return float(spearmanr(x[keep], y[keep]).statistic)


def deterministic_stratified_sample(active: np.ndarray, cap: np.ndarray,
                                    shell: np.ndarray, per_stratum: int,
                                    seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []
    for cap_id, _ in CAPS:
        for shell_id in range(4):
            ids = np.flatnonzero(active & (cap == cap_id) & (shell == shell_id))
            if len(ids) > per_stratum:
                ids = np.sort(rng.choice(ids, per_stratum, replace=False))
            selected.append(ids)
    return np.concatenate(selected).astype(np.int64)


def host_consistency(table: np.ndarray, active: np.ndarray) -> dict:
    ids = np.flatnonzero(active)
    file_num = np.asarray(table["FILE_NUM"][ids], dtype=np.int64)
    box = np.asarray(table["BOX_INDEX"][ids], dtype=np.int64)
    halo = np.asarray(table["HALO_INDEX"][ids], dtype=np.int64)
    # CWEB was generated from the stored float32 eigenvalues. Preserve that
    # representation for the exact threshold-class closure; upcasting first
    # makes a value equal to float32(0.2) compare greater than float64(0.2).
    lam_native = np.column_stack(
        [table[f"LAMBDA{i}"][ids] for i in (1, 2, 3)]
    ).astype(np.float32, copy=False)
    lam = lam_native.astype(np.float64)
    order = np.lexsort((halo, box, file_num))
    same = (
        (file_num[order][1:] == file_num[order][:-1])
        & (box[order][1:] == box[order][:-1])
        & (halo[order][1:] == halo[order][:-1])
    )
    delta = np.abs(lam[order][1:] - lam[order][:-1])
    repeated_pairs = int(same.sum())
    max_delta = float(np.max(delta[same])) if repeated_pairs else 0.0
    class_check = stored_class_consistency(lam_native, table["CWEB"][ids])
    cweb_mismatch = int(class_check["mismatch"].sum())
    cweb_boundary = int(class_check["boundary_ambiguous"].sum())
    cweb_nonboundary = int(class_check["nonboundary_mismatch"].sum())
    return {
        "active_rows": int(len(ids)),
        "repeated_host_adjacent_pairs": repeated_pairs,
        "max_abs_dlambda_within_repeated_host": max_delta,
        "cweb_threshold": 0.2,
        "cweb_mismatch_rows": cweb_mismatch,
        "cweb_threshold_quantization_ambiguity_rows": cweb_boundary,
        "cweb_nonboundary_mismatch_rows": cweb_nonboundary,
    }


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/graph_constructions/"
        "path1_fiberassign_mock_bgs_maglim_rs7_points.npy"))
    ap.add_argument("--index", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p1b_full_footprint/canonical_index.npz"))
    ap.add_argument("--catalogue", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_05062026_rsmooth_7/"
        "mock_bgs_maglim_path1_fiberassign_graph_ready_with_tweb_eigs_rs7_"
        "ngrid2048_thr0p2_halo_xcom.fits"))
    ap.add_argument("--field-manifest", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p3_full_footprint/field_manifest.json"))
    ap.add_argument("--out", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p3_full_footprint/"
        "catalogue_field_closure.json"))
    ap.add_argument("--sample-per-cap-shell", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=20260718)
    ap.add_argument("--min-cap-trace-spearman", type=float, default=0.10)
    ap.add_argument("--min-cap-shuffle-separation", type=float, default=0.08)
    args = ap.parse_args()
    started = time.time()

    points = np.load(args.points, mmap_mode="r")
    index = np.load(args.index)
    cap = np.asarray(index["cap"], dtype=np.uint8)
    shell = np.asarray(index["shell"], dtype=np.int8)
    context = np.asarray(index["context"], dtype=bool)
    active = np.asarray(index["active"], dtype=bool) & np.asarray(index["valid_target"], dtype=bool)
    if points.shape != (len(cap), 4):
        raise RuntimeError("canonical points/index length mismatch")
    if not np.array_equal(np.asarray(points[:, 3], dtype=np.uint8), cap):
        raise RuntimeError("canonical point cap column differs from P1b index")

    columns = ["FILE_NUM", "BOX_INDEX", "HALO_INDEX", "CWEB",
               "LAMBDA1", "LAMBDA2", "LAMBDA3"]
    table = fitsio.read(str(args.catalogue), columns=columns)
    if len(table) != len(cap):
        raise RuntimeError("catalogue/index length mismatch")
    host = host_consistency(table, active)
    sample_ids = deterministic_stratified_sample(
        active, cap, shell, args.sample_per_cap_shell, args.seed)
    manifest = json.loads(args.field_manifest.read_text())
    rng = np.random.default_rng(args.seed + 1)
    cap_reports = {}

    for cap_id, cap_name in CAPS:
        metadata = manifest["components"][cap_name]
        grid = metadata["grid"]
        origin = np.asarray(grid["origin_mpc"], dtype=np.float64)
        cell = float(grid["cell_mpc"])
        shape = tuple(int(v) for v in grid["shape"])
        cap_context = context & (cap == cap_id)
        reconstructed, lost = independent_cic(
            np.asarray(points[cap_context, :3], dtype=np.float64), origin, cell, shape)

        max_abs = 0.0
        sum_abs = 0.0
        mismatched = 0
        compared = 0
        with h5py.File(metadata["file"], "r") as handle:
            count_ds = handle["counts"]
            for slices in iter_chunks(shape, tuple(int(v) for v in count_ds.chunks)):
                observed = np.asarray(count_ds[slices], dtype=np.float32)
                delta = np.abs(observed - reconstructed[slices])
                max_abs = max(max_abs, float(np.max(delta)))
                sum_abs += float(np.sum(delta, dtype=np.float64))
                mismatched += int(np.count_nonzero(delta > 5.0e-5))
                compared += int(delta.size)
            contrast = np.asarray(handle["log_count_ratio"], dtype=np.float32)
        del reconstructed

        ids = sample_ids[cap[sample_ids] == cap_id]
        sampled_contrast, valid_interp = trilinear_sample(
            contrast, np.asarray(points[ids, :3], dtype=np.float64), origin, cell)
        del contrast
        trace = (
            np.asarray(table["LAMBDA1"][ids], dtype=np.float64)
            + np.asarray(table["LAMBDA2"][ids], dtype=np.float64)
            + np.asarray(table["LAMBDA3"][ids], dtype=np.float64)
        )
        pooled_rho = safe_spearman(sampled_contrast, trace)
        shuffled = trace.copy()
        for shell_id in range(4):
            local = np.flatnonzero(shell[ids] == shell_id)
            shuffled[local] = shuffled[rng.permutation(local)]
        shuffled_rho = safe_spearman(sampled_contrast, shuffled)
        strata = {}
        for shell_id, shell_name in enumerate(SHELL_NAMES):
            local = shell[ids] == shell_id
            strata[shell_name] = {
                "n": int(local.sum()),
                "trace_spearman": safe_spearman(sampled_contrast[local], trace[local]),
                "shuffled_trace_spearman": safe_spearman(sampled_contrast[local], shuffled[local]),
                "lambda_spearman": {
                    f"lambda{i}": safe_spearman(
                        sampled_contrast[local],
                        np.asarray(table[f"LAMBDA{i}"][ids[local]], dtype=np.float64),
                    ) for i in (1, 2, 3)
                },
            }
        cap_reports[cap_name] = {
            "context_rows_redeposited": int(cap_context.sum()),
            "cic_lost_weight": float(lost),
            "cic_voxels_compared": compared,
            "cic_max_abs_difference": max_abs,
            "cic_mean_abs_difference": sum_abs / max(compared, 1),
            "cic_voxels_above_5e-5": mismatched,
            "sample_rows": int(len(ids)),
            "sample_interpolation_valid": int(valid_interp.sum()),
            "pooled_trace_spearman": pooled_rho,
            "pooled_shuffled_trace_spearman": shuffled_rho,
            "pooled_shuffle_separation": pooled_rho - shuffled_rho,
            "by_shell": strata,
        }

    gates = {
        "canonical_lengths_match": len(points) == len(table) == len(cap),
        "host_labels_identical_for_repeated_host_keys": (
            host["max_abs_dlambda_within_repeated_host"] <= 1.0e-7),
        "cweb_matches_thresholded_eigenvalues_away_from_quantization_boundary": (
            host["cweb_nonboundary_mismatch_rows"] == 0
        ),
        "independent_cic_redeposit_matches": all(
            v["cic_lost_weight"] <= 1.0e-8
            and v["cic_max_abs_difference"] <= 5.0e-5
            and v["cic_voxels_above_5e-5"] == 0
            for v in cap_reports.values()),
        "all_sample_coordinates_interpolate": all(
            v["sample_interpolation_valid"] == v["sample_rows"]
            for v in cap_reports.values()),
        "trace_spatial_coherence_in_both_caps": all(
            v["pooled_trace_spearman"] >= args.min_cap_trace_spearman
            for v in cap_reports.values()),
        "trace_beats_within_shell_shuffle_in_both_caps": all(
            v["pooled_shuffle_separation"] >= args.min_cap_shuffle_separation
            for v in cap_reports.values()),
        "trace_beats_shuffle_in_every_cap_shell": all(
            shell_report["trace_spearman"]
            - shell_report["shuffled_trace_spearman"] >= args.min_cap_shuffle_separation
            for cap_report in cap_reports.values()
            for shell_report in cap_report["by_shell"].values()),
    }
    payload = {
        "schema_version": 1,
        "stage": "P3a catalogue-field-target closure",
        "interpretation": (
            "CIC equality is an exact coordinate/deposition closure. Host-key equality and CWEB "
            "consistency are target-table closures. Count-field/T-Web correlation relative to "
            "shuffling is a spatial-coherence diagnostic, not a claim that biased galaxy counts "
            "equal the smoothed matter density."
        ),
        "inputs": {
            "points": str(args.points), "points_sha256": sha256(args.points),
            "canonical_index": str(args.index), "canonical_index_sha256": sha256(args.index),
            "catalogue": str(args.catalogue),
            "field_manifest": str(args.field_manifest),
            "field_manifest_sha256": sha256(args.field_manifest),
        },
        "sample": {"per_cap_shell_max": args.sample_per_cap_shell, "seed": args.seed},
        "host_target_consistency": host,
        "caps": cap_reports,
        "gates": gates,
        "pass": all(gates.values()),
        "elapsed_seconds": time.time() - started,
        "repo": str(repo),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=bool) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True, default=bool))
    if not payload["pass"]:
        raise RuntimeError(f"catalogue/field closure failed: {gates}")


if __name__ == "__main__":
    main()
