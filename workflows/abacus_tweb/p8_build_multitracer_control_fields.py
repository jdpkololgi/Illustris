#!/usr/bin/env python3
"""Build label-free MT3b field controls on the frozen P3 lattices.

Products
--------
``original_tsc``
    Exact TSC deposition of the unchanged Bright and Proxy-Faint context.
``thin_seed*``
    CIC deposition after thinning the Bright+Faint union to the original
    Bright count independently in cap/redshift strata.
``faint_position_null``
    CIC Faint field after permuting angular directions among Faint objects in
    the same cap/redshift stratum while retaining every radius and the angular
    direction multiset.  Bright remains the immutable P3 CIC field.

No tidal target, supervised fold, or validation label is read.  These products
are observation controls only; their existence does not complete MT3.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p3a_build_canonical_fields import (
    GridSpec,
    cic_deposit,
    fractional_index,
    sha256,
)
from workflows.abacus_tweb.p6_refit_fullcap_selection import radius_to_redshift_grid
from workflows.abacus_tweb.p8_build_multitracer_fields import complete_cic_support
from workflows.abacus_tweb.p8_deterministic_common import atomic_json


ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1")
P3 = Path("/pscratch/sd/d/dkololgi/abacus/p3_full_footprint/field_manifest.json")
CAPS = ((0, "SGC"), (1, "NGC"))
RED_SHIFT_STRATA = (
    ("context_low", 0.10, 0.15),
    ("0p15_0p25", 0.15, 0.25),
    ("0p25_0p35", 0.25, 0.35),
    ("0p35_0p45", 0.35, 0.45),
    ("0p45_0p55", 0.45, 0.55),
    ("context_high", 0.55, 0.60),
)


def complete_tsc_support(xyz: np.ndarray, spec: GridSpec) -> np.ndarray:
    """Whether every point's 3x3x3 TSC stencil lies on the frozen grid."""
    centre = np.floor(fractional_index(xyz, spec) + 0.5).astype(np.int64)
    shape = np.asarray(spec.shape, dtype=np.int64)
    return np.all((centre - 1 >= 0) & (centre + 1 < shape), axis=1)


def tsc_deposit(
    xyz: np.ndarray, spec: GridSpec, out: np.ndarray | None = None
) -> tuple[np.ndarray, dict]:
    """Mass-conserving triangular-shaped-cloud deposition."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if out is None:
        out = np.zeros(spec.shape, dtype=np.float32)
    if out.shape != spec.shape or out.dtype != np.float32:
        raise ValueError("TSC destination must be float32 with canonical shape")
    u = fractional_index(xyz, spec)
    centre = np.floor(u + 0.5).astype(np.int64)
    distance = u - centre
    weights = (
        (
            0.5 * (0.5 - distance[:, axis]) ** 2,
            0.75 - distance[:, axis] ** 2,
            0.5 * (0.5 + distance[:, axis]) ** 2,
        )
        for axis in range(3)
    )
    wx, wy, wz = weights
    shape = np.asarray(spec.shape, dtype=np.int64)
    deposited = 0.0
    lost = 0.0
    for ax, dx in enumerate((-1, 0, 1)):
        for ay, dy in enumerate((-1, 0, 1)):
            for az, dz in enumerate((-1, 0, 1)):
                index = centre + np.array([dx, dy, dz], dtype=np.int64)
                weight = wx[ax] * wy[ay] * wz[az]
                valid = np.all((index >= 0) & (index < shape), axis=1)
                np.add.at(
                    out,
                    (index[valid, 0], index[valid, 1], index[valid, 2]),
                    weight[valid].astype(np.float32),
                )
                deposited += float(weight[valid].sum(dtype=np.float64))
                lost += float(weight[~valid].sum(dtype=np.float64))
    return out, {
        "input_points": int(len(xyz)),
        "deposited_weight": deposited,
        "lost_weight": lost,
    }


def deposit_chunked(
    xyz: np.ndarray, spec: GridSpec, scheme: str, chunk: int
) -> tuple[np.ndarray, dict]:
    result = np.zeros(spec.shape, dtype=np.float32)
    total = {"input_points": 0, "deposited_weight": 0.0, "lost_weight": 0.0}
    function = cic_deposit if scheme == "cic" else tsc_deposit
    support_function = complete_cic_support if scheme == "cic" else complete_tsc_support
    excluded = 0
    for start in range(0, len(xyz), chunk):
        block = np.asarray(xyz[start : start + chunk], dtype=np.float64)
        supported = support_function(block, spec)
        excluded += int(np.count_nonzero(~supported))
        block = block[supported]
        result, stats = function(block, spec, out=result)
        for key in total:
            total[key] += stats[key]
    total["input_rows_before_support"] = int(len(xyz))
    total["grid_edge_excluded_rows"] = excluded
    total["sum_readback"] = float(result.sum(dtype=np.float64))
    total["conserved"] = bool(
        abs(total["sum_readback"] - total["input_points"])
        <= max(1.0e-4, 2.0e-6 * max(total["input_points"], 1))
        and total["lost_weight"] <= 1.0e-8
    )
    return result, total


def redshift_stratum(redshift: np.ndarray) -> np.ndarray:
    result = np.full(len(redshift), -1, dtype=np.int8)
    for index, (_, lower, upper) in enumerate(RED_SHIFT_STRATA):
        result[(redshift >= lower) & (redshift < upper)] = index
    return result


def density_matched_indices(
    *, tracer: np.ndarray, context: np.ndarray, cap: np.ndarray,
    stratum: np.ndarray, seed: int,
) -> tuple[np.ndarray, dict]:
    """Thin the union to the Bright count in every cap/redshift stratum."""
    rng = np.random.default_rng(seed)
    selected = []
    audit = {}
    for cap_id, cap_name in CAPS:
        audit[cap_name] = {}
        for shell_id, (shell_name, _, _) in enumerate(RED_SHIFT_STRATA):
            candidate = np.flatnonzero(context & (cap == cap_id) & (stratum == shell_id))
            n_bright = int(np.count_nonzero(tracer[candidate] == 0))
            if n_bright > len(candidate):
                raise RuntimeError("Bright count exceeds union count")
            chosen = (
                np.sort(rng.choice(candidate, size=n_bright, replace=False))
                if n_bright else np.empty(0, dtype=np.int64)
            )
            selected.append(chosen)
            original_by_tracer = [int(np.count_nonzero(tracer[candidate] == value)) for value in (0, 1)]
            retained_by_tracer = [int(np.count_nonzero(tracer[chosen] == value)) for value in (0, 1)]
            audit[cap_name][shell_name] = {
                "union_rows": int(len(candidate)), "target_bright_density_rows": n_bright,
                "retained_rows": int(len(chosen)),
                "original_by_tracer": {"bright": original_by_tracer[0], "faint": original_by_tracer[1]},
                "retained_by_tracer": {"bright": retained_by_tracer[0], "faint": retained_by_tracer[1]},
                "retention_fraction_by_tracer": {
                    "bright": retained_by_tracer[0] / max(original_by_tracer[0], 1),
                    "faint": retained_by_tracer[1] / max(original_by_tracer[1], 1),
                },
            }
    rows = np.sort(np.concatenate(selected))
    return rows, audit


def angular_null(
    xyz: np.ndarray, *, cap: np.ndarray, stratum: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Pair every radius with another same-stratum angular direction."""
    xyz = np.asarray(xyz, dtype=np.float64)
    radius = np.linalg.norm(xyz, axis=1)
    direction = xyz / radius[:, None]
    result = np.empty_like(xyz)
    donor = np.empty(len(xyz), dtype=np.int64)
    rng = np.random.default_rng(seed)
    audit = {}
    for cap_id, cap_name in CAPS:
        audit[cap_name] = {}
        group_ids = np.unique(stratum[cap == cap_id])
        for group_id in group_ids:
            shell_name = f"fine_z_group_{int(group_id):02d}"
            rows = np.flatnonzero((cap == cap_id) & (stratum == group_id))
            order = rng.permutation(len(rows))
            if len(rows) > 1 and np.array_equal(order, np.arange(len(rows))):
                order = np.roll(order, 1)
            chosen = rows[order]
            result[rows] = radius[rows, None] * direction[chosen]
            donor[rows] = chosen
            audit[cap_name][shell_name] = {
                "rows": int(len(rows)),
                "fixed_points": int(np.count_nonzero(chosen == rows)),
            }
    assigned = stratum >= 0
    if not np.allclose(np.linalg.norm(result[assigned], axis=1), radius[assigned], rtol=0, atol=1e-9):
        raise RuntimeError("angular null changed Faint radii")
    return result, donor, audit


def grid_spec(component: dict) -> GridSpec:
    grid = component["grid"]
    return GridSpec(
        origin=tuple(float(value) for value in grid["origin_mpc"]),
        shape=tuple(int(value) for value in grid["shape"]),
        cell_mpc=float(grid["cell_mpc"]),
        padding_mpc=float(grid["padding_mpc"]),
    )


def write_fields(
    *, output: Path, name: str, scheme: str, points: np.ndarray,
    tracer: np.ndarray, cap: np.ndarray, selected: np.ndarray,
    p3: dict, chunk: int, include_bright: bool, include_faint: bool,
    force: bool,
) -> dict:
    product = output / name
    product.mkdir(parents=True, exist_ok=True)
    product_manifest = product / "manifest.json"
    product_marker = product / "CONTROL_FIELD_READY"
    if product_manifest.exists() and product_marker.exists() and not force:
        return json.loads(product_manifest.read_text())
    components = {}
    gates = []
    for cap_id, cap_name in CAPS:
        spec = grid_spec(p3["components"][cap_name])
        path = product / f"{cap_name.lower()}_{scheme}_counts.h5"
        rows = selected[cap[selected] == cap_id]
        datasets, deposition = {}, {}
        print(
            f"[MT3b fields] product={name} cap={cap_name} scheme={scheme} rows={len(rows):,}",
            flush=True,
        )
        for tracer_id, tracer_name, enabled in (
            (0, "bright", include_bright), (1, "faint", include_faint)
        ):
            if not enabled:
                continue
            tracer_rows = rows[tracer[rows] == tracer_id]
            field, stats = deposit_chunked(points[tracer_rows, :3], spec, scheme, chunk)
            datasets[f"{tracer_name}_counts"] = field
            deposition[tracer_name] = stats
            gates.append(stats["conserved"])
        with h5py.File(path, "w") as handle:
            for dataset_name, field in datasets.items():
                handle.create_dataset(
                    dataset_name, data=field,
                    chunks=tuple(min(64, size) for size in spec.shape),
                    compression="lzf", shuffle=True,
                )
            handle.attrs["assignment_scheme"] = scheme
            handle.attrs["control_product"] = name
        components[cap_name] = {
            "file": str(path), "file_sha256": sha256(path), "grid": spec.as_dict(),
            "datasets": list(datasets), "deposition": deposition,
        }
        del datasets
    report = {
        "name": name, "scheme": scheme, "components": components,
        "pass": all(gates), "labels_read": False,
    }
    atomic_json(product_manifest, report)
    if report["pass"]:
        product_marker.write_text(f"manifest_sha256={sha256(product_manifest)}\n")
    elif product_marker.exists():
        product_marker.unlink()
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--p3-manifest", type=Path, default=P3)
    parser.add_argument("--product", default="bf_proxy_response_v1")
    parser.add_argument("--thin-seeds", type=int, nargs="+", default=(17, 42, 2718))
    parser.add_argument("--null-seed", type=int, default=314159)
    parser.add_argument("--chunk", type=int, default=250_000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.root / "classical/control_fields" / args.product
    marker = output / "MT3B_CONTROL_FIELDS_READY"
    manifest_path = output / "manifest.json"
    if marker.exists() and manifest_path.exists() and not args.force:
        print(manifest_path.read_text(), flush=True)
        return
    started = time.time()
    output.mkdir(parents=True, exist_ok=True)
    catalogue_manifest_path = args.root / "catalogues" / args.product / "manifest.json"
    catalogue = json.loads(catalogue_manifest_path.read_text())
    p3 = json.loads(args.p3_manifest.read_text())
    points = np.load(catalogue["points"], mmap_mode="r")
    index = np.load(catalogue["index"], mmap_mode="r")
    tracer = np.asarray(index["tracer_type"], dtype=np.uint8)
    context = np.asarray(index["context"], dtype=bool)
    cap = np.asarray(index["cap"], dtype=np.uint8)
    radius_grid, redshift_grid = radius_to_redshift_grid(0.10, 0.60)
    radius = np.linalg.norm(np.asarray(points[:, :3], dtype=np.float64), axis=1)
    redshift = np.interp(radius, radius_grid, redshift_grid)
    stratum = redshift_stratum(redshift)
    if np.count_nonzero(context & (stratum < 0)):
        raise RuntimeError("context rows fall outside registered redshift strata")
    context_rows = np.flatnonzero(context)

    products = {}
    products["original_tsc"] = write_fields(
        output=output, name="original_tsc", scheme="tsc", points=points,
        tracer=tracer, cap=cap, selected=context_rows, p3=p3, chunk=args.chunk,
        include_bright=True, include_faint=True, force=args.force,
    )
    thin_records = {}
    for seed in args.thin_seeds:
        selected, audit = density_matched_indices(
            tracer=tracer, context=context, cap=cap, stratum=stratum, seed=seed
        )
        selected_path = output / f"thin_seed{seed}_parent_rows.npy"
        np.save(selected_path, selected)
        name = f"thin_seed{seed}_cic"
        products[name] = write_fields(
            output=output, name=name, scheme="cic", points=points,
            tracer=tracer, cap=cap, selected=selected, p3=p3, chunk=args.chunk,
            include_bright=True, include_faint=True, force=args.force,
        )
        thin_records[str(seed)] = {
            "selected_parent_rows": str(selected_path),
            "selected_parent_rows_sha256": sha256(selected_path), "audit": audit,
        }

    faint_rows = np.flatnonzero(context & (tracer == 1))
    null_stratum = np.floor((redshift - 0.10) / 0.01).astype(np.int16)
    faint_null, donor_local, null_audit = angular_null(
        np.asarray(points[faint_rows, :3], dtype=np.float64),
        cap=cap[faint_rows], stratum=null_stratum[faint_rows], seed=args.null_seed,
    )
    donor_path = output / "faint_position_null_direction_donor_parent_rows.npy"
    np.save(donor_path, faint_rows[donor_local])
    # Reuse the canonical row order and replace only Faint XYZ in a temporary
    # in-memory view passed to the depositor.  Bright is referenced from P3.
    null_points = np.empty((len(faint_rows), 4), dtype=np.float64)
    null_points[:, :3] = faint_null
    null_points[:, 3] = cap[faint_rows]
    null_tracer = np.ones(len(faint_rows), dtype=np.uint8)
    null_cap = cap[faint_rows]
    products["faint_position_null_cic"] = write_fields(
        output=output, name="faint_position_null_cic", scheme="cic",
        points=null_points, tracer=null_tracer, cap=null_cap,
        selected=np.arange(len(faint_rows), dtype=np.int64), p3=p3, chunk=args.chunk,
        include_bright=False, include_faint=True, force=args.force,
    )

    gates = {
        "all_field_products_conserve_counts": all(row["pass"] for row in products.values()),
        "three_density_matched_seeds": len(thin_records) == 3,
        "null_preserves_all_faint_rows": int(sum(
            row["rows"] for cap_row in null_audit.values() for row in cap_row.values()
        )) == len(faint_rows),
        "labels_not_read": True,
        "frozen_p3_lattices_reused": True,
    }
    manifest = {
        "schema_version": "p8-mt3b-control-fields-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "product": args.product, "status": "complete" if all(gates.values()) else "failed_gate",
        "pass": all(gates.values()), "elapsed_seconds": time.time() - started,
        "products": products, "density_matched_thinning": thin_records,
        "faint_position_null": {
            "seed": args.null_seed, "audit": null_audit,
            "direction_donor_parent_rows": str(donor_path),
            "direction_donor_parent_rows_sha256": sha256(donor_path),
            "contract": "radii unchanged; direction multiset permuted within cap and Delta-z=0.01 bins",
        },
        "inputs": {
            "catalogue_manifest": str(catalogue_manifest_path),
            "catalogue_manifest_sha256": sha256(catalogue_manifest_path),
            "p3_manifest": str(args.p3_manifest),
            "p3_manifest_sha256": sha256(args.p3_manifest),
        },
        "gates": gates,
        "completion_scope": "field inputs only; MT3 estimators remain unevaluated",
    }
    atomic_json(manifest_path, manifest)
    if not manifest["pass"]:
        if marker.exists():
            marker.unlink()
        raise RuntimeError(f"MT3b control field gates failed: {gates}")
    marker.write_text(f"manifest_sha256={sha256(manifest_path)}\nmt3_complete=false\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
