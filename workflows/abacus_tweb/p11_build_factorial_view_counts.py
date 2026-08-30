#!/usr/bin/env python3
"""Materialize truth-free P11 dense/assigned count views on canonical P3 grids.

The targetable forFA catalogue defines V_dense and the phase-matched alternate-MTL
assignment catalogue defines V_assign.  Both stages are represented as separate
BRIGHT and FAINT CIC count fields, masked to the identical random-derived support
used by the deployable V_final view.  V_final BRIGHT remains the immutable P3 field.
Because the current mocks have C_z=1 and no separately validated final-FAINT LSS
catalogue, V_final FAINT is explicitly an identity reference to supported V_assign
FAINT; it is not claimed to be a Loa pointwise product.

This builder reads no T-web target and refuses ph001.
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

from workflows.abacus_tweb.p8_build_multitracer_catalogues import sky_to_points
from workflows.abacus_tweb.p8_build_multitracer_control_fields import write_fields
from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.abacus_tweb.p8_build_multitracer_fields import context_redshift
from workflows.abacus_tweb.p10_multitracer_source_audit import (
    BRIGHT_BITS,
    FAINT_BITS,
    _read,
    paths_for_phase,
    target_counts,
)


VISIBLE = ("ph002", "ph003", "ph004", "ph005", "ph006")
SEALED = "ph001"
ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase")
OUTPUT = ROOT / "p11_factorial_views_v1"
REGISTRY = REPO_ROOT / "configs/p10_phase_registry_v1.json"
SOURCE_MARKER = OUTPUT / "FACTORIAL_VIEW_SOURCES_READY.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def classify(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.int64)
    bright = (bits & BRIGHT_BITS) != 0
    faint = (bits & FAINT_BITS) != 0
    if np.any(bright & faint):
        raise RuntimeError("ambiguous BRIGHT+FAINT target rows")
    result = np.full(len(bits), 255, dtype=np.uint8)
    result[bright] = 0
    result[faint] = 1
    return result


def targetable_rows(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    table = _read(
        path,
        ("TARGETID", "BGS_TARGET", "RA", "DEC", "RSDZ", "TRUEZ", "IN_Y5"),
    )
    tracer = classify(table["BGS_TARGET"])
    keep = (
        (tracer != 255)
        & np.asarray(table["IN_Y5"], dtype=bool)
        & context_redshift(table["RSDZ"])
    )
    table = table[keep]
    tracer = tracer[keep]
    targetid = np.asarray(table["TARGETID"], dtype=np.int64)
    order = np.argsort(targetid, kind="mergesort")
    table = table[order]
    tracer = tracer[order]
    targetid = targetid[order]
    unique = np.r_[True, targetid[1:] != targetid[:-1]]
    duplicate_rows = int(np.count_nonzero(~unique))
    if duplicate_rows:
        # forFA should be unique.  Retaining the first row is safe only when all
        # duplicate rows have already been caught upstream; fail here instead.
        raise RuntimeError(f"forFA has {duplicate_rows} duplicate accepted TARGETIDs")
    return table, tracer, {
        "source_rows": int(len(keep)),
        "accepted_rows": int(len(table)),
        "bright_rows": int(np.count_nonzero(tracer == 0)),
        "faint_rows": int(np.count_nonzero(tracer == 1)),
        "requires_IN_Y5": True,
        "redshift_contract": "0.10<=RSDZ<0.60 excluding sentinel 0.585--0.595",
    }


def assigned_subset(
    dense: np.ndarray, dense_tracer: np.ndarray, assigned_path: Path
) -> tuple[np.ndarray, np.ndarray, dict]:
    assigned = _read(assigned_path, ("TARGETID", "BGS_TARGET"))
    collapsed = target_counts(assigned["TARGETID"], assigned["BGS_TARGET"])
    if np.any(collapsed["bright"] & collapsed["faint"]):
        raise RuntimeError("assigned catalogue has ambiguous BRIGHT+FAINT TARGETIDs")
    dense_id = np.asarray(dense["TARGETID"], dtype=np.int64)
    assigned_id = np.asarray(collapsed["targetid"], dtype=np.int64)
    position = np.searchsorted(dense_id, assigned_id)
    matched = (position < len(dense_id)) & (
        dense_id[np.minimum(position, len(dense_id) - 1)] == assigned_id
    )
    position = position[matched]
    source_tracer = np.where(
        collapsed["bright"][matched], 0, np.where(collapsed["faint"][matched], 1, 255)
    ).astype(np.uint8)
    valid = source_tracer != 255
    position = position[valid]
    source_tracer = source_tracer[valid]
    if not np.array_equal(source_tracer, dense_tracer[position]):
        raise RuntimeError("assigned/forFA tracer identity mismatch")
    return dense[position], source_tracer, {
        "source_rows": int(len(assigned)),
        "source_unique_targetids": int(len(assigned_id)),
        "source_duplicate_rows": int(collapsed["duplicate_rows"]),
        "matched_accepted_rows": int(len(position)),
        "bright_rows": int(np.count_nonzero(source_tracer == 0)),
        "faint_rows": int(np.count_nonzero(source_tracer == 1)),
        "assigned_is_targetid_subset_of_dense": True,
    }


def grid_equal(left: dict, right: dict) -> bool:
    return (
        tuple(left["shape"]) == tuple(right["shape"])
        and np.allclose(left["origin_mpc"], right["origin_mpc"], rtol=0, atol=0)
        and float(left["cell_mpc"]) == float(right["cell_mpc"])
    )


def apply_random_support(product: dict, response: dict) -> dict:
    """Mask counts in place and refresh hashes after support application."""
    components = {}
    for cap_name in ("NGC", "SGC"):
        component = product["components"][cap_name]
        response_component = response["components"][cap_name]
        if not grid_equal(component["grid"], response_component["grid"]):
            raise RuntimeError(f"{cap_name} factorial/response grid mismatch")
        path = Path(component["file"])
        outside, supported = {}, {}
        with h5py.File(path, "r+") as counts, h5py.File(
            response_component["file"], "r"
        ) as overlay:
            if int(counts.attrs.get("random_support_applied", 0)) != 1:
                for name in ("bright_counts", "faint_counts"):
                    dataset = counts[name]
                    outside_sum = 0.0
                    supported_sum = 0.0
                    for selection in dataset.iter_chunks():
                        values = np.asarray(dataset[selection], dtype=np.float32)
                        mask = np.asarray(overlay["support_random"][selection], dtype=bool)
                        outside_sum += float(values[~mask].sum(dtype=np.float64))
                        values[~mask] = 0.0
                        supported_sum += float(values.sum(dtype=np.float64))
                        dataset[selection] = values
                    outside[name] = outside_sum
                    supported[name] = supported_sum
                counts.attrs["random_support_applied"] = 1
                counts.attrs["random_support_source"] = response_component["file"]
            else:
                for name in ("bright_counts", "faint_counts"):
                    total = 0.0
                    for selection in counts[name].iter_chunks():
                        total += float(
                            np.asarray(counts[name][selection]).sum(dtype=np.float64)
                        )
                    outside[name] = float("nan")
                    supported[name] = total
        components[cap_name] = {
            **component,
            "file_sha256": sha256(path),
            "random_support_source": response_component["file"],
            "random_support_source_sha256": response_component["file_sha256"],
            "counts_removed_outside_common_support": outside,
            "supported_count_sum": supported,
        }
    return {**product, "components": components, "common_random_support_applied": True}


def build_stage(
    *, phase_root: Path, stage: str, table: np.ndarray, tracer: np.ndarray,
    p3: dict, response: dict, force: bool, source_audit: dict,
) -> dict:
    points = sky_to_points(table["RA"], table["DEC"], table["RSDZ"])
    cap = np.asarray(points[:, 3], dtype=np.uint8)
    raw = write_fields(
        output=phase_root / stage,
        name="counts_cic",
        scheme="cic",
        points=points,
        tracer=tracer,
        cap=cap,
        selected=np.arange(len(points), dtype=np.int64),
        p3=p3,
        chunk=250_000,
        include_bright=True,
        include_faint=True,
        force=force,
    )
    masked = apply_random_support(raw, response)
    manifest = {
        "schema_version": "p11-factorial-count-stage-v1",
        "created_utc": utc_now(),
        "phase": phase_root.name,
        "stage": stage,
        "tracers": ["BGS_BRIGHT", "BGS_FAINT"],
        "truth_or_targets_read": False,
        "common_random_support": True,
        "source_audit": source_audit,
        "counts": masked,
        "pass": bool(masked["pass"] and masked["common_random_support_applied"]),
    }
    path = phase_root / stage / "manifest.json"
    atomic_json(path, manifest)
    return manifest


def nested_count_audit(dense: dict, assigned: dict, p3: dict, response: dict) -> dict:
    gates = {}
    details = {}
    for cap_name in ("NGC", "SGC"):
        dense_path = dense["counts"]["components"][cap_name]["file"]
        assigned_path = assigned["counts"]["components"][cap_name]["file"]
        p3_path = p3["components"][cap_name]["file"]
        response_path = response["components"][cap_name]["file"]
        maximum_assign_minus_dense = -np.inf
        maximum_final_minus_assign = -np.inf
        with h5py.File(dense_path, "r") as d, h5py.File(
            assigned_path, "r"
        ) as a, h5py.File(p3_path, "r") as f, h5py.File(response_path, "r") as r:
            for selection in d["bright_counts"].iter_chunks():
                db = np.asarray(d["bright_counts"][selection], dtype=np.float32)
                df = np.asarray(d["faint_counts"][selection], dtype=np.float32)
                ab = np.asarray(a["bright_counts"][selection], dtype=np.float32)
                af = np.asarray(a["faint_counts"][selection], dtype=np.float32)
                final = np.asarray(f["counts"][selection], dtype=np.float32)
                support = np.asarray(r["support_random"][selection], dtype=bool)
                final[~support] = 0.0
                maximum_assign_minus_dense = max(
                    maximum_assign_minus_dense,
                    float(np.max(ab - db, initial=-np.inf)),
                    float(np.max(af - df, initial=-np.inf)),
                )
                maximum_final_minus_assign = max(
                    maximum_final_minus_assign,
                    float(np.max(final - ab, initial=-np.inf)),
                )
        key = cap_name.lower()
        details[key] = {
            "max_assign_minus_dense": maximum_assign_minus_dense,
            "max_final_bright_minus_assign_bright": maximum_final_minus_assign,
        }
        gates[f"{key}_assigned_counts_nested_in_dense"] = maximum_assign_minus_dense <= 5e-5
        gates[f"{key}_final_bright_nested_in_assigned"] = maximum_final_minus_assign <= 5e-5
    return {"details": details, "gates": gates, "pass": bool(all(gates.values()))}


def build_phase(phase: str, registry: dict, source_contract: dict, force: bool) -> dict:
    if phase == SEALED or phase not in VISIBLE:
        raise PermissionError(f"phase {phase} is not visible")
    phase_root = OUTPUT / phase
    final_path = phase_root / "PHASE_FACTORIAL_VIEW_COUNTS_READY.json"
    if final_path.exists() and not force:
        existing = json.loads(final_path.read_text())
        if existing.get("pass") and not existing.get("sealed_phase_opened"):
            return existing
    phase_root.mkdir(parents=True, exist_ok=True)
    forfa, assigned_path, resolution = paths_for_phase(registry, phase)
    source_row = source_contract["phase_records"][phase]
    if str(forfa) != source_row["sources"]["forfa_targetable"]["path"]:
        raise RuntimeError(f"{phase} forFA source-contract mismatch")
    if str(assigned_path) != source_row["sources"]["alternate_mtl_assigned"]["path"]:
        raise RuntimeError(f"{phase} assignment source-contract mismatch")
    dense_table, dense_tracer, dense_audit = targetable_rows(forfa)
    assigned_table, assigned_tracer, assigned_audit = assigned_subset(
        dense_table, dense_tracer, assigned_path
    )
    p3_path = ROOT / phase / "p3_fields/field_manifest.json"
    response_path = ROOT / phase / "p3b_random_response_v1/manifest.json"
    p3 = json.loads(p3_path.read_text())
    response = json.loads(response_path.read_text())
    source_common = {
        "path_resolution": resolution,
        "forfa": str(forfa),
        "forfa_sha256": source_row["sources"]["forfa_targetable"]["sha256"],
        "assigned": str(assigned_path),
        "assigned_sha256": source_row["sources"]["alternate_mtl_assigned"]["sha256"],
    }
    dense = build_stage(
        phase_root=phase_root, stage="V_dense", table=dense_table,
        tracer=dense_tracer, p3=p3, response=response, force=force,
        source_audit={**source_common, **dense_audit},
    )
    assigned = build_stage(
        phase_root=phase_root, stage="V_assign", table=assigned_table,
        tracer=assigned_tracer, p3=p3, response=response, force=force,
        source_audit={**source_common, **assigned_audit},
    )
    nested = nested_count_audit(dense, assigned, p3, response)
    final_view = {
        "bright": {
            "semantics": "immutable canonical P3 final BGS_BRIGHT counts",
            "components": {
                cap: {"file": p3["components"][cap]["file"], "dataset": "counts"}
                for cap in ("NGC", "SGC")
            },
        },
        "faint": {
            "semantics": (
                "identity reference to supported V_assign FAINT because current mock "
                "C_z=1; not a pointwise Loa final-FAINT product"
            ),
            "components": {
                cap: {
                    "file": assigned["counts"]["components"][cap]["file"],
                    "dataset": "faint_counts",
                }
                for cap in ("NGC", "SGC")
            },
        },
        "response": {cap: response["components"][cap]["file"] for cap in ("NGC", "SGC")},
        "C_z": 1.0,
    }
    report = {
        "schema_version": "p11-phase-factorial-view-counts-ready-v1",
        "created_utc": utc_now(),
        "phase": phase,
        "sealed_phase": SEALED,
        "sealed_phase_opened": False,
        "truth_or_targets_read": False,
        "V_dense": dense,
        "V_assign": assigned,
        "V_final": final_view,
        "nesting_audit": nested,
        "p3_manifest": str(p3_path),
        "p3_manifest_sha256": sha256(p3_path),
        "response_manifest": str(response_path),
        "response_manifest_sha256": sha256(response_path),
        "pass": bool(dense["pass"] and assigned["pass"] and nested["pass"]),
    }
    if not report["pass"]:
        raise RuntimeError(f"{phase} factorial count gates failed: {nested}")
    atomic_json(final_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=VISIBLE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    started = time.time()
    source_contract = json.loads(SOURCE_MARKER.read_text())
    if not source_contract.get("pass") or source_contract.get("sealed_phase_opened"):
        raise RuntimeError("passing factorial source contract is required")
    registry = json.loads(REGISTRY.read_text())
    phases = (args.phase,) if args.phase else VISIBLE
    records = {phase: build_phase(phase, registry, source_contract, args.force) for phase in phases}
    if args.phase is None:
        final = {
            "schema_version": "p11-factorial-view-products-ready-v1",
            "created_utc": utc_now(),
            "visible_phases": list(VISIBLE),
            "sealed_phase": SEALED,
            "sealed_phase_opened": False,
            "truth_or_targets_read": False,
            "phase_manifests": {
                phase: str(OUTPUT / phase / "PHASE_FACTORIAL_VIEW_COUNTS_READY.json")
                for phase in VISIBLE
            },
            "scope": (
                "nested count products and common support; view-specific response/selection "
                "channel transformations remain a separate frozen adapter gate"
            ),
            "elapsed_seconds": time.time() - started,
            "pass": bool(all(row["pass"] for row in records.values())),
        }
        atomic_json(OUTPUT / "FACTORIAL_VIEW_PRODUCTS_READY.json", final)
        print(json.dumps(final, indent=2), flush=True)
    else:
        print(json.dumps(records[args.phase], indent=2), flush=True)


if __name__ == "__main__":
    main()
