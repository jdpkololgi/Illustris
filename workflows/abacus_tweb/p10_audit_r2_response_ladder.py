#!/usr/bin/env python3
"""Audit the observable-view response contract required before P10 R2.

This is deliberately an audit, not an R2 product builder.  It tests whether the
full LSS catalogue fields can support a nested

    V_dense -> V_assign -> V_final

view ladder and whether the assignment-completeness candidates behave like
probabilities.  It never opens ph001 and it does not write
P10_VIEW_LADDER_READY.json; that marker belongs to the later spatial-product
builder once the response semantics have been frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitsio
import numpy as np


VISIBLE_PHASES = ("ph000", "ph002", "ph003", "ph004", "ph005", "ph006")
REQUIRED_FULL = (
    "TARGETID",
    "TRUEZ",
    "Z_not4clus",
    "ZWARN",
    "GOODPRI",
    "GOODHARDLOC",
    "LOCATION_ASSIGNED",
    "TILELOCID_ASSIGNED",
    "FRACZ_TILELOCID",
    "FRAC_TLOBS_TILES",
    "NTILE",
    "PHOTSYS",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def finite_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    good = values[finite]
    if not good.size:
        return {"count": int(values.size), "finite_fraction": 0.0}
    q = np.quantile(good, [0.0, 0.01, 0.5, 0.99, 1.0])
    return {
        "count": int(values.size),
        "finite_fraction": float(finite.mean()),
        "mean": float(good.mean()),
        "std": float(good.std()),
        "min": float(q[0]),
        "p01": float(q[1]),
        "median": float(q[2]),
        "p99": float(q[3]),
        "max": float(q[4]),
    }


def calibration_table(probability: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    probability = np.asarray(probability, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    valid = np.isfinite(probability) & (probability >= 0.0) & (probability <= 1.0)
    p = probability[valid]
    y = observed[valid].astype(np.float64)
    edges = np.linspace(0.0, 1.0, 11)
    index = np.minimum(np.digitize(p, edges[1:-1]), 9)
    bins: list[dict[str, Any]] = []
    ece = 0.0
    for ibin in range(10):
        select = index == ibin
        if not np.any(select):
            continue
        predicted = float(p[select].mean())
        empirical = float(y[select].mean())
        fraction = float(select.mean())
        ece += fraction * abs(predicted - empirical)
        bins.append(
            {
                "lo": float(edges[ibin]),
                "hi": float(edges[ibin + 1]),
                "n": int(select.sum()),
                "predicted_mean": predicted,
                "observed_fraction": empirical,
                "residual": empirical - predicted,
            }
        )
    return {
        "n": int(p.size),
        "observed_fraction": float(y.mean()) if y.size else None,
        "predicted_mean": float(p.mean()) if p.size else None,
        "expected_to_observed_ratio": (
            float(p.sum() / y.sum()) if y.sum() > 0 else None
        ),
        "brier": float(np.mean((p - y) ** 2)) if p.size else None,
        "ece_10bin": float(ece),
        "bins": bins,
    }


def _phase_payload(phase: str, registry_path: str, output_root: str) -> dict[str, Any]:
    registry = json.loads(Path(registry_path).read_text())
    source = registry["mock_phases"][phase]["data"]
    full_meta = source["full"]
    clustering_meta = source["clustering"]
    full_path = Path(full_meta["path"])
    clustering_path = Path(clustering_meta["path"])

    full = fitsio.read(full_path, columns=list(REQUIRED_FULL), ext=1)
    cluster = fitsio.read(
        clustering_path,
        columns=["TARGETID", "WEIGHT_ZFAIL"],
        ext=1,
    )

    targetid = np.asarray(full["TARGETID"], dtype=np.int64)
    cluster_targetid = np.asarray(cluster["TARGETID"], dtype=np.int64)
    eligible = np.asarray(full["GOODPRI"], dtype=bool) & np.asarray(
        full["GOODHARDLOC"], dtype=bool
    )
    location_assigned = np.asarray(full["LOCATION_ASSIGNED"], dtype=bool)
    tileloc_assigned = np.asarray(full["TILELOCID_ASSIGNED"], dtype=bool)
    assigned = eligible & location_assigned
    truez = np.asarray(full["TRUEZ"], dtype=np.float64)
    z_obs = np.asarray(full["Z_not4clus"], dtype=np.float64)
    zwarn = np.asarray(full["ZWARN"], dtype=np.int64)
    good_redshift = assigned & (zwarn == 0) & np.isfinite(z_obs) & (z_obs > 0.0)

    frac_tileloc = np.asarray(full["FRACZ_TILELOCID"], dtype=np.float64)
    frac_tiles = np.asarray(full["FRAC_TLOBS_TILES"], dtype=np.float64)
    c_fibre = frac_tileloc * frac_tiles

    full_unique = np.unique(targetid).size == targetid.size
    cluster_unique = np.unique(cluster_targetid).size == cluster_targetid.size
    final_member = np.isin(targetid, cluster_targetid, assume_unique=full_unique and cluster_unique)

    assignment_mask = eligible & np.isfinite(c_fibre)
    redshift_denominator = assigned
    zfail_weights = np.asarray(cluster["WEIGHT_ZFAIL"], dtype=np.float64)
    zfail_weight_nontrivial = np.isfinite(zfail_weights) & (
        np.abs(zfail_weights - 1.0) > 1.0e-8
    )

    range_gate = bool(
        np.all(np.isfinite(frac_tileloc[eligible]))
        and np.all(np.isfinite(frac_tiles[eligible]))
        and np.all((frac_tileloc[eligible] >= 0.0) & (frac_tileloc[eligible] <= 1.0))
        and np.all((frac_tiles[eligible] >= 0.0) & (frac_tiles[eligible] <= 1.0))
    )
    nesting_gate = bool(
        not np.any(final_member & ~assigned)
        and not np.any(good_redshift & ~assigned)
    )

    result: dict[str, Any] = {
        "schema_version": "p10-r2-response-audit-v1",
        "created_utc": utc_now(),
        "phase": phase,
        "blind_phase_opened": False,
        "sources": {
            "full": {
                "path": str(full_path),
                "registered_sha256": full_meta["sha256"],
                "registered_rows": int(full_meta["rows"]),
            },
            "clustering": {
                "path": str(clustering_path),
                "registered_sha256": clustering_meta["sha256"],
                "registered_rows": int(clustering_meta["rows"]),
            },
        },
        "rows": {
            "full": int(targetid.size),
            "full_unique_targetid": bool(full_unique),
            "eligible_dense_view": int(eligible.sum()),
            "assigned_view": int(assigned.sum()),
            "good_redshift": int(good_redshift.sum()),
            "final_clustering_view": int(final_member.sum()),
            "clustering": int(cluster_targetid.size),
            "clustering_unique_targetid": bool(cluster_unique),
            "location_tileloc_assignment_mismatch": int(
                np.count_nonzero(location_assigned != tileloc_assigned)
            ),
            "final_not_assigned": int(np.count_nonzero(final_member & ~assigned)),
        },
        "view_fractions": {
            "assigned_given_eligible": float(assigned.sum() / max(eligible.sum(), 1)),
            "redshift_success_given_assigned": float(
                good_redshift.sum() / max(redshift_denominator.sum(), 1)
            ),
            "final_given_eligible": float(final_member.sum() / max(eligible.sum(), 1)),
        },
        "candidate_stats": {
            "FRACZ_TILELOCID": finite_stats(frac_tileloc[eligible]),
            "FRAC_TLOBS_TILES": finite_stats(frac_tiles[eligible]),
            "C_fibre_product": finite_stats(c_fibre[eligible]),
            "WEIGHT_ZFAIL_clustering": finite_stats(zfail_weights),
            "WEIGHT_ZFAIL_nontrivial_fraction": float(zfail_weight_nontrivial.mean()),
        },
        "assignment_calibration": {
            "FRACZ_TILELOCID": calibration_table(
                frac_tileloc[assignment_mask], location_assigned[assignment_mask]
            ),
            "FRAC_TLOBS_TILES": calibration_table(
                frac_tiles[assignment_mask], location_assigned[assignment_mask]
            ),
            "product": calibration_table(
                c_fibre[assignment_mask], location_assigned[assignment_mask]
            ),
        },
        "redshift_success_diagnostics": {
            "definition": "assigned & ZWARN==0 & finite(Z_not4clus) & Z_not4clus>0",
            "continuous_mock_C_z_available": False,
            "reason": (
                "FRACZ_TILELOCID and FRAC_TLOBS_TILES are assignment-completeness "
                "quantities; mock full catalogues do not contain Loa mod_success_rate"
            ),
            "success_fraction": float(
                good_redshift.sum() / max(redshift_denominator.sum(), 1)
            ),
            "failure_count": int(np.count_nonzero(redshift_denominator & ~good_redshift)),
        },
        "redshift_range_diagnostics": {
            "truez_0p05_0p6_eligible": int(
                np.count_nonzero(eligible & (truez >= 0.05) & (truez < 0.6))
            ),
            "truez_0p05_0p6_final": int(
                np.count_nonzero(final_member & (truez >= 0.05) & (truez < 0.6))
            ),
        },
        "gates": {
            "registered_row_identity": bool(
                targetid.size == int(full_meta["rows"])
                and cluster_targetid.size == int(clustering_meta["rows"])
            ),
            "candidate_probability_ranges": range_gate,
            "view_nesting": nesting_gate,
            "targetid_unique": bool(full_unique and cluster_unique),
        },
    }
    result["pass"] = bool(all(result["gates"].values()))
    out = Path(output_root) / phase / "response_audit.json"
    atomic_json(out, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        default="configs/p10_response_sources_v1.json",
        help="Tracked response-source registry.",
    )
    parser.add_argument(
        "--output-root",
        default="/pscratch/sd/d/dkololgi/abacus/p10_multiphase/r2_response_audit_v1",
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--phases", nargs="+", default=list(VISIBLE_PHASES))
    args = parser.parse_args()

    phases = tuple(args.phases)
    if "ph001" in phases:
        raise SystemExit("ph001 is sealed and cannot be opened by this audit")
    invalid = sorted(set(phases) - set(VISIBLE_PHASES))
    if invalid:
        raise SystemExit(f"unregistered visible phases: {invalid}")

    registry_path = str(Path(args.registry).resolve())
    output_root = str(Path(args.output_root))
    results: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=min(args.workers, len(phases))) as pool:
        futures = {
            pool.submit(_phase_payload, phase, registry_path, output_root): phase
            for phase in phases
        }
        for future in as_completed(futures):
            phase = futures[future]
            results[phase] = future.result()
            print(f"[{phase}] pass={results[phase]['pass']}", flush=True)

    ordered = [results[p] for p in phases]
    audit_pass = bool(all(item["pass"] for item in ordered))
    continuous_cz = bool(
        all(
            item["redshift_success_diagnostics"]["continuous_mock_C_z_available"]
            for item in ordered
        )
    )
    aggregate = {
        "schema_version": "p10-r2-response-audit-v1",
        "created_utc": utc_now(),
        "phases": list(phases),
        "blind_phase_opened": False,
        "source_registry": {
            "path": registry_path,
            "sha256": sha256(Path(registry_path)),
        },
        "phase_reports": {
            item["phase"]: {
                "path": str(Path(output_root) / item["phase"] / "response_audit.json"),
                "sha256": sha256(Path(output_root) / item["phase"] / "response_audit.json"),
                "pass": item["pass"],
            }
            for item in ordered
        },
        "summary": {
            "source_and_nesting_audit_pass": audit_pass,
            "continuous_mock_C_z_available": continuous_cz,
            "r2_training_ready": bool(audit_pass and continuous_cz),
            "view_ladder_marker_written": False,
            "interpretation": (
                "The audit can validate nested catalogue views and assignment response. "
                "R2 remains blocked unless a mock-to-Loa compatible continuous C_z is "
                "identified or the plan explicitly freezes C_z=1 as the mock contract."
            ),
        },
    }
    aggregate_path = Path(output_root) / "R2_RESPONSE_AUDIT.json"
    atomic_json(aggregate_path, aggregate)
    print(aggregate_path)
    print(json.dumps(aggregate["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
