#!/usr/bin/env python3
"""Audit multi-phase BGS_FAINT context sources without opening ph001.

The official DA2 SecondGen products contain targetable and fibre-assigned BGS
rows for every visible P10 phase even though the existing P10 representation
tree retains only final BGS_BRIGHT galaxies.  This audit counts exact unique
Bright/Faint target IDs at the targetable and assigned stages and verifies that
assigned Faint IDs can recover RSD redshifts from the phase-matched forFA table.
It creates no model input and reads no T-web truth.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import fitsio
import numpy as np
from desitarget.targetmask import bgs_mask

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256


VISIBLE = ("ph000", "ph002", "ph003", "ph004", "ph005", "ph006")
MOCK = {phase: int(phase[-3:]) for phase in VISIBLE}
ROOT = Path("/global/cfs/cdirs/desi/survey/catalogs/DA2/mocks/SecondGenMocks/AbacusSummitBGS_v2")
OUTPUT = Path(
    "/pscratch/sd/d/dkololgi/abacus/p10_multiphase/multitracer/source_audit_v1.json"
)
BRIGHT_BITS = int(
    bgs_mask.BGS_BRIGHT | bgs_mask.BGS_BRIGHT_NORTH | bgs_mask.BGS_BRIGHT_SOUTH
)
FAINT_BITS = int(
    bgs_mask.BGS_FAINT
    | bgs_mask.BGS_FAINT_HIP
    | bgs_mask.BGS_FAINT_NORTH
    | bgs_mask.BGS_FAINT_SOUTH
)


def paths_for_phase(phase: str) -> tuple[Path, Path]:
    mock = MOCK[phase]
    return (
        ROOT / f"forFA{mock}_nomask.fits",
        ROOT / f"altmtl{mock}/fba{mock}/datcomb_brightwdup.fits",
    )


def _read(path: Path, columns: tuple[str, ...]) -> np.ndarray:
    with fitsio.FITS(str(path), "r") as handle:
        return handle[1].read(columns=list(columns))


def target_counts(targetid: np.ndarray, bits: np.ndarray) -> dict:
    order = np.argsort(targetid, kind="mergesort")
    ordered_id = np.asarray(targetid[order], dtype=np.int64)
    ordered_bits = np.asarray(bits[order], dtype=np.int64)
    starts = np.r_[0, np.flatnonzero(ordered_id[1:] != ordered_id[:-1]) + 1]
    unique_id = ordered_id[starts]
    unique_bits = np.bitwise_or.reduceat(ordered_bits, starts)
    return {
        "targetid": unique_id,
        "bright": (unique_bits & BRIGHT_BITS) != 0,
        "faint": (unique_bits & FAINT_BITS) != 0,
        "duplicate_rows": int(len(targetid) - len(unique_id)),
    }


def audit_phase(phase: str) -> dict:
    forfa_path, assigned_path = paths_for_phase(phase)
    if not forfa_path.exists() or not assigned_path.exists():
        raise FileNotFoundError(f"missing {phase} source: {forfa_path}, {assigned_path}")
    forfa = _read(forfa_path, ("TARGETID", "BGS_TARGET", "RSDZ", "TRUEZ"))
    assigned = _read(assigned_path, ("TARGETID", "BGS_TARGET"))
    targetable = target_counts(forfa["TARGETID"], forfa["BGS_TARGET"])
    observed = target_counts(assigned["TARGETID"], assigned["BGS_TARGET"])
    assigned_faint_id = observed["targetid"][observed["faint"]]
    order = np.argsort(np.asarray(forfa["TARGETID"], dtype=np.int64))
    sorted_id = np.asarray(forfa["TARGETID"][order], dtype=np.int64)
    lookup = np.searchsorted(sorted_id, assigned_faint_id)
    matched = (lookup < len(sorted_id)) & (
        sorted_id[np.minimum(lookup, len(sorted_id) - 1)] == assigned_faint_id
    )
    matched_rows = order[lookup[matched]]
    rsdz = np.asarray(forfa["RSDZ"][matched_rows], dtype=np.float64)
    truez = np.asarray(forfa["TRUEZ"][matched_rows], dtype=np.float64)
    report = {
        "phase": phase,
        "mock": MOCK[phase],
        "forfa": str(forfa_path),
        "forfa_sha256": sha256(forfa_path),
        "assigned": str(assigned_path),
        "assigned_sha256": sha256(assigned_path),
        "targetable_unique": int(len(targetable["targetid"])),
        "targetable_bright_unique": int(targetable["bright"].sum()),
        "targetable_faint_unique": int(targetable["faint"].sum()),
        "assigned_unique": int(len(observed["targetid"])),
        "assigned_bright_unique": int(observed["bright"].sum()),
        "assigned_faint_unique": int(observed["faint"].sum()),
        "assigned_duplicate_rows": observed["duplicate_rows"],
        "assigned_faint_matched_to_forfa": int(matched.sum()),
        "assigned_faint_match_fraction": float(matched.mean()) if len(matched) else 0.0,
        "matched_faint_finite_positive_rsdz": int(
            np.count_nonzero(np.isfinite(rsdz) & (rsdz > 0))
        ),
        "matched_faint_finite_positive_truez": int(
            np.count_nonzero(np.isfinite(truez) & (truez > 0))
        ),
    }
    report["pass"] = bool(
        report["targetable_faint_unique"] > 0
        and report["assigned_faint_unique"] > 0
        and report["assigned_faint_match_fraction"] == 1.0
        and report["matched_faint_finite_positive_rsdz"]
        == report["assigned_faint_unique"]
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        existing = json.loads(args.output.read_text())
        if (
            existing.get("schema_version") == "p10-multitracer-source-audit-v1"
            and existing.get("all_visible_phases_pass") is True
            and existing.get("sealed_phase_opened") is False
        ):
            print(json.dumps(existing, indent=2), flush=True)
            return
    phases = {phase: audit_phase(phase) for phase in VISIBLE}
    result = {
        "schema_version": "p10-multitracer-source-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "visible_phases": list(VISIBLE),
        "sealed_phase_opened": False,
        "scope": "source feasibility only; no T-web truth and no model input written",
        "phases": phases,
        "all_visible_phases_pass": bool(all(row["pass"] for row in phases.values())),
        "next_product": (
            "phase-matched BRIGHT-target/FAINT-context Proxy and cap+redshift-stratified "
            "angular-scramble Null views"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
