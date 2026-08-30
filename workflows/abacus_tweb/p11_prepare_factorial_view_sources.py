#!/usr/bin/env python3
"""Freeze the lightweight source contract for P11 factorial observation views.

This command reads only tracked/small manifests and filesystem metadata.  It is
safe on a login node and intentionally does not stream FITS rows or materialize
voxel fields.  The output distinguishes source readiness from the later heavy
catalogue/field build.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256


VISIBLE = ("ph002", "ph003", "ph004", "ph005", "ph006")
SEALED = "ph001"
ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase")
CONFIG = REPO_ROOT / "configs/p11_factorial_views_v1.json"
SOURCE_AUDIT = REPO_ROOT / "docs/evidence/p10/multitracer_source_audit_20260820.json"
R2_ROOT = ROOT / "r2_response_audit_v1"
MULTITRACER_ROOT = ROOT / "multitracer/v1"
OUTPUT = ROOT / "p11_factorial_views_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def checked_manifest(path: Path, *, require_pass: bool = True) -> tuple[dict, dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    gate_values = payload.get("gates", {})
    inferred_pass = payload.get(
        "pass",
        payload.get(
            "all_visible_phases_pass",
            bool(gate_values) and all(bool(value) for value in gate_values.values()),
        ),
    )
    if require_pass and not inferred_pass:
        raise RuntimeError(f"manifest is not passing: {path}")
    if payload.get("sealed_phase_opened") or payload.get("blind_phase_opened"):
        raise RuntimeError(f"manifest opened the sealed phase: {path}")
    return payload, {"path": str(path), "sha256": sha256(path)}


def phase_record(phase: str, source_audit: dict) -> dict:
    if phase == SEALED:
        raise PermissionError("ph001 is sealed")
    source = source_audit["phases"][phase]
    r2, r2_ref = checked_manifest(R2_ROOT / phase / "response_audit.json")
    p1, p1_ref = checked_manifest(ROOT / phase / "p1_canonical/manifest.json", require_pass=False)
    p3, p3_ref = checked_manifest(ROOT / phase / "p3_fields/field_manifest.json")
    random_response, random_ref = checked_manifest(
        ROOT / phase / "p3b_random_response_v1/manifest.json"
    )
    tracer, tracer_ref = checked_manifest(
        MULTITRACER_ROOT / "phases" / phase / "PHASE_MULTITRACER_VIEWS_READY.json"
    )
    if source["forfa"] != r2["sources"]["full"]["path"]:
        # The R2 full LSS catalogue and forFA are different products by design;
        # make the distinction explicit rather than silently equating them.
        full_distinct_from_forfa = True
    else:
        full_distinct_from_forfa = False
    for path in (Path(source["forfa"]), Path(source["assigned"]), Path(p1["parent"])):
        if not path.exists():
            raise FileNotFoundError(path)
    materialized = {
        "V_dense": {"bright": False, "faint": False},
        "V_assign": {"bright": False, "faint": True},
        # The existing FAINT product is assignment-stage context.  It must not
        # silently be relabelled as a final-quality Loa catalogue.  The heavy
        # builder may register an identity reference only under audited C_z=1.
        "V_final": {"bright": True, "faint": False},
    }
    return {
        "phase": phase,
        "sources": {
            "forfa_targetable": {
                "path": source["forfa"],
                "sha256": source["forfa_sha256"],
            },
            "alternate_mtl_assigned": {
                "path": source["assigned"],
                "sha256": source["assigned_sha256"],
            },
            "final_lss_full": r2["sources"]["full"],
            "final_lss_clustering": r2["sources"]["clustering"],
            "canonical_final_bright_parent": {
                "path": p1["parent"],
                "sha256": p1["parent_sha256"],
            },
        },
        "manifests": {
            "r2_response": r2_ref,
            "p1_final_bright": p1_ref,
            "p3_final_bright": p3_ref,
            "random_response": random_ref,
            "assigned_faint": tracer_ref,
        },
        "materialized_view_counts": materialized,
        "full_lss_distinct_from_forfa": full_distinct_from_forfa,
        "continuous_mock_C_z_available": bool(
            r2["redshift_success_diagnostics"]["continuous_mock_C_z_available"]
        ),
        "pass": bool(
            all(bool(value) for value in p3.get("gates", {}).values())
            and random_response.get("pass")
            and tracer.get("pass")
            and not r2["redshift_success_diagnostics"]["continuous_mock_C_z_available"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--source-audit", type=Path, default=SOURCE_AUDIT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    source_audit = json.loads(args.source_audit.read_text())
    if not source_audit.get("all_visible_phases_pass") or source_audit.get("sealed_phase_opened"):
        raise RuntimeError("visible-phase multitracer source audit is not valid")
    audited_visible = tuple(source_audit["visible_phases"])
    if not set(VISIBLE).issubset(audited_visible):
        raise RuntimeError("source audit does not contain every factorial-contract phase")
    contract_visible = tuple(
        config["phase_split"]["training"]
        + config["phase_split"]["validation_and_selection"]
    )
    if contract_visible != VISIBLE:
        raise RuntimeError("config phase split differs from the executable factorial contract")
    phases = {phase: phase_record(phase, source_audit) for phase in VISIBLE}
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "p11-factorial-view-sources-ready-v1",
        "created_utc": utc_now(),
        "config": str(args.config),
        "config_sha256": sha256(args.config),
        "source_audit": str(args.source_audit),
        "source_audit_sha256": sha256(args.source_audit),
        "visible_phases": list(VISIBLE),
        "sealed_phase": SEALED,
        "sealed_phase_opened": False,
        "phase_records": phases,
        "materialization_missing": {
            "V_dense": ["bright_counts", "faint_counts"],
            "V_assign": ["bright_counts"],
            "V_final": ["faint_Cz1_identity_reference"],
        },
        "heavy_materialization_complete": False,
        "safe_login_scope_complete": True,
        "pass": bool(all(row["pass"] for row in phases.values())),
    }
    atomic_json(output / "FACTORIAL_VIEW_SOURCES_READY.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
