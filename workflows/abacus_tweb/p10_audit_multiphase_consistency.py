#!/usr/bin/env python3
"""Write one cross-phase physics/representation readiness audit for P10.

Exact contracts are gates. Galaxy counts, shell populations, and graph degree are
reported relative to ph000 but are not required to be identical across independent
cosmic realizations.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
from typing import Any

from p10_phase_assets import DEFAULT_REGISTRY, load_registry, sha256_file
from p10_validate_phase_products import phase_paths


PHASES = ("ph001", "ph002", "ph003", "ph004", "ph005")
PH000_P1 = Path("/pscratch/sd/d/dkololgi/abacus/p1b_full_footprint/manifest.json")
PH000_UNION = Path("/pscratch/sd/d/dkololgi/abacus/p2b_full_footprint/p2b_union_manifest.json")
PH000_CONVENTION = Path("docs/evidence/p10/ph002_convention_parity.json")


class ConsistencyError(RuntimeError):
    pass


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--phases", nargs="+", default=list(PHASES))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    registry = load_registry(args.registry)
    ph000_p1 = json.loads(PH000_P1.read_text())
    ph000_union = json.loads(PH000_UNION.read_text())
    ph000_convention = json.loads(PH000_CONVENTION.read_text())
    reference_context = int(ph000_p1["counts"]["context"])
    reference_mean_degree = (
        2.0 * int(ph000_union["counts"]["union_pairs_context"]) / reference_context
    )

    records: dict[str, Any] = {}
    target_contracts: list[dict[str, Any]] = []
    field_channel_orders: list[list[str]] = []
    patch_unit_contracts: list[dict[str, Any]] = []
    particle_counts: list[int] = []
    tweb_sizes: list[int] = []
    for phase in args.phases:
        paths = phase_paths(registry, phase)
        blind = registry["phases"][phase]["role"] == "sealed_blind"
        terminal = paths["root"] / ("BLIND_INPUT_COMPLETE.json" if blind else "PHASE_COMPLETE.json")
        required = {
            "terminal": terminal,
            "p1_marker": paths["p1"] / "CATALOGUE_COMPLETE.json",
            "p1_manifest": paths["p1"] / "manifest.json",
            "p2": paths["p2_complete"],
            "p3": paths["p3"] / "field_manifest.json",
            "p4": paths["p4"] / "p4_validation.json",
        }
        if not blind:
            required["catalogue_field_target_closure"] = paths["catalogue_field_target_closure"]
        missing = [str(path) for path in required.values() if not path.is_file()]
        if missing:
            raise ConsistencyError(f"{phase} incomplete: {missing}")
        completion = json.loads(terminal.read_text())
        p1 = json.loads(required["p1_marker"].read_text())
        p1_manifest = json.loads(required["p1_manifest"].read_text())
        p2 = json.loads(required["p2"].read_text())
        p3 = json.loads(required["p3"].read_text())
        p4 = json.loads(required["p4"].read_text())
        context = int(p1["counts"]["context"])
        mean_degree = 2.0 * int(p2["counts"]["union_context_pairs"]) / context
        shell_counts = {name: int(value["all"]) for name, value in p1["counts"]["by_shell"].items()}
        target_contracts.append(p1["target_contract"])
        field_channel_orders.append(p3["channel_order"])
        patch_unit_contracts.append(p4["unit_contract"])
        truth: dict[str, Any] | None = None
        closure_summary: dict[str, Any] | None = None
        if not blind:
            density = json.loads(paths["density"].read_text())
            tweb = json.loads(paths["tweb"].read_text())
            closure = json.loads(required["catalogue_field_target_closure"].read_text())
            particle_counts.append(int(density["build"]["particle_count"]))
            tweb_sizes.append(int(tweb["outputs"]["total_bytes"]))
            truth = {
                "particle_count": int(density["build"]["particle_count"]),
                "relative_count_error": float(density["build"]["relative_count_error"]),
                "processed_file_count": int(density["build"]["processed_file_count"]),
                "tweb_rank_count": int(tweb["outputs"]["rank_count"]),
                "tweb_x_coverage": tweb["outputs"]["x_coverage"],
                "tweb_total_bytes": int(tweb["outputs"]["total_bytes"]),
            }
            closure_summary = {
                "pass": bool(closure["pass"]),
                "cweb_mismatch_rows": int(
                    closure["host_target_consistency"]["cweb_mismatch_rows"]),
                "caps": {
                    cap: {
                        "trace_spearman": float(report["pooled_trace_spearman"]),
                        "shuffle_separation": float(report["pooled_shuffle_separation"]),
                        "cic_max_abs_difference": float(report["cic_max_abs_difference"]),
                    }
                    for cap, report in closure["caps"].items()
                },
            }
        records[phase] = {
            "role": registry["phases"][phase]["role"],
            "terminal_marker": str(terminal),
            "terminal_sha256": sha256_file(terminal),
            "terminal_pass": bool(completion["pass"]),
            "catalogue_id": p1_manifest["catalogue_id"],
            "target_truth_present": bool(p1_manifest["target_truth_present"]),
            "observer_geometry": p1["geometry"],
            "exact_physics_gates": completion["exact_physics_gates"],
            "counts": {
                "parent": int(p1["counts"]["total"]),
                "context": context,
                "active": int(p1["counts"]["active"]),
                "shell_active": shell_counts,
                "context_ratio_to_ph000": context / reference_context,
            },
            "graph": {
                "delaunay_edges": int(p2["counts"]["delaunay_edges"]),
                "union_context_pairs": int(p2["counts"]["union_context_pairs"]),
                "union_mean_degree": mean_degree,
                "mean_degree_ratio_to_ph000": mean_degree / reference_mean_degree,
                "cross_cap_edges": int(p2["counts"]["cross_cap_edges"]),
            },
            "p3_pass": all(bool(value) for value in p3["gates"].values()),
            "p3_channel_order": p3["channel_order"],
            "p4_pass": bool(p4["pass"]),
            "p4_unit_contract": p4["unit_contract"],
            "truth": truth,
            "catalogue_field_target_closure": closure_summary,
        }

    first_contract = target_contracts[0]
    truth_records = [
        record["truth"] for record in records.values() if record["truth"] is not None
    ]
    gates = {
        "all_terminal_markers_pass": all(record["terminal_pass"] for record in records.values()),
        "one_target_contract_all_phases": all(contract == first_contract for contract in target_contracts),
        "target_contract_matches_registry": first_contract == registry["target_contract"],
        "ph000_target_convention_parity_previously_verified": bool(
            ph000_convention["verified"]
            and ph000_convention["target_metadata"]["metadata_equal"]
        ),
        "phase_not_exposed_as_model_input": not bool(first_contract["phase_is_model_input"]),
        "all_exact_physics_gates_pass": all(
            all(bool(value) for value in record["exact_physics_gates"].values())
            for record in records.values()
        ),
        "observer_frame_is_planck18_mpc": all(
            record["observer_geometry"]["cosmology"] == "Astropy Planck18"
            and record["observer_geometry"]["units"] == "Mpc"
            for record in records.values()
        ),
        "one_field_channel_contract_all_phases": all(
            channels == field_channel_orders[0] for channels in field_channel_orders
        ),
        "one_patch_unit_contract_all_phases": all(
            contract == patch_unit_contracts[0] for contract in patch_unit_contracts
        ),
        "all_p3_gates_pass": all(record["p3_pass"] for record in records.values()),
        "all_p4_gates_pass": all(record["p4_pass"] for record in records.values()),
        "all_caps_disconnected": all(record["graph"]["cross_cap_edges"] == 0 for record in records.values()),
        "all_shells_nonempty": all(
            all(count > 0 for count in record["counts"]["shell_active"].values())
            for record in records.values()
        ),
        "context_counts_plausible_vs_ph000": all(
            0.5 < record["counts"]["context_ratio_to_ph000"] < 1.5 for record in records.values()
        ),
        "graph_degrees_finite_positive": all(
            math.isfinite(record["graph"]["union_mean_degree"])
            and record["graph"]["union_mean_degree"] > 0 for record in records.values()
        ),
        "graph_degrees_plausible_vs_ph000": all(
            0.5 < record["graph"]["mean_degree_ratio_to_ph000"] < 1.5
            for record in records.values()
        ),
        "sealed_phase_has_no_truth": (
            not records["ph001"]["target_truth_present"] and records["ph001"]["truth"] is None
        ),
        "training_phases_have_truth": all(
            records[phase]["target_truth_present"] and records[phase]["truth"] is not None
            for phase in args.phases if phase != "ph001"
        ),
        "training_phase_catalogue_field_target_closure_pass": all(
            records[phase]["catalogue_field_target_closure"] is not None
            and records[phase]["catalogue_field_target_closure"]["pass"]
            and records[phase]["catalogue_field_target_closure"]["cweb_mismatch_rows"] == 0
            for phase in args.phases if phase != "ph001"
        ),
        "uniform_particle_count": len(set(particle_counts)) == 1,
        "all_density_counts_conserved_below_1e_6": all(
            truth["relative_count_error"] < 1.0e-6 for truth in truth_records
        ),
        "all_136_particle_slabs_processed": all(
            truth["processed_file_count"] == 136 for truth in truth_records
        ),
        "all_tweb_layouts_have_16_ranks": all(
            truth["tweb_rank_count"] == 16 for truth in truth_records
        ),
        "uniform_tweb_layout_size": len(set(tweb_sizes)) == 1,
    }
    payload = {
        "schema_version": "p10-multiphase-consistency-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reference": {
            "phase": "ph000",
            "p1": str(PH000_P1),
            "p1_sha256": sha256_file(PH000_P1),
            "union": str(PH000_UNION),
            "union_sha256": sha256_file(PH000_UNION),
            "target_convention_parity": str(PH000_CONVENTION),
            "target_convention_parity_sha256": sha256_file(PH000_CONVENTION),
            "context_count": reference_context,
            "union_mean_degree": reference_mean_degree,
        },
        "phases": records,
        "gates": gates,
        "pass": all(gates.values()),
        "interpretation": (
            "exact physics, units, representation schemas and sealed-blind rules are gates; "
            "counts and graph degree are diagnostics because independent phases contain cosmic variance"
        ),
    }
    if not payload["pass"]:
        raise ConsistencyError(f"cross-phase gates failed: {gates}")
    atomic_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (ConsistencyError, OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
