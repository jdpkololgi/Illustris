#!/usr/bin/env python3
"""Report normalized ph000--ph006 product status and scientific phase roles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


PHASES = tuple(f"ph{i:03d}" for i in range(0, 7))
TRAINING_PHASES = ("ph000", "ph002", "ph003", "ph004", "ph005")


def record(root: Path, phase: str) -> dict:
    p = root / phase
    prefix = f"{phase}_bgs_bright_full_delaunay"
    blind = phase == "ph001"
    reference = phase == "ph000"
    checks = {
        "density": p / f"targets/density/AbacusSummit_base_c000_{phase}_z0.200_ngrid2048_ab10_tsc_counts.manifest.json",
        "tweb": p / "targets/tweb/backend_optimized_ngrid_2048_rsmooth_7/TWEB_COMPLETE.json",
        "observed": p / (f"catalogues/blind_observed/{phase}_bgs_bright_full_observed_geometry.fits.complete.json"
                           if blind else f"catalogues/observed/{phase}_bgs_bright_full_observed_with_tweb.fits.complete.json"),
        "p1": p / "p1_canonical/CATALOGUE_COMPLETE.json",
        "p1_manifest": p / "p1_canonical/manifest.json",
        "p2_ngc": p / "p2_graph/caps/ngc/CAP_GRAPH_COMPLETE.json",
        "p2_sgc": p / "p2_graph/caps/sgc/CAP_GRAPH_COMPLETE.json",
        "p2_graph": p / "p2_graph/GRAPH_COMPLETE.json",
        "p2_metrics": p / f"p2_graph/{prefix}_cugraph_gnn_metadata.json",
        "p2": p / "p2_graph/P2_COMPLETE.json",
        "p3": p / "p3_fields/FIELD_COMPLETE",
        "p4": p / "p4_patches/PATCH_MANIFEST_COMPLETE",
        "phase_complete": p / (
            "REFERENCE_PHASE_COMPLETE.json" if reference
            else "BLIND_INPUT_COMPLETE.json" if blind
            else "PHASE_COMPLETE.json"
        ),
    }
    if blind:
        checks["density"] = Path("/forbidden/blind-density")
        checks["tweb"] = Path("/forbidden/blind-tweb")
    status = {name: path.is_file() and path.stat().st_size > 0 for name, path in checks.items()}
    status["density"] = "forbidden" if blind else status["density"]
    status["tweb"] = "forbidden" if blind else status["tweb"]
    if reference:
        # ph000 predates the cap-parallel graph builder.  Its canonical global
        # graph is already cap-disconnected, so cap build markers are not
        # applicable rather than missing.
        status["p2_ngc"] = "legacy_global_graph"
        status["p2_sgc"] = "legacy_global_graph"
    role = (
        "training_development_reference" if reference else
        "sealed_blind" if blind else
        "validation_and_selection" if phase == "ph006" else
        "training"
    )
    return {"phase": phase, "role": role, "blind": blind,
            "eligible_for_training": phase in set(TRAINING_PHASES),
            "status": status,
            "paths": {name: str(path) for name, path in checks.items()}}


def readiness(root: Path, records: dict[str, dict]) -> dict:
    contract_root = root / "training_contract"
    loader_marker = contract_root / "TRAINING_LOADER_READY.json"
    response_marker = contract_root / "P10_RESPONSE_SOURCES_READY.json"
    blind_marker = contract_root / "P10_BLIND_PROTOCOL_FROZEN.json"
    training_products = all(
        records[phase]["status"]["phase_complete"] for phase in TRAINING_PHASES
    )
    validation_products = bool(records["ph006"]["status"]["phase_complete"])
    blind_inputs = bool(records["ph001"]["status"]["phase_complete"])
    reference_normalized = bool(records["ph000"]["status"]["phase_complete"])
    product_gate = training_products and validation_products
    loader_ready = loader_marker.is_file() and loader_marker.stat().st_size > 0
    response_ready = response_marker.is_file() and response_marker.stat().st_size > 0
    blind_frozen = blind_marker.is_file() and blind_marker.stat().st_size > 0
    return {
        "training_phase_products_complete": training_products,
        "validation_phase_products_complete": validation_products,
        "sealed_blind_inputs_complete": blind_inputs,
        "ph000_reference_normalized": reference_normalized,
        "p1_p4_products_ready": product_gate,
        "phase_balanced_loader_canary_complete": loader_ready,
        "response_source_contract_complete": response_ready,
        "blind_evaluation_protocol_frozen": blind_frozen,
        "ready_to_launch_deterministic_training": (
            product_gate and loader_ready and response_ready and blind_frozen
        ),
        "loader_marker": str(loader_marker),
        "response_marker": str(response_marker),
        "blind_marker": str(blind_marker),
        "interpretation": (
            "P1-P4, loader, response-source and blind-protocol gates are distinct; "
            "ph000 is training-eligible but its scores are reference-only; ph001 remains sealed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase"))
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    records = {phase: record(args.root, phase) for phase in PHASES}
    payload = {"schema_version": "p10-multiphase-status-v4",
               "phases": records, "readiness": readiness(args.root, records)}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
