#!/usr/bin/env python3
"""Report normalized ph000--ph006 product status and scientific phase roles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


PHASES = tuple(f"ph{i:03d}" for i in range(0, 7))


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
        "development_reference" if reference else
        "sealed_blind" if blind else
        "validation_and_selection" if phase == "ph006" else
        "training"
    )
    return {"phase": phase, "role": role, "blind": blind,
            "eligible_for_training": phase in {"ph002", "ph003", "ph004", "ph005"},
            "status": status,
            "paths": {name: str(path) for name, path in checks.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase"))
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    payload = {"schema_version": "p10-multiphase-status-v2",
               "phases": {phase: record(args.root, phase) for phase in PHASES}}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
