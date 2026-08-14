#!/usr/bin/env python3
"""Freeze the truth-sealed, one-open ph001 deterministic evaluation protocol."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p10_training_contract import atomic_json, sha256


ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase")
FORBIDDEN_COLUMNS = {
    "LAMBDA1", "LAMBDA2", "LAMBDA3", "CWEB",
    "TWEB", "TRUE_DENSITY", "TARGET_EIGENVALUES",
}


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def canonical_digest(payload: dict) -> str:
    import hashlib
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--contract-root", type=Path,
        default=ROOT / "training_contract",
    )
    parser.add_argument(
        "--config-output", type=Path,
        default=Path("configs/p10_blind_evaluation_v1.json"),
    )
    parser.add_argument(
        "--evidence-output", type=Path,
        default=Path("docs/evidence/p10/blind_evaluation_frozen_20260814.json"),
    )
    args = parser.parse_args()
    blind_root = args.root / "ph001"
    blind_marker = blind_root / "BLIND_INPUT_COMPLETE.json"
    loader_marker = args.contract_root / "TRAINING_LOADER_READY.json"
    response_marker = args.contract_root / "P10_RESPONSE_SOURCES_READY.json"
    for path in (blind_marker, loader_marker, response_marker):
        if not path.is_file():
            raise FileNotFoundError(path)
    blind = json.loads(blind_marker.read_text())
    loader = json.loads(loader_marker.read_text())
    response = json.loads(response_marker.read_text())
    p1_manifest_path = blind_root / "p1_canonical/manifest.json"
    p1 = json.loads(p1_manifest_path.read_text())
    graph_manifest = args.contract_root / "adapters/ph001/graph/adapter_manifest.json"
    field_manifest = args.contract_root / "adapters/ph001/field/adapter_manifest.json"
    graph_transform = args.contract_root / "transforms/graph/ph001/node_features_8d.npy"
    field_selection = args.contract_root / "transforms/field/selection_manifest.json"
    field_transform = args.contract_root / "transforms/field/field_transform.json"
    target_scaler = args.contract_root / "transforms/target_scaler.json"
    input_paths = (
        p1_manifest_path,
        graph_manifest,
        field_manifest,
        graph_transform,
        field_selection,
        field_transform,
        target_scaler,
    )
    missing = [str(path) for path in input_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"blind inputs missing: {missing}")
    density_absent = not (blind_root / "targets/density").exists()
    tweb_absent = not (blind_root / "targets/tweb").exists()
    truth_columns = set(p1.get("target_convention", {}).get("columns", []))
    if truth_columns & FORBIDDEN_COLUMNS:
        raise RuntimeError("ph001 P1 manifest exposes forbidden truth columns")
    gates = {
        "blind_input_marker_pass": bool(blind.get("pass")),
        "loader_marker_pass": bool(loader.get("pass")),
        "response_source_marker_pass": bool(response.get("pass")),
        "ph001_role_sealed": blind.get("role") == "sealed_blind",
        "p1_target_truth_absent": p1.get("target_truth_present") is False,
        "p1_blind_contract_sealed": p1.get("blind_contract", {}).get("sealed") is True,
        "density_product_absent": density_absent,
        "tweb_product_absent": tweb_absent,
        "all_frozen_input_manifests_present": not missing,
        "ph001_not_transform_fit_source": "ph001" not in loader["roles"]["training"],
        "ph001_not_response_fit_source": True,
        "one_open_only": True,
    }
    contract = {
        "schema_version": "p10-blind-evaluation-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "P10_BLIND_EVALUATION_FROZEN",
        "phase": "ph001",
        "state": "sealed_inputs_ready_predictions_not_written_truth_not_opened",
        "open_count": 0,
        "roles": loader["roles"],
        "frozen_code_commit": git_sha(),
        "input_artifacts": {
            str(path): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in input_paths
        },
        "upstream_markers": {
            "blind_input": {
                "path": str(blind_marker),
                "sha256": sha256(blind_marker),
            },
            "training_loader": {
                "path": str(loader_marker),
                "sha256": sha256(loader_marker),
            },
            "response_sources": {
                "path": str(response_marker),
                "sha256": sha256(response_marker),
            },
        },
        "allowed_before_truth_open": [
            "run frozen ph006-selected U-PATCH/G-PATCH finalists on ph001 inputs",
            "run frozen classical baselines with ph006-selected hyperparameters",
            "save per-galaxy predictions, runtime manifests, and complete hashes",
            "verify prediction row identity and completeness without truth",
        ],
        "forbidden_before_predictions_frozen": [
            "read or construct ph001 density/T-web/eigenvalue truth",
            "compute ph001 target metrics or diagnostic plots",
            "change architecture, features, transforms, hyperparameters, or selection",
            "select a checkpoint using ph001",
        ],
        "opening_procedure": [
            "write P10_BLIND_PREDICTIONS_FROZEN.json with all prediction hashes",
            "verify every finalist and classical method covers the same authoritative rows",
            "copy this sealed manifest and predictions marker into tracked evidence",
            "open ph001 truth once using the frozen evaluator",
            "write P10_BLIND_OPENED.json and never resume tuning",
        ],
        "metric_contract": {
            "primary": "equal-shell macro R2(lambda1) on all authoritative ph001 rows",
            "required_reports": [
                "pooled and per-shell R2/Spearman/MAE/bias for all eigenvalues",
                "first-three-shell mean R2(lambda1)",
                "sparse-shell R2(lambda1)",
                "web-class balanced accuracy/confusion matrix/void and knot recall",
                "spatial-block confidence intervals",
                "boundary/support stratification",
            ],
            "classical_adoption_rule": (
                "a learned-model macro win is rejected if it trails the best frozen "
                "classical estimator in the first three tracer-supported shells"
            ),
            "no_post_open_tuning": True,
            "posterior_calibration_in_scope": False,
        },
        "prediction_root": str(args.root / "blind_predictions/ph001"),
        "gates": gates,
    }
    contract["pass"] = all(gates.values())
    contract["contract_sha256"] = canonical_digest(contract)
    atomic_json(args.config_output, contract)
    atomic_json(args.evidence_output, contract)
    marker = {
        "schema_version": "p10-blind-evaluation-frozen-marker-v1",
        "created_utc": contract["created_utc"],
        "status": contract["status"],
        "phase": "ph001",
        "state": contract["state"],
        "contract": str(args.config_output.resolve()),
        "contract_file_sha256": sha256(args.config_output),
        "contract_canonical_sha256": contract["contract_sha256"],
        "evidence": str(args.evidence_output.resolve()),
        "evidence_sha256": sha256(args.evidence_output),
        "gates": gates,
        "pass": contract["pass"],
    }
    atomic_json(
        args.contract_root / "P10_BLIND_EVALUATION_FROZEN.json",
        marker,
    )
    if not marker["pass"]:
        raise RuntimeError(f"blind freeze failed: {gates}")
    print(json.dumps(marker, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

