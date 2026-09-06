#!/usr/bin/env python3
"""Frozen FMPE-versus-Gaussian joint energy score on opened ph001 audit rows.

The decisive score is computed from samples in physical ordered-eigenvalue
coordinates.  It therefore evaluates the same rejection-truncated FMPE
distribution that produced the frozen 512 production draws and does not rely on
an unnormalised ``DirectPosterior.log_prob``.  Lower energy score is better; the
registered comparison is Gaussian minus FMPE, so a positive paired interval
favours FMPE.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np

from workflows.abacus_tweb.p8_deterministic_common import sha256
from workflows.sbi.p12a_blind_energy_score import (
    align_positions,
    clustered_mean_interval,
    gaussian_samples,
    joint_energy_score,
)
from workflows.sbi.p12a_immutable_io import (
    write_json_exclusive,
    write_or_validate_npz_exclusive,
)
from workflows.sbi.p12a_evaluate_blind import (
    load_prediction_arrays,
    validate_open_state,
    validate_proper_score_report,
)


SCHEMA = "p12a-ph001-joint-energy-score-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _prediction_manifests(frozen: dict) -> dict[str, dict]:
    result = {}
    for record in frozen["prediction_manifests"]:
        path = Path(record["path"])
        if sha256(path) != record["sha256"]:
            raise RuntimeError("prediction manifest changed after blind freeze")
        payload = json.loads(path.read_text())
        result[payload["schema_version"]] = payload
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-predictions", type=Path, required=True)
    parser.add_argument("--opened-marker", type=Path, required=True)
    parser.add_argument("--evaluation-contract", type=Path, required=True)
    parser.add_argument("--truth-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-array", type=Path, required=True)
    args = parser.parse_args()
    frozen, opened, contract, truth_manifest = validate_open_state(
        frozen_path=args.frozen_predictions,
        opened_path=args.opened_marker,
        contract_path=args.evaluation_contract,
        truth_manifest_path=args.truth_manifest,
    )
    expected_output = Path(contract["canonical_outputs"]["energy_score_report"]).resolve()
    expected_array = Path(contract["canonical_outputs"]["energy_score_array"]).resolve()
    if args.output.resolve() != expected_output or args.score_array.resolve() != expected_array:
        raise PermissionError("energy-score output paths differ from the frozen contract")
    protocol = contract["evaluation_protocol"]
    summary, audit = load_prediction_arrays(frozen)
    if (
        len(audit["parent_node_id"]) != int(protocol["audit_rows"])
        or audit["eigenvalue_draws"].shape
        != (int(protocol["audit_rows"]), int(protocol["posterior_draws"]), 3)
    ):
        raise RuntimeError("frozen production audit draws violate the score contract")
    truth_path = Path(truth_manifest["array"]["path"])
    with np.load(truth_path, mmap_mode="r") as truth_archive:
        truth_parent = np.asarray(truth_archive["parent_node_id"], dtype=np.int64)
        if not np.array_equal(truth_parent, summary["parent_node_id"]):
            raise RuntimeError("truth package differs from frozen posterior rows")
        audit_position = align_positions(truth_parent, audit["parent_node_id"])
        truth = np.asarray(truth_archive["eigenvalues"][audit_position], dtype=np.float64)
    manifests = _prediction_manifests(frozen)
    context_path = Path(manifests["p12a-blind-base-context-v1"]["array"])
    with np.load(context_path, mmap_mode="r") as context_archive:
        context_position = align_positions(
            context_archive["parent_node_id"], audit["parent_node_id"]
        )
        core = np.asarray(context_archive["core_id"][context_position], dtype=np.int64)
    gaussian_path = Path(contract["gaussian_baseline"]["path"])
    if sha256(gaussian_path) != contract["gaussian_baseline"]["sha256"]:
        raise RuntimeError("Gaussian baseline changed after evaluation-contract freeze")
    gaussian = json.loads(gaussian_path.read_text())
    gaussian_draws = gaussian_samples(
        base=summary["base_prediction_eigenvalues"][audit_position],
        shell=summary["shell"][audit_position],
        cap=summary["cap"][audit_position],
        gaussian=gaussian,
        draws=int(protocol["posterior_draws"]),
        seed=int(protocol["gaussian_draw_seed"]),
    )
    fmpe_score = joint_energy_score(
        audit["eigenvalue_draws"],
        truth,
        pairing_offset=int(protocol["energy_pairing_offset"]),
    )
    gaussian_score = joint_energy_score(
        gaussian_draws,
        truth,
        pairing_offset=int(protocol["energy_pairing_offset"]),
    )
    difference = gaussian_score - fmpe_score
    comparison = clustered_mean_interval(
        difference,
        core,
        repeats=int(protocol["bootstrap_repetitions"]),
        seed=int(protocol["bootstrap_seed"]),
    )
    score_arrays = {
        "parent_node_id": np.asarray(audit["parent_node_id"], dtype=np.int64),
        "core_id": core,
        "fmpe_joint_energy_score": fmpe_score.astype(np.float64),
        "gaussian_joint_energy_score": gaussian_score.astype(np.float64),
        "gaussian_minus_fmpe": difference.astype(np.float64),
    }
    write_or_validate_npz_exclusive(
        args.score_array,
        **score_arrays,
    )
    if args.output.exists():
        existing = json.loads(args.output.read_text())
        validate_proper_score_report(
            existing,
            frozen_path=args.frozen_predictions,
            opened_path=args.opened_marker,
            contract_path=args.evaluation_contract,
            truth_manifest_path=args.truth_manifest,
            contract=contract,
            audit_parent=audit["parent_node_id"],
            audit_core=core,
            audit_draws=audit["eigenvalue_draws"],
            audit_truth=truth,
            gaussian_base=summary["base_prediction_eigenvalues"][audit_position],
            audit_shell=summary["shell"][audit_position],
            audit_cap=summary["cap"][audit_position],
            expected_truth_files=opened["truth_files_read"],
        )
        print(json.dumps(existing, indent=2), flush=True)
        return
    report = {
        "schema_version": SCHEMA,
        "created_utc": utc_now(),
        "git_revision": git_revision(),
        "phase": "ph001",
        "score": "multivariate energy score in physical ordered-eigenvalue coordinates",
        "lower_is_better": True,
        "comparison": "gaussian_minus_fmpe; positive favours FMPE",
        "rows": int(len(truth)),
        "posterior_draws": int(protocol["posterior_draws"]),
        "energy_pairing_offset": int(protocol["energy_pairing_offset"]),
        "gaussian_draw_seed": int(protocol["gaussian_draw_seed"]),
        "gaussian_ordering_transform": protocol["gaussian_ordering_transform"],
        "fmpe_mean_joint_energy_score": float(fmpe_score.mean()),
        "gaussian_mean_joint_energy_score": float(gaussian_score.mean()),
        **comparison,
        "score_array": {
            "path": str(args.score_array.resolve()),
            "sha256": sha256(args.score_array),
            "bytes": args.score_array.stat().st_size,
        },
        "frozen_predictions": {
            "path": str(args.frozen_predictions.resolve()),
            "sha256": sha256(args.frozen_predictions),
        },
        "opened_marker": {
            "path": str(args.opened_marker.resolve()),
            "sha256": sha256(args.opened_marker),
        },
        "evaluation_contract": {
            "path": str(args.evaluation_contract.resolve()),
            "sha256": sha256(args.evaluation_contract),
        },
        "truth_manifest": {
            "path": str(args.truth_manifest.resolve()),
            "sha256": sha256(args.truth_manifest),
        },
        "unnormalized_fmpe_log_score_used": False,
        "post_open_refit_performed": False,
        "post_open_tuning_allowed": False,
        "truth_files_read": opened["truth_files_read"],
        "open_count": 1,
        "sealed_phase_opened": True,
        "pass": bool(
            comparison["ci95"][0]
            > contract["gates"][
                "gaussian_minus_fmpe_energy_score_ci95_lower_minimum"
            ]
        ),
    }
    write_json_exclusive(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
