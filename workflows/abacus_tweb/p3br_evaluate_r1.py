#!/usr/bin/env python3
"""Matched ph006 evaluation and promotion decision for random-response R1."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p8_deterministic_common import (
    SHELL_NAMES,
    _one_shell_metrics,
    authoritative_mask,
    atomic_json,
    evaluate_complete_phase,
    sha256,
)
from workflows.abacus_tweb.p10_training_contract import P10PhaseBalancedLoader


ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def promotion_decision(r0: dict, r1: dict) -> dict:
    gain = float(r1["primary_macro_r2_lambda1"] - r0["primary_macro_r2_lambda1"])
    changes = {
        name: float(
            r1["per_shell"][name]["lambda1"]["r2"]
            - r0["per_shell"][name]["lambda1"]["r2"]
        )
        for name in SHELL_NAMES
    }
    no_degradation = min(changes.values()) >= -0.01
    if gain >= 0.03 and no_degradation:
        action = "R1_PROMOTION_CANDIDATE_REPLICATE_SECOND_SEED"
        promoted_seed42 = True
    elif 0.01 <= gain < 0.03:
        action = "RUN_SECOND_SEED_PROMOTE_ONLY_IF_BOTH_POSITIVE_AND_MEAN_AT_LEAST_0P02"
        promoted_seed42 = False
    else:
        action = "RETAIN_RANDOM_RESPONSE_FOR_POSTERIOR_AND_SAFETY_NO_ACCURACY_CLAIM"
        promoted_seed42 = False
    if gain >= 0.03 and not no_degradation:
        action = "DO_NOT_PROMOTE_SUPPORTED_SHELL_DEGRADATION"
        promoted_seed42 = False
    return {
        "macro_r2_gain": gain,
        "per_shell_lambda1_r2_change": changes,
        "no_shell_degradation_worse_than_0p01": no_degradation,
        "action": action,
        "seed42_promotion_candidate": promoted_seed42,
        "second_seed_required_before_final_freeze": gain >= 0.01,
    }


def align_prediction(parent: np.ndarray, prediction: np.ndarray,
                     required_parent: np.ndarray) -> np.ndarray:
    parent = np.asarray(parent, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.float64)
    order = np.argsort(parent)
    sorted_parent = parent[order]
    lookup = np.searchsorted(sorted_parent, required_parent)
    if (
        np.any(lookup == len(sorted_parent))
        or not np.array_equal(
            sorted_parent[np.minimum(lookup, len(sorted_parent) - 1)], required_parent
        )
        or len(parent) != len(required_parent)
    ):
        raise RuntimeError("prediction does not exactly cover authoritative ph006 parents")
    return prediction[order][lookup]


def sample_response(contract: Path, points: np.ndarray,
                    parent: np.ndarray) -> dict[str, np.ndarray]:
    inventory = load(contract / "adapter_inventory.json")
    field_manifest = load(Path(inventory["phases"]["ph006"]["field_manifest"]))
    positions = np.asarray(points[parent, :3], dtype=np.float64)
    cap_id = np.asarray(points[parent, 3], dtype=np.int8)
    result = {
        name: np.zeros(len(parent), dtype=np.float32)
        for name in (
            "expected_counts_random", "angular_response",
            "distance_to_support_boundary", "support_random",
        )
    }
    for wanted_cap, cap_name in ((0, "SGC"), (1, "NGC")):
        selected = cap_id == wanted_cap
        component = field_manifest["caps"][cap_name]
        origin = np.asarray(component["grid"]["origin_mpc"], dtype=np.float64)
        cell = float(component["grid"]["cell_mpc"])
        index = np.floor((positions[selected] - origin) / cell).astype(np.int64)
        with h5py.File(component["field_path"], "r") as handle:
            shape = np.asarray(handle["counts"].shape, dtype=np.int64)
            inside = np.all((index >= 0) & (index < shape), axis=1)
            if not inside.all():
                raise RuntimeError(f"{cap_name} authoritative galaxies fall outside R1 grid")
            ix, iy, iz = index.T
            for name in result:
                # h5py permits only limited multi-axis fancy indexing.  Group
                # by the first grid axis so every HDF5 read is one contiguous
                # 2-D plane, then use NumPy for the remaining point selection.
                sampled = np.empty(len(ix), dtype=np.float32)
                for plane_index in np.unique(ix):
                    rows = ix == plane_index
                    plane = np.asarray(handle[name][int(plane_index)], dtype=np.float32)
                    sampled[rows] = plane[iy[rows], iz[rows]]
                result[name][selected] = sampled
    return result


def subset_metrics(truth: np.ndarray, prediction: np.ndarray,
                   selected: np.ndarray) -> dict:
    n = int(np.sum(selected))
    if n < 3:
        return {"n": n, "metrics": None}
    return {"n": n, "metrics": _one_shell_metrics(truth[selected], prediction[selected])}


def quantile_bins(values: np.ndarray, eligible: np.ndarray, truth: np.ndarray,
                  predictions: dict[str, np.ndarray]) -> dict:
    edges = np.quantile(values[eligible], [0.0, 0.25, 0.5, 0.75, 1.0])
    result = {}
    for index in range(4):
        selected = eligible & (values >= edges[index])
        selected &= values <= edges[index + 1] if index == 3 else values < edges[index + 1]
        result[f"q{index + 1}"] = {
            "range_expected_counts": [float(edges[index]), float(edges[index + 1])],
            "models": {
                name: subset_metrics(truth, prediction, selected)
                for name, prediction in predictions.items()
            },
        }
    return result


def stratified_metrics(truth: np.ndarray, predictions: dict[str, np.ndarray],
                       response: dict[str, np.ndarray], shell: np.ndarray) -> dict:
    expected = np.asarray(response["expected_counts_random"], dtype=np.float64)
    distance = np.asarray(response["distance_to_support_boundary"], dtype=np.float64)
    support = np.asarray(response["support_random"], dtype=bool)
    positive = support & (expected > 0) & np.isfinite(expected)
    response_bins = quantile_bins(expected, positive, truth, predictions)
    response_bins_by_shell = {
        name: quantile_bins(
            expected, positive & (np.asarray(shell) == shell_id), truth, predictions
        )
        for shell_id, name in enumerate(SHELL_NAMES)
    }
    boundary_edges = (0.0, 10.4, 20.8, 40.0, np.inf)
    boundary_bins = {}
    for index, (left, right) in enumerate(zip(boundary_edges[:-1], boundary_edges[1:])):
        selected = positive & (distance >= left) & (distance < right)
        boundary_bins[f"b{index + 1}"] = {
            "range_mpc": [float(left), None if not np.isfinite(right) else float(right)],
            "models": {
                name: subset_metrics(truth, prediction, selected)
                for name, prediction in predictions.items()
            },
        }
    return {
        "supported_fraction": float(support.mean()),
        "positive_expected_fraction": float(positive.mean()),
        "response_quantiles": response_bins,
        "response_quantiles_within_shell": response_bins_by_shell,
        "distance_to_random_support_boundary": boundary_bins,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--r1-contract", type=Path, default=ROOT / "training_contract_r1_random"
    )
    parser.add_argument(
        "--r0-run", type=Path,
        default=ROOT / "arm_a_training/arm_a_r0_v1/unet/seed_42",
    )
    parser.add_argument(
        "--r1-run", type=Path,
        default=ROOT / "response_training/p3br_r1_v1/unet/seed_42",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "response_evaluation/r0_r1_classical_ph006.json"
    )
    args = parser.parse_args()

    loader = P10PhaseBalancedLoader(args.r1_contract, include_blind=False)
    if loader.blind_phase != "ph001":
        raise RuntimeError("unexpected blind phase")
    phase = loader.phase_records["ph006"]
    assignment_path = Path(phase["inputs"]["assignment"])
    truth_path = Path(phase["target"]["path"])
    p1_manifest = load(Path(phase["inputs"]["p1_manifest"]))
    points_path = Path(p1_manifest["points"])
    assignment = np.load(assignment_path, mmap_mode="r")
    truth_by_parent = np.load(truth_path, mmap_mode="r")
    points = np.load(points_path, mmap_mode="r")
    rows = np.flatnonzero(authoritative_mask(assignment))
    required_parent = np.asarray(assignment["parent_node_id"][rows], dtype=np.int64)
    shell = np.asarray(assignment["shell"][rows], dtype=np.int8)
    truth = np.asarray(truth_by_parent[required_parent], dtype=np.float64)

    artifacts = {
        "R0": (
            args.r0_run / "best_validation_parent_node_id.npy",
            args.r0_run / "best_validation_eigenvalues.npy",
        ),
        "R1": (
            args.r1_run / "best_validation_parent_node_id.npy",
            args.r1_run / "best_validation_eigenvalues.npy",
        ),
        "CIC_R1": (
            args.root / "classical/cic_random_response_v1/ph006/parent_node_id.npy",
            args.root / "classical/cic_random_response_v1/ph006/cic_train_affine_eigenvalues.npy",
        ),
        "DTFE_R1": (
            args.root / "classical/dtfe_random_response_v1/ph006/parent_node_id.npy",
            args.root / "classical/dtfe_random_response_v1/ph006/dtfe_train_affine_eigenvalues.npy",
        ),
    }
    predictions = {}
    reports = {}
    provenance = {}
    for name, (parent_path, prediction_path) in artifacts.items():
        if not parent_path.is_file() or not prediction_path.is_file():
            raise FileNotFoundError(f"missing {name} evaluation artifact")
        parent = np.load(parent_path, mmap_mode="r")
        prediction = np.load(prediction_path, mmap_mode="r")
        aligned = align_prediction(parent, prediction, required_parent)
        predictions[name] = aligned
        reports[name] = evaluate_complete_phase(
            parent_node_id=required_parent,
            predicted_eigenvalues=aligned,
            truth_by_parent=truth_by_parent,
            assignment=assignment,
            phase="ph006",
        )
        provenance[name] = {
            "parent": str(parent_path), "parent_sha256": sha256(parent_path),
            "prediction": str(prediction_path),
            "prediction_sha256": sha256(prediction_path),
        }
    response = sample_response(args.r1_contract, points, required_parent)
    result = {
        "schema_version": "p3br-r0-r1-classical-ph006-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "estimand": "complete authoritative ph006 ordered tidal eigenvalues",
        "models": reports,
        "stratified": stratified_metrics(truth, predictions, response, shell),
        "decision": promotion_decision(reports["R0"], reports["R1"]),
        "provenance": provenance,
        "assignment": str(assignment_path),
        "assignment_sha256": sha256(assignment_path),
        "truth": str(truth_path),
        "truth_sha256": sha256(truth_path),
        "points": str(points_path),
        "points_sha256": sha256(points_path),
        "r1_contract": str(args.r1_contract),
        "r1_contract_marker_sha256": sha256(
            args.r1_contract / "TRAINING_LOADER_READY.json"
        ),
        "ph001_opened": False,
        "pass": True,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
