#!/usr/bin/env python3
"""Paired ph006 evaluation of Bright, real-FAINT Proxy, and angular Null views."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p10_training_contract import atomic_json


ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase")
BRIGHT = ROOT / "arm_a_training/arm_a_r0_v1/unet/seed_42"
RUNS = ROOT / "p12_and_multitracer_training"
PROXY = RUNS / "p10_bf_proxy_v1/unet_multitracer/seed_42"
NULL = RUNS / "p10_bf_null_v1/unet_multitracer/seed_42"
CONTRACT = ROOT / "training_contract"
OUTPUT = ROOT / "multitracer/v1/evaluation/p10_bf_proxy_null_ph006.json"


def load_prediction(root: Path) -> tuple[np.ndarray, np.ndarray]:
    marker = json.loads((root / "ARM_A_TRAINING_COMPLETE.json").read_text())
    if marker.get("blind_truth_accessed") or marker.get("sealed_blind_phase") != "ph001":
        raise RuntimeError(f"invalid blind contract in {root}")
    parent = np.load(root / "best_validation_parent_node_id.npy", mmap_mode="r")
    eigen = np.load(root / "best_validation_eigenvalues.npy", mmap_mode="r")
    return np.asarray(parent, dtype=np.int64), np.asarray(eigen, dtype=np.float64)


def align(reference: np.ndarray, parent: np.ndarray, values: np.ndarray) -> np.ndarray:
    if len(np.unique(parent)) != len(parent):
        raise RuntimeError("prediction parents are not unique")
    order = np.argsort(parent)
    position = np.searchsorted(parent[order], reference)
    if np.any(position >= len(parent)) or not np.array_equal(parent[order][position], reference):
        raise RuntimeError("prediction parent set differs from the frozen reference")
    return values[order][position]


def r2_from_sums(sum_w: float, sum_y: float, sum_y2: float, sse: float) -> float:
    if sum_w <= 0:
        return float("nan")
    sst = sum_y2 - sum_y * sum_y / sum_w
    return float(1.0 - sse / sst) if sst > 0 else float("nan")


def aggregate_core_shell(
    *, core: np.ndarray, shell: np.ndarray, weight: np.ndarray,
    truth: np.ndarray, predictions: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    core_ids, core_inverse = np.unique(core, return_inverse=True)
    n_core = len(core_ids)
    common = np.zeros((n_core, 4, 3), dtype=np.float64)
    methods = {name: np.zeros((n_core, 4), dtype=np.float64) for name in predictions}
    for shell_id in range(4):
        chosen = shell == shell_id
        index = core_inverse[chosen]
        w = weight[chosen]
        y = truth[chosen]
        common[:, shell_id, 0] = np.bincount(index, weights=w, minlength=n_core)
        common[:, shell_id, 1] = np.bincount(index, weights=w * y, minlength=n_core)
        common[:, shell_id, 2] = np.bincount(index, weights=w * y * y, minlength=n_core)
        for name, prediction in predictions.items():
            residual = prediction[chosen] - y
            methods[name][:, shell_id] = np.bincount(
                index, weights=w * residual * residual, minlength=n_core
            )
    return common, methods


def scores(common: np.ndarray, method: np.ndarray, multiplicity=None) -> dict:
    if multiplicity is None:
        multiplicity = np.ones(common.shape[0], dtype=np.float64)
    shell_score = []
    for shell_id in range(4):
        summed = np.sum(common[:, shell_id] * multiplicity[:, None], axis=0)
        sse = float(np.sum(method[:, shell_id] * multiplicity))
        shell_score.append(r2_from_sums(*summed, sse))
    return {
        "shell_r2_lambda1": shell_score,
        "macro_r2_lambda1": float(np.nanmean(shell_score)),
        "first_three_macro_r2_lambda1": float(np.nanmean(shell_score[:3])),
    }


def interval(values: np.ndarray) -> dict:
    return {
        "median": float(np.median(values)),
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "probability_positive": float(np.mean(values > 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bright", type=Path, default=BRIGHT)
    parser.add_argument("--proxy", type=Path, default=PROXY)
    parser.add_argument("--null", type=Path, default=NULL)
    parser.add_argument("--contract-root", type=Path, default=CONTRACT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=732451)
    args = parser.parse_args()
    if args.bootstrap <= 0:
        raise ValueError("bootstrap must be positive")

    parent, bright = load_prediction(args.bright)
    methods = {"bright": bright[:, 0]}
    for name, path in (("proxy", args.proxy), ("null", args.null)):
        found, values = load_prediction(path)
        methods[name] = align(parent, found, values)[:, 0]

    phase = args.contract_root / "phases/ph006"
    truth_all = np.load(phase / "parent_eigenvalues.npy", mmap_mode="r")
    weight_all = np.load(phase / "active_row_weight.npy", mmap_mode="r")
    assignment_path = Path(json.loads((phase / "phase_contract.json").read_text())["inputs"]["assignment"])
    with np.load(assignment_path) as assignment:
        assignment_parent = np.asarray(assignment["parent_node_id"], dtype=np.int64)
        order = np.argsort(assignment_parent)
        position = np.searchsorted(assignment_parent[order], parent)
        if np.any(position >= len(order)) or not np.array_equal(assignment_parent[order][position], parent):
            raise RuntimeError("validation parent lacks assignment row")
        row = order[position]
        core = np.asarray(assignment["core_id"][row], dtype=np.int64)
        shell = np.asarray(assignment["shell"][row], dtype=np.int8)
    truth = np.asarray(truth_all[parent, 0], dtype=np.float64)
    weight = np.asarray(weight_all[parent], dtype=np.float64)
    common, method_sums = aggregate_core_shell(
        core=core, shell=shell, weight=weight, truth=truth, predictions=methods
    )
    point = {name: scores(common, sums) for name, sums in method_sums.items()}

    rng = np.random.default_rng(args.seed)
    delta = {
        "proxy_minus_null_macro": np.empty(args.bootstrap),
        "proxy_minus_null_first_three": np.empty(args.bootstrap),
        "proxy_minus_bright_macro": np.empty(args.bootstrap),
        "proxy_minus_bright_first_three": np.empty(args.bootstrap),
    }
    n_core = common.shape[0]
    for draw in range(args.bootstrap):
        multiplicity = np.bincount(rng.integers(0, n_core, size=n_core), minlength=n_core)
        current = {name: scores(common, sums, multiplicity) for name, sums in method_sums.items()}
        delta["proxy_minus_null_macro"][draw] = current["proxy"]["macro_r2_lambda1"] - current["null"]["macro_r2_lambda1"]
        delta["proxy_minus_null_first_three"][draw] = current["proxy"]["first_three_macro_r2_lambda1"] - current["null"]["first_three_macro_r2_lambda1"]
        delta["proxy_minus_bright_macro"][draw] = current["proxy"]["macro_r2_lambda1"] - current["bright"]["macro_r2_lambda1"]
        delta["proxy_minus_bright_first_three"][draw] = current["proxy"]["first_three_macro_r2_lambda1"] - current["bright"]["first_three_macro_r2_lambda1"]
    report = {
        "schema_version": "p10-multitracer-paired-evaluation-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "ph006",
        "sealed_phase": "ph001",
        "sealed_phase_opened": False,
        "rows": int(len(parent)),
        "spatial_bootstrap_cores": int(n_core),
        "bootstrap_draws": int(args.bootstrap),
        "point": point,
        "paired_delta_bootstrap": {name: interval(values) for name, values in delta.items()},
        "primary_information_estimand": "proxy_minus_null",
        "pass": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
