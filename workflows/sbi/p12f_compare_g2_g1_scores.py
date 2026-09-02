#!/usr/bin/env python3
"""Paired, streaming proper-score comparison for P12-F G2 versus frozen G1."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.sbi.p12f_challenger_common import core_joint_scores, paired_core_bootstrap
from workflows.sbi.p12f_common_evaluator import efficient_crps_ensemble, load_core_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--panel-marker", type=Path, required=True)
    parser.add_argument("--g1-archive", type=Path, required=True)
    parser.add_argument("--g2-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-wall-seconds", type=float, default=13_500.0)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(payload: dict) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _manifest(path: Path, expected_method: str, panel_path: Path) -> dict:
    payload = json.loads(path.read_text())
    if (
        payload.get("schema_version") != "p12f-sample-archive-v1"
        or payload.get("method") != expected_method
        or payload.get("phase") != "ph006"
        or payload.get("ph001_opened")
        or not payload.get("pass")
        or payload.get("panel_sha256") != sha256(panel_path)
    ):
        raise RuntimeError(f"unsafe or mismatched archive {path}")
    for row in payload.get("entries", []):
        artifact = Path(row["path"])
        if not artifact.is_file() or sha256(artifact) != row["sha256"]:
            raise RuntimeError(f"archive artifact changed: {artifact}")
    return payload


def _core_values(record: dict, draws: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bounds = np.asarray(record["core_bounds"], dtype=np.int64)
    core = tuple(
        slice(int(left), int(right))
        for left, right in zip(bounds[0], bounds[1], strict=True)
    )
    return (
        np.asarray(record["delta_samples"][:draws][(slice(None),) + core]),
        np.asarray(record["delta_truth"])[core],
        np.asarray(record["support"], dtype=bool)[core],
    )


def main() -> None:
    args = parse_args()
    if args.output.exists():
        print(args.output.read_text(), flush=True)
        return
    config = json.loads(args.config.read_text())
    panel = json.loads(args.panel_marker.read_text())
    comparison = config["proper_score_comparison"]
    proper_draws = int(comparison["draws"])
    if proper_draws != 64:
        raise RuntimeError("G2 proper-score comparison is preregistered at 64 draws")
    if proper_draws != int(
        config["conditional_covariance_control"]["proper_score_draws"]
    ):
        raise RuntimeError("G2 proper-score draw contracts disagree")
    if panel.get("ph001_opened") or panel.get("truth_files_read"):
        raise PermissionError("G2 comparison requires the truth-free ph006 panel")
    g1 = _manifest(args.g1_archive, "gaussian_correlated_g1", args.panel_marker)
    g2 = _manifest(
        args.g2_archive, "gaussian_shell_correlated_g2", args.panel_marker
    )
    if int(g1["draws"]) < proper_draws or int(g2["draws"]) < proper_draws:
        raise RuntimeError("G1/G2 archives have too few registered draws")
    ids = [int(value) for value in panel["selected_core_id"]]
    g1_rows = {int(row["core_id"]): row for row in g1["entries"]}
    g2_rows = {int(row["core_id"]): row for row in g2["entries"]}
    if list(g1_rows) != ids or list(g2_rows) != ids:
        raise RuntimeError("G1/G2 archives do not exactly match the 1,024-core panel")

    frozen = {
        "config_sha256": sha256(args.config),
        "panel_sha256": sha256(args.panel_marker),
        "g1_archive_sha256": sha256(args.g1_archive),
        "g2_archive_sha256": sha256(args.g2_archive),
        "draws": proper_draws,
        "core_id": ids,
    }
    frozen_digest = _digest(frozen)
    progress_path = args.output.with_name(args.output.stem + "_PROGRESS.json")
    progress = (
        json.loads(progress_path.read_text())
        if progress_path.exists()
        else {
            "schema_version": "p12f-g2-vs-g1-proper-score-progress-v1",
            "frozen_digest": frozen_digest,
            "per_core": [],
            "ph001_opened": False,
        }
    )
    if progress.get("frozen_digest") != frozen_digest or progress.get("ph001_opened"):
        raise RuntimeError("G2 proper-score resume contract changed")
    done = {int(row["core_id"]): row for row in progress["per_core"]}
    started = time.monotonic()
    for ordinal, core_id in enumerate(ids):
        if core_id in done:
            continue
        left = load_core_record(g1_rows[core_id], int(g1["draws"]))
        right = load_core_record(g2_rows[core_id], int(g2["draws"]))
        for name in ("delta_truth", "support", "core_bounds", "galaxy_frac_index_local"):
            if not np.array_equal(left[name], right[name]):
                raise RuntimeError(f"G1/G2 {name} differs for core {core_id}")
        g1_sample, truth, support = _core_values(left, proper_draws)
        g2_sample, g2_truth, g2_support = _core_values(right, proper_draws)
        if not np.array_equal(truth, g2_truth) or not np.array_equal(support, g2_support):
            raise RuntimeError(f"G1/G2 core views differ for core {core_id}")
        seed = 42 + core_id
        score_kwargs = {
            "feature_count": int(comparison["fixed_voxel_features_per_core"]),
            "pair_count": int(comparison["fixed_variogram_pairs_per_core"]),
            "seed": seed,
        }
        g1_score = core_joint_scores(g1_sample, truth, support, **score_kwargs)
        g2_score = core_joint_scores(g2_sample, truth, support, **score_kwargs)
        valid = np.flatnonzero(support.ravel())
        crps_voxels = int(comparison["fixed_crps_voxels_per_core"])
        if len(valid) > crps_voxels:
            valid = valid[
                np.linspace(0, len(valid) - 1, crps_voxels, dtype=np.int64)
            ]
        g1_score["marginal_crps"] = efficient_crps_ensemble(
            g1_sample.reshape(proper_draws, -1)[:, valid], truth.ravel()[valid]
        )
        g2_score["marginal_crps"] = efficient_crps_ensemble(
            g2_sample.reshape(proper_draws, -1)[:, valid], truth.ravel()[valid]
        )
        done[core_id] = {"core_id": core_id, "g1": g1_score, "g2": g2_score}
        progress["per_core"] = [done[value] for value in ids if value in done]
        if (ordinal + 1) % 8 == 0 or ordinal == 0:
            atomic_json(progress_path, progress)
        if ordinal == 0 or (ordinal + 1) % 64 == 0:
            print(json.dumps({"proper_score_core": ordinal + 1, "total": len(ids)}), flush=True)
        if time.monotonic() - started >= args.max_wall_seconds:
            atomic_json(progress_path, progress)
            raise SystemExit(75)

    per_core = [done[value] for value in ids]
    atomic_json(progress_path, progress)

    names = ("energy", "variogram_p0p5", "coarse_energy", "marginal_crps")
    scores = {
        method: {
            name: float(np.mean([row[method][name] for row in per_core]))
            for name in names
        }
        for method in ("g1", "g2")
    }
    bootstrap = {
        name: paired_core_bootstrap(
            np.asarray([row["g2"][name] for row in per_core]),
            np.asarray([row["g1"][name] for row in per_core]),
            replicates=int(comparison["bootstrap_replicates"]),
            seed=142 + index,
        )
        for index, name in enumerate(names)
    }
    gates = config["selection_gates"]
    primary = bootstrap["energy"]
    primary_pass = (
        primary["fractional_improvement"] >= float(gates["joint_improvement"])
        and primary["interval95"][0] > 0.0
    )
    non_regression = {
        name: scores["g2"][name]
        <= scores["g1"][name] * (1.0 + float(gates["other_score_worsening"]))
        for name in names
        if name != "energy"
    }
    report = {
        "schema_version": "p12f-g2-vs-g1-proper-score-v1",
        "created_utc": utc_now(),
        "phase": "ph006",
        "cores": len(ids),
        "draws": proper_draws,
        "scores": scores,
        "paired_core_bootstrap": bootstrap,
        "gates": {
            "primary_energy_pass": bool(primary_pass),
            "non_regression": non_regression,
            "pass": bool(primary_pass and all(non_regression.values())),
        },
        "per_core": per_core,
        "config_sha256": sha256(args.config),
        "panel_sha256": sha256(args.panel_marker),
        "g1_archive_sha256": sha256(args.g1_archive),
        "g2_archive_sha256": sha256(args.g2_archive),
        "resampling_unit": "authoritative patch core",
        "truth_files_read": ["ph006 density from frozen G1/G2 archives"],
        "ph001_opened": False,
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
