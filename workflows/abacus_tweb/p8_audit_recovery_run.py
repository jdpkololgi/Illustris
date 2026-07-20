#!/usr/bin/env python3
"""Audit P8 recovery exposure, loss accounting, validation, and trace integrity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p8_deterministic_common import atomic_json


def jsonl(path: Path) -> list[dict]:
    return [json.loads(row) for row in path.read_text().splitlines() if row.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir
    manifest = json.loads((root / "run_manifest.json").read_text())
    summary = json.loads((root / "recovery_summary.json").read_text())
    report = json.loads((root / "best_validation_report.json").read_text())
    history = jsonl(root / "epoch_history.jsonl")
    trace = jsonl(root / "loss_trace.jsonl")

    expected_rows = np.array(
        list(manifest["roles"]["training_shell_counts"].values()), dtype=np.int64
    )
    trace_steps = np.array([row["global_step"] for row in trace], dtype=np.int64)
    epoch_checks = []
    for row in history:
        checks = {
            "epoch": int(row["epoch"]),
            "all_cores_once": (
                row["eligible_cores"] == manifest["eligible_training_cores"]
                and row["unique_cores_seen"] == manifest["eligible_training_cores"]
                and row["repeat_cores"] == 0
                and row["unique_core_fraction"] == 1.0
            ),
            "all_rows_accounted": bool(
                np.array_equal(row["training_rows_by_shell"], expected_rows)
            ),
            "finite_loss_accounting": bool(
                np.isfinite(row["training_loss_numerator"])
                and np.isfinite(row["training_weight_denominator"])
                and row["training_weight_denominator"] > 0
            ),
            "finite_validation_score": bool(
                np.isfinite(row["primary_macro_r2_lambda1"])
            ),
        }
        checks["pass"] = all(value for key, value in checks.items() if key != "epoch")
        epoch_checks.append(checks)

    checks = {
        "completion_marker": bool(
            (root / "CANARY_COMPLETE").exists()
            or (root / "CONVERGED_EARLY_STOP").exists()
            or (root / "NOT_CONVERGED_MAX_EPOCHS").exists()
        ),
        "history_matches_summary": history == summary["history"],
        "completed_epoch_count": len(history) == summary["epochs_completed"],
        "complete_validation_fold": bool(report["complete_core_coverage"]),
        "zero_patch_failures": report["runtime"]["patch_failures"] == 0,
        "trace_nonempty": len(trace) > 0,
        "trace_steps_strictly_increasing": bool(
            len(trace_steps) > 0 and np.all(np.diff(trace_steps) > 0)
        ),
        "trace_not_beyond_checkpoint": bool(
            len(trace_steps) > 0 and trace_steps[-1] <= summary["global_steps"]
        ),
        "every_epoch_passes": bool(epoch_checks) and all(
            row["pass"] for row in epoch_checks
        ),
    }
    result = {
        "schema_version": 1,
        "stage": "P8 recovery run audit",
        "run_dir": str(root),
        "model": manifest["model"],
        "rotation": manifest["rotation"],
        "status": summary["status"],
        "epoch_checks": epoch_checks,
        "checks": checks,
        "pass": all(checks.values()),
    }
    atomic_json(root / "recovery_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
