#!/usr/bin/env python3
"""Run a bounded truth-free P12-A posterior smoke and project production cost."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.sbi.p12_production_contract import assert_truth_free_payload
from workflows.sbi.p12a_blind_inference import posterior_inference_shard, validate_context_archive


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def core_aligned_stop(core_id: np.ndarray, minimum_rows: int) -> int:
    core = np.asarray(core_id, dtype=np.int64)
    if core.ndim != 1 or len(core) == 0 or np.any(np.diff(core) < 0):
        raise ValueError("blind core IDs must be nonempty and ordered")
    if not 0 < minimum_rows < len(core):
        raise ValueError("smoke minimum rows must be inside the blind population")
    changes = np.flatnonzero(np.diff(core) != 0) + 1
    candidates = changes[changes >= minimum_rows]
    return int(candidates[0]) if len(candidates) else len(core)


def projected_four_gpu_seconds(elapsed_seconds: float, rows: int, total_rows: int) -> float:
    if elapsed_seconds <= 0 or rows <= 0 or total_rows < rows:
        raise ValueError("invalid throughput projection inputs")
    return float(elapsed_seconds * total_rows / (rows * 4.0))


def run_smoke(args: argparse.Namespace) -> dict:
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to overwrite blind smoke: {args.output}")
    plan = json.loads(args.plan.read_text())
    assert_truth_free_payload(plan)
    if (
        plan.get("schema_version") != "p12a-blind-core-shard-plan-v1"
        or plan.get("phase") != "ph001"
        or plan.get("pass") is not True
        or plan.get("context_sha256") != sha256(args.context)
    ):
        raise RuntimeError("blind shard plan/context contract mismatch")
    context = np.load(args.context, mmap_mode="r")
    validate_context_archive(context)
    stop = core_aligned_stop(context["core_id"], args.minimum_rows)
    total_rows = int(len(context["parent_node_id"]))
    context.close()

    started = time.perf_counter()
    result = posterior_inference_shard(
        candidate_marker_path=args.candidate,
        context_path=args.context,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        start=0,
        stop=stop,
        draws=512,
        seed=args.seed,
        device=args.device,
        sample_chunk=args.sample_chunk,
        quality_thresholds_path=args.quality_thresholds,
    )
    elapsed = time.perf_counter() - started
    summary = np.load(args.output, mmap_mode="r")
    probabilities = np.asarray(summary["web_class_probability"], dtype=np.float64)
    finite = all(np.all(np.isfinite(summary[name])) for name in summary.files)
    normalized = np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=0.0)
    summary.close()
    projected_seconds = projected_four_gpu_seconds(elapsed, stop, total_rows)
    marker = {
        "schema_version": "p12a-blind-throughput-smoke-v1",
        "created_utc": utc_now(),
        "git_revision": git_revision(),
        "phase": "ph001",
        "rows": stop,
        "total_rows": total_rows,
        "draws": 512,
        "elapsed_seconds": elapsed,
        "rows_per_second_single_gpu": stop / elapsed,
        "projected_four_gpu_seconds": projected_seconds,
        "projected_four_gpu_hours": projected_seconds / 3600.0,
        "projection_caveat": (
            "linear projection from one bounded core-aligned sample; production array "
            "startup, shard imbalance and filesystem contention are not included"
        ),
        "summary": {"path": str(args.output), "sha256": sha256(args.output)},
        "posterior_marker": result,
        "finite": bool(finite),
        "class_probabilities_normalized": bool(normalized),
        "truth_files_read": [],
        "open_count": 0,
        "sealed_phase_opened": False,
        "pass": bool(finite and normalized and result.get("pass")),
    }
    assert_truth_free_payload(marker)
    atomic_json(args.output.with_suffix(".json"), marker)
    return marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--quality-thresholds", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-rows", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-chunk", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run_smoke(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
