#!/usr/bin/env python3
"""Plan core-safe P12-A shards and freeze a complete truth-free export."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np

from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.sbi.p12_production_contract import assert_truth_free_payload


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def core_safe_shards(core_id: np.ndarray, shard_count: int = 4) -> list[dict]:
    core = np.asarray(core_id, dtype=np.int64)
    if core.ndim != 1 or len(core) == 0 or np.any(core < 0) or np.any(np.diff(core) < 0):
        raise ValueError("context core IDs must be nonempty, nonnegative and ordered")
    if shard_count < 1:
        raise ValueError("shard count must be positive")
    starts = np.r_[0, np.flatnonzero(np.diff(core) != 0) + 1, len(core)]
    if len(starts) - 1 < shard_count:
        raise ValueError("fewer authoritative cores than requested shards")
    boundaries = [0]
    for shard in range(1, shard_count):
        target = len(core) * shard / shard_count
        candidates = starts[1:-1]
        valid = candidates[candidates > boundaries[-1]]
        remaining_shards = shard_count - shard
        valid = valid[valid <= starts[-(remaining_shards + 1)]]
        if len(valid) == 0:
            raise RuntimeError("cannot construct nonempty core-safe shards")
        boundaries.append(int(valid[np.argmin(np.abs(valid - target))]))
    boundaries.append(len(core))
    result = []
    for index, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if start and core[start - 1] == core[start]:
            raise RuntimeError("shard boundary splits an authoritative core")
        result.append(
            {
                "shard": index,
                "start": int(start),
                "stop": int(stop),
                "rows": int(stop - start),
                "first_core_id": int(core[start]),
                "last_core_id": int(core[stop - 1]),
                "cores": int(len(np.unique(core[start:stop]))),
            }
        )
    return result


def build_plan(context_path: Path, output_path: Path, shard_count: int) -> dict:
    if "ph001" not in str(context_path).lower():
        raise PermissionError("production blind shard plan must identify ph001")
    archive = np.load(context_path, mmap_mode="r")
    forbidden = ("truth", "target", "tweb", "cweb")
    if any(any(token in name.lower() for token in forbidden) for name in archive.files):
        raise PermissionError("context archive contains a truth-bearing array")
    required = {"parent_node_id", "core_id", "support_random"}
    if not required.issubset(archive.files):
        raise RuntimeError("blind context is missing core-safe shard arrays")
    if not np.all(np.asarray(archive["support_random"], dtype=bool)):
        raise RuntimeError("M=0 rows cannot enter the blind export plan")
    parent = np.asarray(archive["parent_node_id"], dtype=np.int64)
    if len(np.unique(parent)) != len(parent):
        raise RuntimeError("blind context contains duplicate authoritative parents")
    shards = core_safe_shards(archive["core_id"], shard_count)
    marker = {
        "schema_version": "p12a-blind-core-shard-plan-v1",
        "created_utc": utc_now(),
        "git_revision": git_revision(),
        "phase": "ph001",
        "context": str(context_path),
        "context_sha256": sha256(context_path),
        "rows": int(len(parent)),
        "unique_parents": int(len(parent)),
        "shard_count": shard_count,
        "shards": shards,
        "truth_files_read": [],
        "open_count": 0,
        "sealed_phase_opened": False,
        "pass": True,
    }
    assert_truth_free_payload(marker)
    atomic_json(output_path, marker)
    archive.close()
    return marker


def freeze_complete(plan_path: Path, manifests: list[Path], output_path: Path) -> dict:
    plan = json.loads(plan_path.read_text())
    assert_truth_free_payload(plan)
    if plan.get("schema_version") != "p12a-blind-core-shard-plan-v1" or not plan.get("pass"):
        raise RuntimeError("blind shard plan is not frozen")
    if len(manifests) != int(plan["shard_count"]):
        raise RuntimeError("blind shard manifest count mismatch")
    context_path = Path(plan["context"])
    if sha256(context_path) != plan["context_sha256"]:
        raise RuntimeError("blind context changed after shard planning")
    context = np.load(context_path, mmap_mode="r")
    observed_parent = []
    observed_core = []
    evidence = []
    for expected, path in zip(plan["shards"], manifests):
        payload = json.loads(path.read_text())
        assert_truth_free_payload(payload)
        if payload.get("schema_version") != "p12a-blind-posterior-shard-v1" or not payload.get("pass"):
            raise RuntimeError(f"posterior shard does not pass: {path}")
        for key in ("start", "stop", "rows"):
            if int(payload[key]) != int(expected[key]):
                raise RuntimeError(f"posterior shard interval mismatch: {path}")
        if payload.get("context_sha256") != plan["context_sha256"] or int(payload.get("draws", 0)) != 512:
            raise RuntimeError("posterior shard context/draw contract mismatch")
        summary_path = Path(payload["summary"])
        if sha256(summary_path) != payload["summary_sha256"]:
            raise RuntimeError("posterior shard summary hash mismatch")
        summary = np.load(summary_path, mmap_mode="r")
        required = {"parent_node_id", "core_id", "quality_bitmask", "web_class_probability"}
        if not required.issubset(summary.files) or len(summary["parent_node_id"]) != expected["rows"]:
            raise RuntimeError("posterior shard summary schema mismatch")
        probability = np.asarray(summary["web_class_probability"], dtype=np.float64)
        if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
            raise RuntimeError("posterior class probabilities are not normalized")
        observed_parent.append(np.asarray(summary["parent_node_id"], dtype=np.int64))
        observed_core.append(np.asarray(summary["core_id"], dtype=np.int64))
        evidence.append({"path": str(path), "sha256": sha256(path)})
        summary.close()
    parent = np.concatenate(observed_parent)
    core = np.concatenate(observed_core)
    if not np.array_equal(parent, np.asarray(context["parent_node_id"], dtype=np.int64)):
        raise RuntimeError("posterior shards do not exactly cover blind parents")
    if not np.array_equal(core, np.asarray(context["core_id"], dtype=np.int64)):
        raise RuntimeError("posterior shards do not exactly cover blind cores")
    marker = {
        "schema_version": "p12a-blind-export-complete-v1",
        "created_utc": utc_now(),
        "git_revision": git_revision(),
        "phase": "ph001",
        "plan": {"path": str(plan_path), "sha256": sha256(plan_path)},
        "context": {"path": str(context_path), "sha256": plan["context_sha256"]},
        "rows": int(len(parent)),
        "shards": evidence,
        "truth_files_read": [],
        "open_count": 0,
        "sealed_phase_opened": False,
        "pass": True,
    }
    assert_truth_free_payload(marker)
    atomic_json(output_path, marker)
    context.close()
    return marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--context", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--shards", type=int, default=4)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--plan", type=Path, required=True)
    freeze.add_argument("--manifest", type=Path, action="append", required=True)
    freeze.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = (
        build_plan(args.context, args.output, args.shards)
        if args.command == "plan"
        else freeze_complete(args.plan, args.manifest, args.output)
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
