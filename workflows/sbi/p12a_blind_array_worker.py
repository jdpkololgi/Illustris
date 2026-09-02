#!/usr/bin/env python3
"""Run one core-safe P12-A blind posterior shard on one GPU."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from workflows.abacus_tweb.p8_deterministic_common import sha256
from workflows.sbi.p12_production_contract import P12A_SCHEMA, assert_truth_free_payload
from workflows.sbi.p12a_blind_inference import posterior_inference_shard


def validate_existing_shard(
    *,
    output: Path,
    marker: Path,
    shard: dict,
    draws: int,
    seed: int,
    context_sha256: str,
    checkpoint: Path,
    candidate: Path,
    quality_thresholds: Path,
) -> dict:
    if not (output.exists() and marker.exists()):
        raise FileExistsError("blind shard output/marker pair is incomplete")
    existing = json.loads(marker.read_text())
    assert_truth_free_payload(existing)
    audit = Path(existing.get("audit_draws", ""))
    if (
        existing.get("schema_version") != "p12a-blind-posterior-shard-v1"
        or existing.get("pass") is not True
        or existing.get("start") != shard["start"]
        or existing.get("stop") != shard["stop"]
        or existing.get("draws") != draws
        or existing.get("seed") != seed
        or existing.get("context_sha256") != context_sha256
        or existing.get("checkpoint_sha256") != sha256(checkpoint)
        or existing.get("candidate_sha256") != sha256(candidate)
        or existing.get("quality_thresholds_sha256") != sha256(quality_thresholds)
        or existing.get("summary_sha256") != sha256(output)
        or not audit.is_file()
        or existing.get("audit_draws_sha256") != sha256(audit)
    ):
        raise RuntimeError("existing blind shard fails exact replay contract")
    return existing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--quality-thresholds", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--draws", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-chunk", type=int, default=2048)
    args = parser.parse_args()
    rank = int(os.environ.get("SLURM_PROCID", "0")) if args.rank is None else args.rank
    candidate = json.loads(args.candidate.read_text())
    plan = json.loads(args.plan.read_text())
    for payload in (candidate, plan):
        assert_truth_free_payload(payload)
    if candidate.get("schema_version") != P12A_SCHEMA or not candidate.get("pass"):
        raise RuntimeError("P12-A production candidate is not frozen")
    if plan.get("schema_version") != "p12a-blind-core-shard-plan-v1" or not plan.get("pass"):
        raise RuntimeError("P12-A blind shard plan is not frozen")
    if args.draws != 512 or int(plan.get("shard_count", 0)) != 4:
        raise RuntimeError("production blind export requires four shards and 512 draws")
    if not 0 <= rank < 4:
        raise ValueError("invalid four-GPU shard rank")
    if sha256(args.context) != plan["context_sha256"]:
        raise RuntimeError("blind context differs from the frozen shard plan")
    artifacts = candidate["artifacts"]
    if sha256(args.checkpoint) != artifacts["checkpoint"]["sha256"]:
        raise RuntimeError("P12-A checkpoint differs from the frozen candidate")
    if sha256(args.quality_thresholds) != artifacts["quality_thresholds"]["sha256"]:
        raise RuntimeError("quality thresholds differ from the frozen candidate")
    shard = plan["shards"][rank]
    output = args.output_root / f"shard_{rank:03d}.npz"
    marker = output.with_suffix(".json")
    if output.exists() or marker.exists():
        existing = validate_existing_shard(
            output=output,
            marker=marker,
            shard=shard,
            draws=args.draws,
            seed=args.seed + rank,
            context_sha256=plan["context_sha256"],
            checkpoint=args.checkpoint,
            candidate=args.candidate,
            quality_thresholds=args.quality_thresholds,
        )
        print(json.dumps({**existing, "reused": True}, indent=2), flush=True)
        return
    result = posterior_inference_shard(
        candidate_marker_path=args.candidate,
        context_path=args.context,
        checkpoint_path=args.checkpoint,
        output_path=output,
        start=int(shard["start"]),
        stop=int(shard["stop"]),
        draws=args.draws,
        seed=args.seed + rank,
        device="cuda",
        sample_chunk=args.sample_chunk,
        quality_thresholds_path=args.quality_thresholds,
    )
    if result.get("start") != shard["start"] or result.get("stop") != shard["stop"]:
        raise RuntimeError("blind worker output interval changed")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
