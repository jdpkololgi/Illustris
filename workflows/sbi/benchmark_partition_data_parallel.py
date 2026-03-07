"""Benchmark partition-batch throughput in data-parallel launch settings.

This benchmark is designed for multi-GPU / multi-node `srun` launches where
each rank processes a disjoint stride of partition files.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-manifest", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--max-partitions", type=int, default=40)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--bench-steps", type=int, default=20)
    parser.add_argument("--log-dir", default="/pscratch/sd/d/dkololgi/logs")
    return parser.parse_args()


def _rank_info() -> tuple[int, int]:
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0")))
    world = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", "1")))
    return rank, world


def load_partition(path: Path) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    with np.load(path) as d:
        x = jnp.array(d["x"], dtype=jnp.float32)
        edge_attr = jnp.array(d["edge_attr"], dtype=jnp.float32)
        core_mask = jnp.array(d["core_mask_local"], dtype=bool)
    return x, edge_attr, core_mask


def main() -> None:
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    rank, world = _rank_info()

    with open(args.partition_manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    base = Path(args.partition_manifest).resolve().parent
    parts = [p for p in manifest["partitions"] if p["split"] == args.split]
    if args.max_partitions > 0:
        parts = parts[: args.max_partitions]
    local_parts = parts[rank::world]
    if not local_parts:
        print(f"[rank {rank}] no partitions assigned")
        return

    print(f"[rank {rank}/{world}] local partitions={len(local_parts)} devices={jax.devices()}")

    # Lightweight kernel that still touches node and edge tensors.
    @jax.jit
    def step_fn(x, e, core_mask):
        xh = jnp.tanh(x @ jnp.ones((x.shape[1], 64), dtype=jnp.float32))
        eh = jnp.tanh(e @ jnp.ones((e.shape[1], 64), dtype=jnp.float32))
        score = jnp.mean(xh[core_mask]) + 0.01 * jnp.mean(eh)
        return score

    # Warmup
    for i in range(min(args.warmup_steps, len(local_parts))):
        x, e, m = load_partition(base / local_parts[i]["file"])
        _ = float(step_fn(x, e, m))

    # Timed benchmark
    t0 = time.time()
    n_steps = min(args.bench_steps, len(local_parts))
    scores = []
    for i in range(n_steps):
        x, e, m = load_partition(base / local_parts[i]["file"])
        scores.append(float(step_fn(x, e, m)))
    elapsed = time.time() - t0
    pps = n_steps / max(elapsed, 1e-9)

    log_path = Path(args.log_dir) / f"partition_dp_bench_rank{rank:04d}.json"
    payload = {
        "rank": rank,
        "world_size": world,
        "split": args.split,
        "steps": n_steps,
        "elapsed_sec": elapsed,
        "partitions_per_sec": pps,
        "score_mean": float(np.mean(scores)) if scores else None,
        "manifest": args.partition_manifest,
    }
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[rank {rank}] steps={n_steps} elapsed={elapsed:.2f}s pps={pps:.3f} log={log_path}")


if __name__ == "__main__":
    main()
