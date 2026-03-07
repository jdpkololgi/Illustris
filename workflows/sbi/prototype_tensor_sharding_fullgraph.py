"""Bounded tensor-sharding prototype on full Abacus cache tensors.

This is an experimental benchmark to answer the "brute-force tensor sharding"
question with measured step-time and memory behavior, without changing the
production trainer.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sbi-cache-path", required=True, help="Monolithic SBI cache path.")
    parser.add_argument("--steps", type=int, default=50, help="Timed benchmark steps.")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--node-latent", type=int, default=80)
    parser.add_argument("--edge-latent", type=int, default=80)
    parser.add_argument("--log-path", default="/pscratch/sd/d/dkololgi/logs/tensor_shard_prototype.json")
    return parser.parse_args()


def maybe_init_distributed() -> None:
    # Safe no-op if not launched multi-process.
    if jax.process_count() > 1:
        return
    if "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ:
        try:
            rank = int(os.environ["SLURM_PROCID"])
            world = int(os.environ["SLURM_NTASKS"])
            # Typical srun PMI environment in NERSC.
            coordinator = os.environ.get("COORDINATOR_ADDRESS")
            if coordinator is None:
                # Use first hostname + default port if not provided.
                hostlist = os.environ.get("SLURM_JOB_NODELIST", "")
                # keep simple: rely on user-provided COORDINATOR_ADDRESS if needed.
                if hostlist:
                    coordinator = "127.0.0.1:12345"
            if coordinator:
                jax.distributed.initialize(
                    coordinator_address=coordinator,
                    num_processes=world,
                    process_id=rank,
                )
        except Exception:
            # Prototype should still run in single-process mode if init fails.
            pass


def main() -> None:
    args = parse_args()
    maybe_init_distributed()

    with open(args.sbi_cache_path, "rb") as f:
        cache = pickle.load(f)

    graph = cache["graph"]
    x = np.asarray(graph.nodes, dtype=np.float32)
    e = np.asarray(graph.edges, dtype=np.float32)
    receivers = np.asarray(graph.receivers, dtype=np.int32)
    n_nodes = x.shape[0]

    devices = np.array(jax.devices())
    mesh = Mesh(devices, ("d",))
    shard_rows = NamedSharding(mesh, PartitionSpec("d", None))

    x_s = jax.device_put(jnp.array(x), shard_rows)
    e_s = jax.device_put(jnp.array(e), shard_rows)
    receivers_j = jnp.array(receivers, dtype=jnp.int32)

    w_node = jnp.ones((x.shape[1], args.node_latent), dtype=jnp.float32)
    w_edge = jnp.ones((e.shape[1], args.edge_latent), dtype=jnp.float32)

    @jax.jit
    def bench_step(xv, ev):
        node_h = jnp.tanh(xv @ w_node)
        edge_h = jnp.tanh(ev @ w_edge)
        # Small graph-like aggregation to exercise scatter semantics.
        edge_scalar = edge_h[:, 0]
        agg = jax.ops.segment_sum(edge_scalar, receivers_j, n_nodes)
        return jnp.mean(node_h) + 1e-6 * jnp.mean(agg)

    for _ in range(args.warmup_steps):
        _ = float(bench_step(x_s, e_s))

    t0 = time.time()
    vals = []
    for _ in range(args.steps):
        vals.append(float(bench_step(x_s, e_s)))
    elapsed = time.time() - t0
    sps = args.steps / max(elapsed, 1e-9)

    payload = {
        "cache_path": args.sbi_cache_path,
        "devices": [str(d) for d in jax.devices()],
        "process_count": int(jax.process_count()),
        "n_nodes": int(n_nodes),
        "n_edges": int(e.shape[0]),
        "steps": int(args.steps),
        "elapsed_sec": float(elapsed),
        "steps_per_sec": float(sps),
        "value_mean": float(np.mean(vals)) if vals else None,
    }
    log_path = Path(args.log_path).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
