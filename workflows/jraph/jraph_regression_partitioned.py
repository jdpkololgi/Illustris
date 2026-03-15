"""Partition-aware Jraph regression trainer for Abacus caches.

This script mirrors the partition orchestration style of
`workflows/sbi/jraph_sbi_flowjax_partitioned.py` but trains the deterministic
regression pipeline (MSE on eigenvalue targets) instead of a flow model.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from functools import partial
import json
import os
import pickle
from pathlib import Path
import subprocess
import time

import haiku as hk
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
import jraph
import numpy as np
import optax

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.config_paths import CANONICAL_OUTPUT_ROOT
from shared.eigenvalue_transformations import increments_to_eigenvalues
from shared.graph_net_models import make_graph_network
from shared.resource_requirements import require_gpu_slurm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-manifest", required=True, help="Path to partition_manifest.json")
    parser.add_argument(
        "--source-cache-path",
        default="",
        help=(
            "Optional path to source cache with scaler/raw eigenvalue metadata. "
            "Defaults to manifest['source_cache_path'] when present."
        ),
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-passes", type=int, default=8)
    parser.add_argument("--latent-size", type=int, default=80)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--train-partition-limit", type=int, default=0, help="0 means all train partitions.")
    parser.add_argument("--val-partition-limit", type=int, default=8, help="Quick validation partitions.")
    parser.add_argument(
        "--max-partition-nodes",
        type=int,
        default=0,
        help="Drop partitions with n_total_nodes above this limit (0 disables).",
    )
    parser.add_argument(
        "--max-partition-edges",
        type=int,
        default=0,
        help="Drop partitions with n_edges above this limit (0 disables).",
    )
    parser.add_argument("--full-val-every", type=int, default=25, help="Run full validation every N epochs.")
    parser.add_argument(
        "--full-val-partition-limit",
        type=int,
        default=0,
        help="If >0, cap full-validation partitions.",
    )
    parser.add_argument(
        "--train-partitions-per-epoch",
        type=int,
        default=0,
        help="If >0, sample this many train partitions per epoch.",
    )
    parser.add_argument(
        "--data-parallel",
        action="store_true",
        help="Enable local pmap data-parallel mode (one partition per device per step).",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable multi-process JAX distributed initialization from Slurm env vars.",
    )
    parser.add_argument(
        "--coordinator-address",
        default="",
        help="Optional coordinator host:port override for jax.distributed.initialize.",
    )
    parser.add_argument(
        "--bucket-span-multiplier",
        type=int,
        default=8,
        help="Window size multiplier for shape-bucketed pmap collation.",
    )
    parser.add_argument(
        "--bucket-sort-key",
        choices=("edges", "nodes", "max"),
        default="max",
        help="Metadata key used to bucket partitions by shape.",
    )
    parser.add_argument(
        "--pad-node-multiple",
        type=int,
        default=1024,
        help="Round padded node count up to this multiple (1 disables).",
    )
    parser.add_argument(
        "--pad-edge-multiple",
        type=int,
        default=32768,
        help="Round padded edge count up to this multiple (1 disables).",
    )
    parser.add_argument(
        "--mixed-precision",
        choices=("none", "bf16"),
        default="bf16",
        help="Mixed-precision compute mode (master params/optimizer remain fp32).",
    )
    parser.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable activation rematerialization for training forward pass.",
    )
    parser.add_argument(
        "--partition-cache-size",
        type=int,
        default=512,
        help="Max partition arrays in host-side LRU cache (0 disables).",
    )
    parser.add_argument(
        "--prefetch-workers",
        type=int,
        default=4,
        help="Thread workers for partition prefetching (0 disables).",
    )
    parser.add_argument(
        "--prefetch-lookahead-steps",
        type=int,
        default=4,
        help="How many future train/eval steps to prefetch.",
    )
    parser.add_argument(
        "--output-dir",
        default=f"{CANONICAL_OUTPUT_ROOT}/jraph_partitioned_regression",
        help="Directory for model/logs/predictions.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default="",
        help="Optional checkpoint .pkl created by this script.",
    )
    parser.add_argument(
        "--checkpoint-every-epochs",
        type=int,
        default=5,
        help="Write periodic checkpoints every N epochs (0 disables).",
    )
    parser.add_argument(
        "--train-progress-every",
        type=int,
        default=1,
        help="Print train progress every N partition steps.",
    )
    return parser.parse_args()


def _rank_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0")))
    world = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", "1")))
    local_rank = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", "0")))
    return rank, world, local_rank


def _discover_coordinator() -> str:
    override = os.environ.get("COORDINATOR_ADDRESS")
    if override:
        return override
    nodelist = os.environ.get("SLURM_NODELIST")
    if nodelist:
        out = subprocess.check_output(["scontrol", "show", "hostnames", nodelist], text=True)
        hosts = [x.strip() for x in out.splitlines() if x.strip()]
        if hosts:
            return f"{hosts[0]}:12355"
    return "127.0.0.1:12355"


def _infer_local_device_ids(local_rank: int) -> list[int] | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        ids = [x.strip() for x in visible.split(",") if x.strip()]
        tasks_per_node_raw = os.environ.get("SLURM_NTASKS_PER_NODE", "1")
        tasks_per_node = int(tasks_per_node_raw.split("(")[0].split(",")[0])
        if len(ids) == 1:
            return [0]
        if tasks_per_node == 1:
            return list(range(len(ids)))
        if len(ids) > 1:
            return [int(local_rank)]
    return None


def _maybe_init_distributed(args: argparse.Namespace, rank: int, world: int, local_rank: int) -> None:
    if not args.distributed and world <= 1:
        return
    if jax.distributed.is_initialized():
        return
    coordinator = args.coordinator_address or _discover_coordinator()
    local_device_ids = _infer_local_device_ids(local_rank)
    print(
        f"Initializing distributed runtime: coordinator={coordinator}, "
        f"process_id={rank}, num_processes={world}, local_rank={local_rank}, "
        f"local_device_ids={local_device_ids}",
        flush=True,
    )
    jax.distributed.initialize(
        coordinator_address=coordinator,
        num_processes=world,
        process_id=rank,
        local_device_ids=local_device_ids,
    )


def _compute_dtype_from_mode(mode: str) -> jnp.dtype:
    if mode == "bf16":
        return jnp.bfloat16
    return jnp.float32


def _shape_key(part: dict, mode: str) -> int:
    n_nodes = int(part.get("n_total_nodes", 0))
    n_edges = int(part.get("n_edges", 0))
    if mode == "nodes":
        return n_nodes
    if mode == "edges":
        return n_edges
    return max(n_nodes, n_edges)


def _build_epoch_groups(
    parts: list[dict],
    *,
    n_local_devices: int,
    rng_seed: int,
    bucket_span_multiplier: int,
    bucket_sort_key: str,
) -> list[list[dict]]:
    if n_local_devices <= 0:
        return []
    rng = np.random.default_rng(rng_seed)
    order = rng.permutation(len(parts))
    ordered = [parts[int(i)] for i in order]
    span = max(n_local_devices, n_local_devices * max(1, bucket_span_multiplier))
    grouped: list[list[dict]] = []
    for start in range(0, len(ordered), span):
        window = ordered[start : start + span]
        window.sort(key=lambda p: _shape_key(p, bucket_sort_key))
        for j in range(0, len(window), n_local_devices):
            batch = window[j : j + n_local_devices]
            if len(batch) == n_local_devices:
                grouped.append(batch)
    rng.shuffle(grouped)
    return grouped


class _PartitionArrayCache:
    def __init__(self, max_items: int):
        self._max_items = max(0, int(max_items))
        self._items: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def has(self, path: Path) -> bool:
        return str(path) in self._items

    def set(self, path: Path, value: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self._max_items <= 0:
            return value
        key = str(path)
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self._max_items:
            self._items.popitem(last=False)
        return value

    def get(self, path: Path) -> dict[str, np.ndarray]:
        key = str(path)
        if self._max_items > 0 and key in self._items:
            self._items.move_to_end(key)
            return self._items[key]
        return self.set(path, _load_partition_arrays(path))


def _load_partition_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as d:
        x = np.asarray(d["x"], dtype=np.float32)
        edge_index = np.asarray(d["edge_index"], dtype=np.int32)
        edge_attr = np.asarray(d["edge_attr"], dtype=np.float32)
        targets = np.asarray(d["targets"], dtype=np.float32)
        core_mask_local = np.asarray(d["core_mask_local"], dtype=bool)
        if "global_node_ids" in d:
            global_node_ids = np.asarray(d["global_node_ids"], dtype=np.int64)
        else:
            global_node_ids = np.arange(x.shape[0], dtype=np.int64)
    return {
        "x": x,
        "senders": edge_index[0],
        "receivers": edge_index[1],
        "edge_attr": edge_attr,
        "targets": targets,
        "core_mask": core_mask_local,
        "global_node_ids": global_node_ids,
        "n_nodes": np.int32(x.shape[0]),
        "n_edges": np.int32(edge_index.shape[1]),
    }


def load_partition(
    path: Path,
    *,
    compute_dtype: jnp.dtype = jnp.float32,
) -> tuple[jraph.GraphsTuple, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    d = _load_partition_arrays(path)
    graph = jraph.GraphsTuple(
        nodes=jnp.array(d["x"], dtype=compute_dtype),
        edges=jnp.array(d["edge_attr"], dtype=compute_dtype),
        senders=jnp.array(d["senders"], dtype=jnp.int32),
        receivers=jnp.array(d["receivers"], dtype=jnp.int32),
        n_node=jnp.array([d["n_nodes"]], dtype=jnp.int32),
        n_edge=jnp.array([d["n_edges"]], dtype=jnp.int32),
        globals=None,
    )
    return (
        graph,
        jnp.array(d["targets"], dtype=jnp.float32),
        jnp.array(d["core_mask"], dtype=bool),
        jnp.array(d["global_node_ids"], dtype=jnp.int64),
    )


def _collate_padded_partition_batch(
    base_dir: Path,
    batch_parts: list[dict],
    *,
    compute_dtype: jnp.dtype,
    pad_nodes: int | None = None,
    pad_edges: int | None = None,
    array_loader=None,
) -> tuple[jraph.GraphsTuple, jnp.ndarray, jnp.ndarray]:
    loader = _load_partition_arrays if array_loader is None else array_loader
    loaded = [loader(base_dir / p["file"]) for p in batch_parts]
    n_dev = len(loaded)
    local_max_nodes = max(int(x["n_nodes"]) for x in loaded)
    local_max_edges = max(int(x["n_edges"]) for x in loaded)
    max_nodes = local_max_nodes if pad_nodes is None else max(local_max_nodes, int(pad_nodes))
    max_edges = local_max_edges if pad_edges is None else max(local_max_edges, int(pad_edges))
    node_feat_dim = int(loaded[0]["x"].shape[1])
    edge_feat_dim = int(loaded[0]["edge_attr"].shape[1])
    target_dim = int(loaded[0]["targets"].shape[1])

    nodes = np.zeros((n_dev, max_nodes, node_feat_dim), dtype=np.float32)
    targets = np.zeros((n_dev, max_nodes, target_dim), dtype=np.float32)
    core_mask = np.zeros((n_dev, max_nodes), dtype=bool)
    node_valid_mask = np.zeros((n_dev, max_nodes), dtype=bool)
    edge_attr = np.zeros((n_dev, max_edges, edge_feat_dim), dtype=np.float32)

    dummy_idx = np.int32(max_nodes - 1)
    senders = np.full((n_dev, max_edges), dummy_idx, dtype=np.int32)
    receivers = np.full((n_dev, max_edges), dummy_idx, dtype=np.int32)
    n_node = np.zeros((n_dev, 1), dtype=np.int32)
    n_edge = np.zeros((n_dev, 1), dtype=np.int32)

    for i, d in enumerate(loaded):
        nn = int(d["n_nodes"])
        ne = int(d["n_edges"])
        nodes[i, :nn, :] = d["x"]
        targets[i, :nn, :] = d["targets"]
        core_mask[i, :nn] = d["core_mask"]
        node_valid_mask[i, :nn] = True
        edge_attr[i, :ne, :] = d["edge_attr"]
        senders[i, :ne] = d["senders"]
        receivers[i, :ne] = d["receivers"]
        n_node[i, 0] = np.int32(nn)
        n_edge[i, 0] = np.int32(ne)

    graph = jraph.GraphsTuple(
        nodes=jnp.array(nodes, dtype=compute_dtype),
        edges=jnp.array(edge_attr, dtype=compute_dtype),
        senders=jnp.array(senders),
        receivers=jnp.array(receivers),
        n_node=jnp.array(n_node),
        n_edge=jnp.array(n_edge),
        globals=None,
    )
    return graph, jnp.array(targets, dtype=jnp.float32), jnp.array(core_mask & node_valid_mask, dtype=bool)


def _part_node_edge_counts(part: dict) -> tuple[int, int]:
    n_nodes = int(part.get("n_total_nodes", part.get("n_nodes", part.get("n_core_nodes", 0))))
    n_edges = int(part.get("n_edges", 0))
    return n_nodes, n_edges


def _batch_node_edge_bounds(batch_parts: list[dict]) -> tuple[int, int]:
    max_nodes = 0
    max_edges = 0
    for p in batch_parts:
        n_nodes, n_edges = _part_node_edge_counts(p)
        max_nodes = max(max_nodes, n_nodes)
        max_edges = max(max_edges, n_edges)
    return max_nodes, max_edges


def _round_up(value: int, multiple: int) -> int:
    m = max(1, int(multiple))
    return int(((max(1, int(value)) + m - 1) // m) * m)


def _global_pad_shape(
    max_nodes: int,
    max_edges: int,
    world: int,
    *,
    node_multiple: int = 1,
    edge_multiple: int = 1,
) -> tuple[int, int]:
    if world <= 1:
        return _round_up(max_nodes, node_multiple), _round_up(max_edges, edge_multiple)
    local_shape = np.array([[int(max_nodes), int(max_edges)]], dtype=np.int32)
    all_shapes = np.asarray(multihost_utils.process_allgather(local_shape)).reshape(-1, 2)
    return _round_up(int(all_shapes[:, 0].max()), node_multiple), _round_up(
        int(all_shapes[:, 1].max()), edge_multiple
    )


def _filter_partitions_by_size(
    parts: list[dict],
    *,
    max_nodes: int,
    max_edges: int,
) -> tuple[list[dict], int]:
    if max_nodes <= 0 and max_edges <= 0:
        return parts, 0
    kept: list[dict] = []
    dropped = 0
    for p in parts:
        n_nodes, n_edges = _part_node_edge_counts(p)
        if max_nodes > 0 and n_nodes > max_nodes:
            dropped += 1
            continue
        if max_edges > 0 and n_edges > max_edges:
            dropped += 1
            continue
        kept.append(p)
    return kept, dropped


def _schedule_prefetch(
    *,
    step_idx: int,
    groups: list[list[dict]],
    base_dir: Path,
    lookahead_steps: int,
    cache: _PartitionArrayCache,
    inflight: dict[str, Future],
    pool: ThreadPoolExecutor | None,
) -> None:
    if pool is None or lookahead_steps <= 0:
        return
    end = min(len(groups), step_idx + 1 + lookahead_steps)
    for idx in range(step_idx + 1, end):
        for p in groups[idx]:
            path = base_dir / p["file"]
            key = str(path)
            if cache.has(path) or key in inflight:
                continue
            inflight[key] = pool.submit(_load_partition_arrays, path)


def _truncate_for_distributed(parts: list[dict], rank: int, world: int, n_local_devices: int) -> list[dict]:
    local = parts[rank::world]
    if world <= 1:
        return local
    local_n = np.array([len(local)], dtype=np.int32)
    all_n = np.asarray(multihost_utils.process_allgather(local_n)).reshape(-1)
    min_n = int(all_n.min())
    full_groups = (min_n // max(1, n_local_devices)) * max(1, n_local_devices)
    return local[:full_groups]


def _take_first_replica(tree):
    return jax.tree_util.tree_map(lambda x: x[0], tree)


def _compute_regression_metrics(preds: np.ndarray, targets: np.ndarray) -> dict[str, object]:
    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))
    ss_res = np.sum((targets - preds) ** 2, axis=0)
    ss_tot = np.sum((targets - np.mean(targets, axis=0)) ** 2, axis=0)
    r2_per_dim = 1.0 - ss_res / (ss_tot + 1e-8)
    return {
        "mse": mse,
        "mae": mae,
        "r2_per_dim": r2_per_dim.tolist(),
        "r2_mean": float(np.mean(r2_per_dim)),
    }


def _serialize_rng_key(rng_key: jax.Array) -> np.ndarray:
    """Serialize typed PRNG key to plain uint32 array for checkpoints."""
    return np.asarray(jax.random.key_data(rng_key), dtype=np.uint32)


def _deserialize_rng_key(rng_payload) -> jax.Array:
    """Load PRNG key from either new uint32 payload or legacy checkpoint formats."""
    arr = jnp.asarray(rng_payload)
    # New format: raw key data (uint32[2]) -> typed key.
    if arr.dtype == jnp.uint32:
        return jax.random.wrap_key_data(arr)
    # Legacy fallback: typed key may be directly stored.
    if getattr(arr.dtype, "name", "") == "key<fry>":
        return arr
    # Last-resort compatibility for older ad-hoc formats.
    return jax.random.wrap_key_data(arr.astype(jnp.uint32))


def main(args: argparse.Namespace) -> None:
    rank, world, local_rank = _rank_info()
    _maybe_init_distributed(args, rank, world, local_rank)
    require_gpu_slurm("jraph_regression_partitioned.py", min_gpus=1)
    os.makedirs(args.output_dir, exist_ok=True)

    local_devices = jax.local_devices()
    n_local_devices = len(local_devices)
    if n_local_devices < 1:
        raise RuntimeError("No local devices available after distributed initialization.")
    compute_dtype = _compute_dtype_from_mode(args.mixed_precision)

    with open(args.partition_manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    base_dir = Path(args.partition_manifest).resolve().parent

    source_cache_path = args.source_cache_path
    if not source_cache_path:
        source_cache_path = manifest.get("source_cache_path", "")
    source_cache = None
    if source_cache_path and Path(source_cache_path).exists():
        with open(source_cache_path, "rb") as f:
            source_cache = pickle.load(f)

    train_parts = [p for p in manifest["partitions"] if p["split"] == "train"]
    val_parts_full = [p for p in manifest["partitions"] if p["split"] == "val"]
    test_parts = [p for p in manifest["partitions"] if p["split"] == "test"]
    train_parts, dropped_train = _filter_partitions_by_size(
        train_parts,
        max_nodes=args.max_partition_nodes,
        max_edges=args.max_partition_edges,
    )
    val_parts_full, dropped_val = _filter_partitions_by_size(
        val_parts_full,
        max_nodes=args.max_partition_nodes,
        max_edges=args.max_partition_edges,
    )
    test_parts, dropped_test = _filter_partitions_by_size(
        test_parts,
        max_nodes=args.max_partition_nodes,
        max_edges=args.max_partition_edges,
    )
    if rank == 0 and (dropped_train or dropped_val or dropped_test):
        print(
            f"Dropped oversized partitions: train={dropped_train}, val={dropped_val}, test={dropped_test} "
            f"(max_nodes={args.max_partition_nodes or 'off'}, max_edges={args.max_partition_edges or 'off'})",
            flush=True,
        )

    if args.train_partition_limit > 0:
        train_parts = train_parts[: args.train_partition_limit]
    if args.full_val_partition_limit > 0:
        val_parts_full = val_parts_full[: args.full_val_partition_limit]
    val_parts_quick = val_parts_full if args.val_partition_limit <= 0 else val_parts_full[: args.val_partition_limit]

    train_parts = _truncate_for_distributed(train_parts, rank, world, n_local_devices)
    val_parts_full = _truncate_for_distributed(val_parts_full, rank, world, n_local_devices)
    val_parts_quick = _truncate_for_distributed(val_parts_quick, rank, world, n_local_devices)
    test_parts = _truncate_for_distributed(test_parts, rank, world, n_local_devices)

    if not train_parts:
        raise ValueError("No train partitions found after rank/world assignment.")

    print("=" * 70, flush=True)
    print("Partitioned Jraph Regression Trainer", flush=True)
    print("=" * 70, flush=True)
    print(f"Rank {rank}/{world} | local_devices={n_local_devices}", flush=True)
    print(
        f"Partitions train/val_quick/val_full/test = "
        f"{len(train_parts)}/{len(val_parts_quick)}/{len(val_parts_full)}/{len(test_parts)}",
        flush=True,
    )
    print(
        f"Config: epochs={args.epochs}, lr={args.lr}, latent={args.latent_size}, "
        f"passes={args.num_passes}, data_parallel={args.data_parallel}, "
        f"distributed={args.distributed or world > 1}, mixed_precision={args.mixed_precision}",
        flush=True,
    )

    rng = jax.random.key(args.seed)
    first_graph, _, _, _ = load_partition(base_dir / train_parts[0]["file"], compute_dtype=compute_dtype)
    net_fn = make_graph_network(
        num_passes=args.num_passes,
        latent_size=args.latent_size,
        num_heads=args.num_heads,
        dropout_rate=args.dropout,
        output_dim=3,
    )
    net = hk.transform(net_fn)
    rng, init_key = jax.random.split(rng)
    params = net.init(init_key, first_graph, is_training=True)

    warmup_steps = min(200, max(1, args.epochs // 10))
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.lr,
        warmup_steps=warmup_steps,
        decay_steps=max(args.epochs, warmup_steps + 1),
        end_value=1e-5,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(lr_schedule, weight_decay=args.weight_decay),
    )
    opt_state = optimizer.init(params)

    def _forward_train(p, key, graph):
        return net.apply(p, key, graph, is_training=True).nodes

    def _forward_eval(p, key, graph):
        return net.apply(p, key, graph, is_training=False).nodes

    if args.activation_checkpointing:
        _forward_train = jax.checkpoint(_forward_train, prevent_cse=False)

    def _masked_sums(preds, targets, core_mask):
        per_node_mse = jnp.mean((preds - targets) ** 2, axis=-1)
        per_node_mae = jnp.mean(jnp.abs(preds - targets), axis=-1)
        mask_f = core_mask.astype(jnp.float32)
        mse_sum = jnp.sum(per_node_mse * mask_f)
        mae_sum = jnp.sum(per_node_mae * mask_f)
        n_core = jnp.sum(mask_f)
        return mse_sum, mae_sum, n_core

    def train_loss_fn(p, graph, targets, core_mask, step_key):
        preds = _forward_train(p, step_key, graph)
        mse_sum, mae_sum, n_core = _masked_sums(preds, targets, core_mask)
        mse = mse_sum / jnp.maximum(n_core, 1.0)
        return mse, (preds, mae_sum, n_core)

    def eval_loss_fn(p, graph, targets, core_mask, step_key):
        preds = _forward_eval(p, step_key, graph)
        mse_sum, mae_sum, n_core = _masked_sums(preds, targets, core_mask)
        mse = mse_sum / jnp.maximum(n_core, 1.0)
        mae = mae_sum / jnp.maximum(n_core, 1.0)
        return mse, mae

    @jax.jit
    def train_step_single(p, o_state, graph, targets, core_mask, step_key):
        (loss, (preds, mae_sum, n_core)), grads = jax.value_and_grad(
            lambda p_: train_loss_fn(p_, graph, targets, core_mask, step_key),
            has_aux=True,
        )(p)
        updates, o_state_new = optimizer.update(grads, o_state, p)
        p_new = optax.apply_updates(p, updates)
        mae = mae_sum / jnp.maximum(n_core, 1.0)
        return p_new, o_state_new, loss, mae, preds

    @jax.jit
    def eval_step_single(p, graph, targets, core_mask, step_key):
        preds = _forward_eval(p, step_key, graph)
        mse_sum, mae_sum, n_core = _masked_sums(preds, targets, core_mask)
        mse = mse_sum / jnp.maximum(n_core, 1.0)
        mae = mae_sum / jnp.maximum(n_core, 1.0)
        return mse, mae, preds

    @partial(jax.pmap, axis_name="i")
    def train_step_dp(p, o_state, graph, targets, core_mask, step_keys):
        device_key = jax.random.fold_in(step_keys, jax.lax.axis_index("i"))

        def _loss_fn(p_):
            preds = _forward_train(p_, device_key, graph)
            mse_sum, _, n_core = _masked_sums(preds, targets, core_mask)
            global_mse_sum = jax.lax.psum(mse_sum, axis_name="i")
            global_n_core = jax.lax.psum(n_core, axis_name="i")
            return global_mse_sum / jnp.maximum(global_n_core, 1.0), preds

        (loss, preds), grads = jax.value_and_grad(_loss_fn, has_aux=True)(p)
        grads = jax.lax.pmean(grads, axis_name="i")
        updates, o_state_new = optimizer.update(grads, o_state, p)
        p_new = optax.apply_updates(p, updates)

        mse_sum, mae_sum, n_core = _masked_sums(preds, targets, core_mask)
        global_mse = jax.lax.psum(mse_sum, axis_name="i") / jnp.maximum(
            jax.lax.psum(n_core, axis_name="i"), 1.0
        )
        global_mae = jax.lax.psum(mae_sum, axis_name="i") / jnp.maximum(
            jax.lax.psum(n_core, axis_name="i"), 1.0
        )
        return p_new, o_state_new, global_mse, global_mae

    @partial(jax.pmap, axis_name="i")
    def eval_step_dp(p, graph, targets, core_mask, step_keys):
        device_key = jax.random.fold_in(step_keys, jax.lax.axis_index("i"))
        preds = _forward_eval(p, device_key, graph)
        mse_sum, mae_sum, n_core = _masked_sums(preds, targets, core_mask)
        global_mse = jax.lax.psum(mse_sum, axis_name="i") / jnp.maximum(
            jax.lax.psum(n_core, axis_name="i"), 1.0
        )
        global_mae = jax.lax.psum(mae_sum, axis_name="i") / jnp.maximum(
            jax.lax.psum(n_core, axis_name="i"), 1.0
        )
        return global_mse, global_mae

    array_cache = _PartitionArrayCache(args.partition_cache_size)
    prefetch_pool = ThreadPoolExecutor(max_workers=max(1, args.prefetch_workers)) if args.prefetch_workers > 0 else None
    inflight_prefetch: dict[str, Future] = {}

    def _load_arrays_cached(path: Path) -> dict[str, np.ndarray]:
        key = str(path)
        fut = inflight_prefetch.pop(key, None)
        if fut is not None:
            return array_cache.set(path, fut.result())
        return array_cache.get(path)

    rep_params = None
    rep_opt_state = None
    if args.data_parallel:
        rep_params = jax.device_put_replicated(params, local_devices)
        rep_opt_state = jax.device_put_replicated(opt_state, local_devices)

    history = {"train": [], "val": [], "val_kind": []}
    best_val = float("inf")
    best_params = None
    start_epoch = 0

    ckpt_latest = Path(args.output_dir) / "checkpoint_latest.pkl"
    if args.resume_checkpoint:
        ckpt_path = Path(args.resume_checkpoint).expanduser()
        with ckpt_path.open("rb") as f:
            ckpt = pickle.load(f)
        start_epoch = int(ckpt.get("next_epoch", 0))
        best_val = float(ckpt.get("best_val", best_val))
        history = ckpt.get("history", history)
        rng = _deserialize_rng_key(ckpt["rng"])
        params = ckpt["params"]
        opt_state = ckpt["opt_state"]
        if args.data_parallel:
            rep_params = jax.device_put_replicated(params, local_devices)
            rep_opt_state = jax.device_put_replicated(opt_state, local_devices)
        if rank == 0:
            print(f"Resumed from checkpoint {ckpt_path} at epoch {start_epoch}", flush=True)

    def _snapshot_unreplicated():
        if args.data_parallel:
            return jax.device_get(_take_first_replica(rep_params)), jax.device_get(_take_first_replica(rep_opt_state))
        return jax.device_get(params), jax.device_get(opt_state)

    def _save_checkpoint(next_epoch: int, tagged: bool) -> None:
        if rank != 0:
            return
        p_host, opt_host = _snapshot_unreplicated()
        payload = {
            "next_epoch": int(next_epoch),
            "rng": _serialize_rng_key(rng),
            "params": p_host,
            "opt_state": opt_host,
            "best_val": float(best_val),
            "history": history,
            "config": vars(args),
            "saved_at_unix_s": float(time.time()),
        }
        with ckpt_latest.open("wb") as f:
            pickle.dump(payload, f)
        if tagged:
            tag = Path(args.output_dir) / f"checkpoint_epoch_{int(next_epoch):05d}.pkl"
            with tag.open("wb") as f:
                pickle.dump(payload, f)
            print(f"Saved checkpoint: {tag}", flush=True)

    t0 = time.time()
    try:
        for epoch in range(start_epoch, start_epoch + args.epochs):
            rng, ep_key = jax.random.split(rng)
            train_mse_vals: list[float] = []
            train_mae_vals: list[float] = []

            if args.data_parallel:
                epoch_train_parts = train_parts
                if args.train_partitions_per_epoch > 0 and args.train_partitions_per_epoch < len(train_parts):
                    ep_rng = np.random.default_rng(args.seed + 7919 * (epoch + 1) + rank * 104729)
                    chosen = ep_rng.choice(len(train_parts), size=args.train_partitions_per_epoch, replace=False)
                    epoch_train_parts = [train_parts[int(i)] for i in chosen]

                train_groups = _build_epoch_groups(
                    epoch_train_parts,
                    n_local_devices=n_local_devices,
                    rng_seed=args.seed + epoch + rank * 100003,
                    bucket_span_multiplier=args.bucket_span_multiplier,
                    bucket_sort_key=args.bucket_sort_key,
                )
                n_train_steps = len(train_groups)
                for step_idx, batch_parts in enumerate(train_groups):
                    _schedule_prefetch(
                        step_idx=step_idx,
                        groups=train_groups,
                        base_dir=base_dir,
                        lookahead_steps=args.prefetch_lookahead_steps,
                        cache=array_cache,
                        inflight=inflight_prefetch,
                        pool=prefetch_pool,
                    )
                    if rank == 0 and step_idx % max(1, args.train_progress_every) == 0:
                        print(
                            f"  [train] epoch={epoch:04d} step={step_idx + 1}/{n_train_steps}",
                            flush=True,
                        )
                    pad_nodes, pad_edges = _batch_node_edge_bounds(batch_parts)
                    pad_nodes, pad_edges = _global_pad_shape(
                        pad_nodes,
                        pad_edges,
                        world,
                        node_multiple=args.pad_node_multiple,
                        edge_multiple=args.pad_edge_multiple,
                    )
                    graph_b, targets_b, core_mask_b = _collate_padded_partition_batch(
                        base_dir,
                        batch_parts,
                        compute_dtype=compute_dtype,
                        pad_nodes=pad_nodes,
                        pad_edges=pad_edges,
                        array_loader=_load_arrays_cached,
                    )
                    ep_key, step_key = jax.random.split(ep_key)
                    step_key = jax.random.fold_in(step_key, rank)
                    step_keys = jax.random.split(step_key, n_local_devices)
                    rep_params, rep_opt_state, mse_rep, mae_rep = train_step_dp(
                        rep_params,
                        rep_opt_state,
                        graph_b,
                        targets_b,
                        core_mask_b,
                        step_keys,
                    )
                    train_mse_vals.append(float(jax.device_get(mse_rep[0])))
                    train_mae_vals.append(float(jax.device_get(mae_rep[0])))
            else:
                epoch_train_parts = train_parts
                if args.train_partitions_per_epoch > 0 and args.train_partitions_per_epoch < len(train_parts):
                    ep_rng = np.random.default_rng(args.seed + 7919 * (epoch + 1) + rank * 104729)
                    chosen = ep_rng.choice(len(train_parts), size=args.train_partitions_per_epoch, replace=False)
                    epoch_train_parts = [train_parts[int(i)] for i in chosen]
                order = np.random.default_rng(args.seed + epoch).permutation(len(epoch_train_parts))
                n_train_steps = len(order)
                for step_idx, i in enumerate(order):
                    p = epoch_train_parts[int(i)]
                    if rank == 0 and step_idx % max(1, args.train_progress_every) == 0:
                        print(
                            f"  [train] epoch={epoch:04d} step={step_idx + 1}/{n_train_steps} "
                            f"partition={p['partition_id']}",
                            flush=True,
                        )
                    graph, targets, core_mask, _ = load_partition(base_dir / p["file"], compute_dtype=compute_dtype)
                    ep_key, step_key = jax.random.split(ep_key)
                    params, opt_state, mse, mae, _ = train_step_single(
                        params, opt_state, graph, targets, core_mask, step_key
                    )
                    train_mse_vals.append(float(mse))
                    train_mae_vals.append(float(mae))

            mean_train_mse = float(np.mean(train_mse_vals)) if train_mse_vals else float("nan")
            mean_train_mae = float(np.mean(train_mae_vals)) if train_mae_vals else float("nan")
            history["train"].append((epoch, mean_train_mse, mean_train_mae))

            if epoch % args.eval_every == 0:
                do_full_val = args.full_val_every > 0 and ((epoch + 1) % args.full_val_every == 0)
                epoch_val_parts = val_parts_full if do_full_val else val_parts_quick
                val_kind = "full" if do_full_val else "quick"
                val_mse_vals: list[float] = []
                val_mae_vals: list[float] = []
                if args.data_parallel:
                    val_sorted = sorted(epoch_val_parts, key=lambda p: _shape_key(p, args.bucket_sort_key))
                    val_groups = [val_sorted[i : i + n_local_devices] for i in range(0, len(val_sorted), n_local_devices)]
                    val_groups = [g for g in val_groups if len(g) == n_local_devices]
                    for step_idx, batch_parts in enumerate(val_groups):
                        _schedule_prefetch(
                            step_idx=step_idx,
                            groups=val_groups,
                            base_dir=base_dir,
                            lookahead_steps=args.prefetch_lookahead_steps,
                            cache=array_cache,
                            inflight=inflight_prefetch,
                            pool=prefetch_pool,
                        )
                        pad_nodes, pad_edges = _batch_node_edge_bounds(batch_parts)
                        pad_nodes, pad_edges = _global_pad_shape(
                            pad_nodes,
                            pad_edges,
                            world,
                            node_multiple=args.pad_node_multiple,
                            edge_multiple=args.pad_edge_multiple,
                        )
                        graph_b, targets_b, core_mask_b = _collate_padded_partition_batch(
                            base_dir,
                            batch_parts,
                            compute_dtype=compute_dtype,
                            pad_nodes=pad_nodes,
                            pad_edges=pad_edges,
                            array_loader=_load_arrays_cached,
                        )
                        ep_key, step_key = jax.random.split(ep_key)
                        step_key = jax.random.fold_in(step_key, rank)
                        step_keys = jax.random.split(step_key, n_local_devices)
                        mse_rep, mae_rep = eval_step_dp(rep_params, graph_b, targets_b, core_mask_b, step_keys)
                        val_mse_vals.append(float(jax.device_get(mse_rep[0])))
                        val_mae_vals.append(float(jax.device_get(mae_rep[0])))
                else:
                    for p in epoch_val_parts:
                        graph, targets, core_mask, _ = load_partition(base_dir / p["file"], compute_dtype=compute_dtype)
                        ep_key, step_key = jax.random.split(ep_key)
                        val_mse, val_mae, _ = eval_step_single(params, graph, targets, core_mask, step_key)
                        val_mse_vals.append(float(val_mse))
                        val_mae_vals.append(float(val_mae))

                mean_val_mse = float(np.mean(val_mse_vals)) if val_mse_vals else float("nan")
                mean_val_mae = float(np.mean(val_mae_vals)) if val_mae_vals else float("nan")
                history["val"].append((epoch, mean_val_mse, mean_val_mae))
                history["val_kind"].append((epoch, val_kind))
                if mean_val_mse < best_val:
                    best_val = mean_val_mse
                    if args.data_parallel:
                        best_params = jax.device_get(_take_first_replica(rep_params))
                    else:
                        best_params = jax.device_get(params)
                if rank == 0:
                    print(
                        f"Epoch {epoch:04d} | train_mse={mean_train_mse:.6f} | train_mae={mean_train_mae:.6f} | "
                        f"val_mse={mean_val_mse:.6f} | val_mae={mean_val_mae:.6f} ({val_kind})",
                        flush=True,
                    )
            else:
                if rank == 0:
                    print(
                        f"Epoch {epoch:04d} | train_mse={mean_train_mse:.6f} | train_mae={mean_train_mae:.6f}",
                        flush=True,
                    )

            if args.checkpoint_every_epochs > 0 and ((epoch + 1) % args.checkpoint_every_epochs == 0):
                _save_checkpoint(epoch + 1, tagged=True)
    finally:
        if prefetch_pool is not None:
            prefetch_pool.shutdown(wait=False)

    elapsed = time.time() - t0
    if best_params is None:
        if args.data_parallel:
            best_params = jax.device_get(_take_first_replica(rep_params))
        else:
            best_params = jax.device_get(params)

    if rank == 0:
        print(f"Training finished in {elapsed:.1f}s | best_val_mse={best_val:.6f}", flush=True)
        _save_checkpoint(start_epoch + args.epochs, tagged=False)

    if rank != 0:
        return

    eval_net = hk.transform(net_fn)

    @jax.jit
    def _predict(p, graph, key):
        return eval_net.apply(p, key, graph, is_training=False).nodes

    test_preds_chunks = []
    test_targets_chunks = []
    test_global_ids_chunks = []
    rng, eval_base_key = jax.random.split(rng)
    for p in test_parts:
        graph, targets, core_mask, global_ids = load_partition(base_dir / p["file"], compute_dtype=compute_dtype)
        eval_base_key, step_key = jax.random.split(eval_base_key)
        preds = np.array(_predict(best_params, graph, step_key))
        targets_np = np.array(targets)
        core_np = np.array(core_mask)
        gids_np = np.array(global_ids)
        if np.any(core_np):
            test_preds_chunks.append(preds[core_np])
            test_targets_chunks.append(targets_np[core_np])
            test_global_ids_chunks.append(gids_np[core_np])

    if test_preds_chunks:
        test_preds = np.concatenate(test_preds_chunks, axis=0)
        test_targets = np.concatenate(test_targets_chunks, axis=0)
        test_global_ids = np.concatenate(test_global_ids_chunks, axis=0)
    else:
        test_preds = np.zeros((0, 3), dtype=np.float32)
        test_targets = np.zeros((0, 3), dtype=np.float32)
        test_global_ids = np.zeros((0,), dtype=np.int64)

    metrics_scaled = _compute_regression_metrics(test_preds, test_targets) if len(test_preds) > 0 else None
    metrics_raw = None
    preds_raw = None
    targets_raw = None

    use_transformed = False
    scaler = None
    eigenvalues_raw = None
    if source_cache is not None:
        scaler = source_cache.get("target_scaler")
        eigenvalues_raw = source_cache.get("eigenvalues_raw")
        use_transformed = source_cache.get("stats") is not None

    if scaler is not None and len(test_preds) > 0:
        preds_unscaled = scaler.inverse_transform(test_preds)
        targets_unscaled = scaler.inverse_transform(test_targets)
        if use_transformed:
            preds_raw = np.array(increments_to_eigenvalues(jnp.array(preds_unscaled)))
            targets_raw = np.array(increments_to_eigenvalues(jnp.array(targets_unscaled)))
        else:
            preds_raw = preds_unscaled
            targets_raw = targets_unscaled
        metrics_raw = _compute_regression_metrics(preds_raw, targets_raw)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(args.output_dir, f"jraph_regression_partitioned_model_seed_{args.seed}_{timestamp}.pkl")
    logs_path = os.path.join(args.output_dir, f"jraph_regression_partitioned_logs_seed_{args.seed}_{timestamp}.pkl")
    preds_path = os.path.join(args.output_dir, f"jraph_regression_partitioned_test_preds_seed_{args.seed}_{timestamp}.pkl")
    report_path = os.path.join(args.output_dir, f"jraph_regression_partitioned_report_seed_{args.seed}_{timestamp}.txt")

    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "params": best_params,
                "config": vars(args),
                "partition_manifest": args.partition_manifest,
                "source_cache_path": source_cache_path,
                "best_val_mse": best_val,
            },
            f,
        )
    with open(logs_path, "wb") as f:
        pickle.dump(history, f)
    with open(preds_path, "wb") as f:
        pickle.dump(
            {
                "global_node_ids": test_global_ids,
                "preds_scaled": test_preds,
                "targets_scaled": test_targets,
                "preds_raw": preds_raw,
                "targets_raw": targets_raw,
                "metrics_scaled": metrics_scaled,
                "metrics_raw": metrics_raw,
                "used_transformed_targets": use_transformed,
            },
            f,
        )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Partitioned Jraph Regression Report\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"Train runtime (s): {elapsed:.2f}\n")
        f.write(f"Best validation MSE: {best_val:.6f}\n")
        f.write(f"Test core samples: {len(test_preds):,}\n\n")
        if metrics_scaled is not None:
            f.write("Scaled target-space metrics (test core nodes):\n")
            f.write(f"  MSE: {metrics_scaled['mse']:.6f}\n")
            f.write(f"  MAE: {metrics_scaled['mae']:.6f}\n")
            f.write(
                "  R2 per dim: "
                + ", ".join(f"{x:.4f}" for x in metrics_scaled["r2_per_dim"])
                + "\n"
            )
            f.write(f"  Mean R2: {metrics_scaled['r2_mean']:.4f}\n\n")
        if metrics_raw is not None:
            f.write("Raw eigenvalue-space metrics (test core nodes):\n")
            f.write(f"  MSE: {metrics_raw['mse']:.6f}\n")
            f.write(f"  MAE: {metrics_raw['mae']:.6f}\n")
            f.write(
                "  R2 per eigenvalue: "
                + ", ".join(f"{x:.4f}" for x in metrics_raw["r2_per_dim"])
                + "\n"
            )
            f.write(f"  Mean R2: {metrics_raw['r2_mean']:.4f}\n")

    print(f"Saved model: {model_path}", flush=True)
    print(f"Saved logs: {logs_path}", flush=True)
    print(f"Saved predictions: {preds_path}", flush=True)
    print(f"Saved report: {report_path}", flush=True)


if __name__ == "__main__":
    main(parse_args())
