"""Partition-aware SBI trainer for Abacus caches.

This entrypoint keeps `jraph_sbi_flowjax.py` untouched and trains using
partition artifacts produced by `build_abacus_partition_batches.py`.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
import json
import os
import pickle
import subprocess
import time
from functools import partial
from pathlib import Path
from typing import Callable

import equinox as eqx
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
from flowjax.distributions import Normal
from flowjax.flows import RationalQuadraticSpline, masked_autoregressive_flow
from jax.experimental import multihost_utils

# Allow workflow script to resolve repo-root modules after reorganization.
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.config_paths import CANONICAL_OUTPUT_ROOT
from shared.graph_net_models import make_gnn_encoder
from shared.resource_requirements import require_gpu_slurm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-manifest", required=True, help="Path to partition_manifest.json")
    parser.add_argument("--sbi-cache-path", required=True, help="Original SBI cache path (for scaler/raw eig metadata).")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-passes", type=int, default=4)
    parser.add_argument("--latent-size", type=int, default=80)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-flow-layers", type=int, default=5)
    parser.add_argument("--num-bins", type=int, default=8)
    parser.add_argument("--flow-hidden-size", type=int, default=128)
    parser.add_argument(
        "--mixed-precision",
        choices=("none", "bf16"),
        default="bf16",
        help="Mixed-precision compute mode (bf16 keeps optimizer/master params in fp32).",
    )
    parser.add_argument("--train-partition-limit", type=int, default=0, help="0 means all train partitions.")
    parser.add_argument("--val-partition-limit", type=int, default=8, help="Validation partitions per eval.")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument(
        "--data-parallel",
        action="store_true",
        help=(
            "Enable data parallel training with pmap. Partitions are bucketed and padded "
            "within each multi-device step."
        ),
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
        help="Number of device-groups per bucketized window used for pmap collation.",
    )
    parser.add_argument(
        "--bucket-sort-key",
        choices=("edges", "nodes", "max"),
        default="max",
        help="Metadata key used to bucket partitions by shape before pmap collation.",
    )
    parser.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable activation rematerialization (jax.checkpoint) for the GNN forward pass "
            "during training to reduce peak memory usage."
        ),
    )
    parser.add_argument(
        "--train-partitions-per-epoch",
        type=int,
        default=0,
        help="If >0, sample this many train partitions per epoch (after distributed truncation).",
    )
    parser.add_argument(
        "--full-val-every",
        type=int,
        default=25,
        help="Run full validation every N epochs. Set <=0 to disable full validation passes.",
    )
    parser.add_argument(
        "--full-val-partition-limit",
        type=int,
        default=0,
        help="If >0, cap full-validation partitions to this many after split filtering.",
    )
    parser.add_argument(
        "--partition-cache-size",
        type=int,
        default=512,
        help="Max number of loaded partition arrays to keep in host RAM cache (0 disables).",
    )
    parser.add_argument(
        "--prefetch-workers",
        type=int,
        default=4,
        help="Thread workers for partition prefetching (0 disables prefetch).",
    )
    parser.add_argument(
        "--prefetch-lookahead-steps",
        type=int,
        default=4,
        help="How many future training/eval steps to prefetch partition arrays for.",
    )
    parser.add_argument(
        "--pad-node-multiple",
        type=int,
        default=1024,
        help="Round padded node count up to this multiple to reduce shape diversity (1 disables).",
    )
    parser.add_argument(
        "--pad-edge-multiple",
        type=int,
        default=32768,
        help="Round padded edge count up to this multiple to reduce shape diversity (1 disables).",
    )
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
    parser.add_argument(
        "--train-progress-every",
        type=int,
        default=1,
        help="Print training progress every N partition steps within each epoch.",
    )
    parser.add_argument(
        "--output-dir",
        default=f"{CANONICAL_OUTPUT_ROOT}/sbi_partitioned",
        help="Directory for models/logs.",
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
        # SLURM can encode this as "4(x2)"; keep the leading integer.
        tasks_per_node = int(tasks_per_node_raw.split("(")[0].split(",")[0])
        # If Slurm narrowed visibility to one device per task, always index that as local 0.
        if len(ids) == 1:
            return [0]
        # One task per node should own all visible local devices on that node.
        if tasks_per_node == 1:
            return list(range(len(ids)))
        # If multiple local devices are visible and multiple tasks share the node,
        # bind each process to its local rank device.
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


def _load_partition_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as d:
        x = np.asarray(d["x"], dtype=np.float32)
        edge_index = np.asarray(d["edge_index"], dtype=np.int32)
        edge_attr = np.asarray(d["edge_attr"], dtype=np.float32)
        targets = np.asarray(d["targets"])
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


def _compute_dtype_from_mode(mode: str) -> jnp.dtype:
    if mode == "bf16":
        return jnp.bfloat16
    return jnp.float32


def load_partition(path: Path, *, compute_dtype: jnp.dtype = jnp.float32) -> tuple[jraph.GraphsTuple, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
    """Small in-process LRU cache for partition array payloads."""

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

    def get(self, path: Path, loader: Callable[[Path], dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        key = str(path)
        if self._max_items > 0 and key in self._items:
            self._items.move_to_end(key)
            return self._items[key]
        return self.set(path, loader(path))


def _collate_padded_partition_batch(
    base_dir: Path,
    batch_parts: list[dict],
    *,
    compute_dtype: jnp.dtype,
    pad_nodes: int | None = None,
    pad_edges: int | None = None,
    array_loader: Callable[[Path], dict[str, np.ndarray]] = _load_partition_arrays,
) -> tuple[jraph.GraphsTuple, jnp.ndarray, jnp.ndarray]:
    loaded = [array_loader(base_dir / p["file"]) for p in batch_parts]
    n_dev = len(loaded)
    local_max_nodes = max(int(x["n_nodes"]) for x in loaded)
    local_max_edges = max(int(x["n_edges"]) for x in loaded)
    max_nodes = local_max_nodes if pad_nodes is None else max(local_max_nodes, int(pad_nodes))
    max_edges = local_max_edges if pad_edges is None else max(local_max_edges, int(pad_edges))
    node_feat_dim = int(loaded[0]["x"].shape[1])
    edge_feat_dim = int(loaded[0]["edge_attr"].shape[1])
    target_dim = int(loaded[0]["targets"].shape[1])

    nodes = np.zeros((n_dev, max_nodes, node_feat_dim), dtype=np.float32)
    targets = np.zeros((n_dev, max_nodes, target_dim), dtype=loaded[0]["targets"].dtype)
    core_mask = np.zeros((n_dev, max_nodes), dtype=bool)
    node_valid_mask = np.zeros((n_dev, max_nodes), dtype=bool)
    edge_attr = np.zeros((n_dev, max_edges, edge_feat_dim), dtype=np.float32)
    # Route padded edges to a dummy node so they do not affect real nodes.
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
    return graph, jnp.array(targets), jnp.array(core_mask & node_valid_mask, dtype=bool)


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
    # Ensure every process executes exactly the same number of pmap calls.
    local_n = np.array([len(local)], dtype=np.int32)
    all_n = np.asarray(multihost_utils.process_allgather(local_n)).reshape(-1)
    min_n = int(all_n.min())
    full_groups = (min_n // max(1, n_local_devices)) * max(1, n_local_devices)
    return local[:full_groups]


def _take_first_replica(tree):
    return jax.tree_util.tree_map(lambda x: x[0], tree)


def main(args: argparse.Namespace) -> None:
    rank, world, local_rank = _rank_info()
    _maybe_init_distributed(args, rank, world, local_rank)
    require_gpu_slurm("jraph_sbi_flowjax_partitioned.py", min_gpus=1)
    os.makedirs(args.output_dir, exist_ok=True)
    local_devices = jax.local_devices()
    n_local_devices = len(local_devices)
    if n_local_devices < 1:
        raise RuntimeError("No local devices available after distributed initialization.")

    print("=" * 70, flush=True)
    print("Partitioned SBI Trainer (FlowJAX + Haiku)", flush=True)
    print("=" * 70, flush=True)
    print(f"Output dir: {args.output_dir}", flush=True)
    print(f"Manifest: {args.partition_manifest}", flush=True)
    print(f"SBI cache: {args.sbi_cache_path}", flush=True)
    print(
        f"Config: epochs={args.epochs}, num_passes={args.num_passes}, latent={args.latent_size}, "
        f"flow_layers={args.num_flow_layers}, activation_checkpointing={args.activation_checkpointing}, "
        f"data_parallel={args.data_parallel}, distributed={args.distributed or world > 1}, "
        f"mixed_precision={args.mixed_precision}",
        flush=True,
    )
    compute_dtype = _compute_dtype_from_mode(args.mixed_precision)

    with open(args.partition_manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    base_dir = Path(args.partition_manifest).resolve().parent

    train_parts = [p for p in manifest["partitions"] if p["split"] == "train"]
    val_parts_full = [p for p in manifest["partitions"] if p["split"] == "val"]
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
    if rank == 0 and (dropped_train or dropped_val):
        print(
            f"Dropped oversized partitions: train={dropped_train}, val={dropped_val} "
            f"(max_nodes={args.max_partition_nodes or 'off'}, max_edges={args.max_partition_edges or 'off'})",
            flush=True,
        )
    if args.train_partition_limit > 0:
        train_parts = train_parts[: args.train_partition_limit]
    if args.full_val_partition_limit > 0:
        val_parts_full = val_parts_full[: args.full_val_partition_limit]
    val_parts_quick = val_parts_full
    if args.val_partition_limit > 0:
        val_parts_quick = val_parts_full[: args.val_partition_limit]
    train_parts = _truncate_for_distributed(train_parts, rank, world, n_local_devices)
    val_parts_full = _truncate_for_distributed(val_parts_full, rank, world, n_local_devices)
    val_parts_quick = _truncate_for_distributed(val_parts_quick, rank, world, n_local_devices)
    if args.train_partitions_per_epoch > 0:
        train_parts = train_parts[: min(len(train_parts), args.train_partitions_per_epoch)]

    if not train_parts:
        raise ValueError("No train partitions found after rank/world assignment.")
    print(
        f"Rank {rank}/{world} | local_devices={n_local_devices} | "
        f"Train partitions: {len(train_parts)}, "
        f"Val quick/full partitions: {len(val_parts_quick)}/{len(val_parts_full)}",
        flush=True,
    )
    print(f"Devices: {jax.devices()}", flush=True)

    # Read scaler/raw metadata from the monolithic cache once for output compatibility.
    with open(args.sbi_cache_path, "rb") as f:
        source_cache = pickle.load(f)
    target_scaler = source_cache.get("target_scaler")
    stats = source_cache.get("stats")

    rng = jax.random.key(args.seed)
    print("Loading first train partition for model init...", flush=True)
    first_graph, _, _, _ = load_partition(base_dir / train_parts[0]["file"], compute_dtype=compute_dtype)
    print(
        f"First partition stats: nodes={int(first_graph.n_node[0]):,}, "
        f"edges={int(first_graph.n_edge[0]):,}",
        flush=True,
    )
    gnn_fn = make_gnn_encoder(
        num_passes=args.num_passes,
        latent_size=args.latent_size,
        num_heads=args.num_heads,
        dropout_rate=args.dropout,
    )
    gnn = hk.transform(gnn_fn)
    rng, gnn_init_key = jax.random.split(rng)
    gnn_params = gnn.init(gnn_init_key, first_graph, is_training=True)

    rng, flow_key = jax.random.split(rng)
    flow = masked_autoregressive_flow(
        flow_key,
        base_dist=Normal(jnp.zeros(3), jnp.ones(3)),
        cond_dim=args.latent_size,
        flow_layers=args.num_flow_layers,
        nn_width=args.flow_hidden_size,
        nn_depth=2,
        transformer=RationalQuadraticSpline(knots=args.num_bins, interval=12),
    )
    flow_arrays, flow_static = eqx.partition(flow, eqx.is_inexact_array)

    num_epochs = args.epochs
    warmup_steps = min(200, max(1, num_epochs // 10))
    decay_steps = max(num_epochs - warmup_steps, warmup_steps + 1)
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=1e-5,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(lr_schedule, weight_decay=args.weight_decay),
    )
    gnn_opt_state = optimizer.init(gnn_params)
    flow_opt_state = optimizer.init(flow_arrays)

    def _gnn_forward_train(gnn_p, step_key, graph):
        return gnn.apply(gnn_p, step_key, graph, is_training=True)

    def _gnn_forward_eval(gnn_p, step_key, graph):
        return gnn.apply(gnn_p, step_key, graph, is_training=False)

    if args.activation_checkpointing:
        _gnn_forward_train = jax.checkpoint(_gnn_forward_train, prevent_cse=False)

    def _local_logp_sum_count(emb, flow_arr, targets, core_mask):
        flow_model = eqx.combine(flow_arr, flow_static)
        log_probs = jax.vmap(flow_model.log_prob)(targets, condition=emb)
        log_probs = log_probs.astype(jnp.float32)
        mask_f = core_mask.astype(jnp.float32)
        logp_sum = jnp.sum(log_probs * mask_f)
        n_core = jnp.sum(mask_f)
        return logp_sum, n_core

    def train_loss_fn(gnn_p, flow_arr, graph, targets, core_mask, step_key):
        emb = _gnn_forward_train(gnn_p, step_key, graph)
        logp_sum, n_core = _local_logp_sum_count(emb, flow_arr, targets, core_mask)
        nll = -logp_sum / jnp.maximum(n_core, 1.0)
        return nll, (logp_sum, n_core)

    def eval_loss_fn(gnn_p, flow_arr, graph, targets, core_mask, step_key):
        emb = _gnn_forward_eval(gnn_p, step_key, graph)
        logp_sum, n_core = _local_logp_sum_count(emb, flow_arr, targets, core_mask)
        nll = -logp_sum / jnp.maximum(n_core, 1.0)
        return nll, (logp_sum, n_core)

    @jax.jit
    def train_step_single(gnn_p, gnn_state, flow_arr, flow_state, graph, targets, core_mask, step_key):
        (nll, (logp_sum, n_core)), grads = jax.value_and_grad(
            lambda gp, fa: train_loss_fn(gp, fa, graph, targets, core_mask, step_key),
            argnums=(0, 1),
            has_aux=True,
        )(gnn_p, flow_arr)
        gnn_grads, flow_grads = grads
        gnn_updates, gnn_state_new = optimizer.update(gnn_grads, gnn_state, gnn_p)
        flow_updates, flow_state_new = optimizer.update(flow_grads, flow_state, flow_arr)
        gnn_p_new = optax.apply_updates(gnn_p, gnn_updates)
        flow_arr_new = optax.apply_updates(flow_arr, flow_updates)
        return gnn_p_new, gnn_state_new, flow_arr_new, flow_state_new, nll, logp_sum, n_core

    @jax.jit
    def eval_step_single(gnn_p, flow_arr, graph, targets, core_mask, step_key):
        return eval_loss_fn(gnn_p, flow_arr, graph, targets, core_mask, step_key)

    @partial(jax.pmap, axis_name="dp")
    def train_step_dp(gnn_p, gnn_state, flow_arr, flow_state, graph, targets, core_mask, step_keys):
        device_key = jax.random.fold_in(step_keys, jax.lax.axis_index("dp"))

        def _loss_for_grad(gp, fa):
            emb = _gnn_forward_train(gp, device_key, graph)
            local_logp_sum, local_n_core = _local_logp_sum_count(emb, fa, targets, core_mask)
            global_logp_sum = jax.lax.psum(local_logp_sum, axis_name="dp")
            global_n_core = jax.lax.psum(local_n_core, axis_name="dp")
            nll = -global_logp_sum / jnp.maximum(global_n_core, 1.0)
            return nll, (local_logp_sum, local_n_core)

        (nll, (local_logp_sum, local_n_core)), grads = jax.value_and_grad(
            _loss_for_grad,
            argnums=(0, 1),
            has_aux=True,
        )(gnn_p, flow_arr)
        gnn_grads, flow_grads = grads
        gnn_grads = jax.lax.pmean(gnn_grads, axis_name="dp")
        flow_grads = jax.lax.pmean(flow_grads, axis_name="dp")
        gnn_updates, gnn_state_new = optimizer.update(gnn_grads, gnn_state, gnn_p)
        flow_updates, flow_state_new = optimizer.update(flow_grads, flow_state, flow_arr)
        gnn_p_new = optax.apply_updates(gnn_p, gnn_updates)
        flow_arr_new = optax.apply_updates(flow_arr, flow_updates)
        global_logp_sum = jax.lax.psum(local_logp_sum, axis_name="dp")
        global_n_core = jax.lax.psum(local_n_core, axis_name="dp")
        global_nll = -global_logp_sum / jnp.maximum(global_n_core, 1.0)
        return gnn_p_new, gnn_state_new, flow_arr_new, flow_state_new, global_nll

    @partial(jax.pmap, axis_name="dp")
    def eval_step_dp(gnn_p, flow_arr, graph, targets, core_mask, step_keys):
        device_key = jax.random.fold_in(step_keys, jax.lax.axis_index("dp"))
        emb = _gnn_forward_eval(gnn_p, device_key, graph)
        local_logp_sum, local_n_core = _local_logp_sum_count(emb, flow_arr, targets, core_mask)
        global_logp_sum = jax.lax.psum(local_logp_sum, axis_name="dp")
        global_n_core = jax.lax.psum(local_n_core, axis_name="dp")
        global_nll = -global_logp_sum / jnp.maximum(global_n_core, 1.0)
        return global_nll

    best_val = float("inf")
    best = None
    history = {"train_nll": [], "val_nll": [], "val_kind": []}
    array_cache = _PartitionArrayCache(args.partition_cache_size)
    prefetch_pool = ThreadPoolExecutor(max_workers=max(1, args.prefetch_workers)) if args.prefetch_workers > 0 else None
    inflight_prefetch: dict[str, Future] = {}

    def _load_arrays_cached(path: Path) -> dict[str, np.ndarray]:
        key = str(path)
        fut = inflight_prefetch.pop(key, None)
        if fut is not None:
            return array_cache.set(path, fut.result())
        return array_cache.get(path, _load_partition_arrays)

    if args.data_parallel:
        rep_gnn_params = jax.device_put_replicated(gnn_params, local_devices)
        rep_gnn_opt_state = jax.device_put_replicated(gnn_opt_state, local_devices)
        rep_flow_arrays = jax.device_put_replicated(flow_arrays, local_devices)
        rep_flow_opt_state = jax.device_put_replicated(flow_opt_state, local_devices)

    t0 = time.time()
    try:
        for epoch in range(num_epochs):
            if rank == 0:
                print(f"Epoch {epoch:04d} start", flush=True)
            rng, ep_key = jax.random.split(rng)
            train_nll_epoch: list[float] = []

            if args.data_parallel:
                train_groups = _build_epoch_groups(
                    train_parts,
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
                        pids = ",".join(p["partition_id"] for p in batch_parts[:2])
                        print(
                            f"  [train] epoch={epoch:04d} step={step_idx + 1}/{n_train_steps} "
                            f"sample_partitions={pids}",
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
                    rep_gnn_params, rep_gnn_opt_state, rep_flow_arrays, rep_flow_opt_state, nll_rep = train_step_dp(
                        rep_gnn_params,
                        rep_gnn_opt_state,
                        rep_flow_arrays,
                        rep_flow_opt_state,
                        graph_b,
                        targets_b,
                        core_mask_b,
                        step_keys,
                    )
                    train_nll_epoch.append(float(jax.device_get(nll_rep[0])))
            else:
                order = np.random.default_rng(args.seed + epoch).permutation(len(train_parts))
                n_train_steps = len(order)
                for step_idx, i in enumerate(order):
                    p = train_parts[int(i)]
                    if rank == 0 and step_idx % max(1, args.train_progress_every) == 0:
                        print(
                            f"  [train] epoch={epoch:04d} step={step_idx + 1}/{n_train_steps} "
                            f"partition={p['partition_id']}",
                            flush=True,
                        )
                    graph, targets, core_mask, _ = load_partition(base_dir / p["file"], compute_dtype=compute_dtype)
                    ep_key, step_key = jax.random.split(ep_key)
                    gnn_params, gnn_opt_state, flow_arrays, flow_opt_state, nll, _, _ = train_step_single(
                        gnn_params, gnn_opt_state, flow_arrays, flow_opt_state, graph, targets, core_mask, step_key
                    )
                    train_nll_epoch.append(float(nll))

            mean_train_nll = float(np.mean(train_nll_epoch)) if train_nll_epoch else float("nan")
            history["train_nll"].append((epoch, mean_train_nll))

            if epoch % args.eval_every == 0:
                do_full_val = args.full_val_every > 0 and ((epoch + 1) % args.full_val_every == 0)
                epoch_val_parts = val_parts_full if do_full_val else val_parts_quick
                val_losses = []
                val_kind = "full" if do_full_val else "quick"
                if rank == 0:
                    print(
                        f"  [val] epoch={epoch:04d} mode={val_kind} evaluating {len(epoch_val_parts)} partitions",
                        flush=True,
                    )
                if args.data_parallel:
                    # Deterministic eval grouping (no shuffle) for stable comparisons.
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
                        val_nll_rep = eval_step_dp(
                            rep_gnn_params, rep_flow_arrays, graph_b, targets_b, core_mask_b, step_keys
                        )
                        val_losses.append(float(jax.device_get(val_nll_rep[0])))
                else:
                    for p in epoch_val_parts:
                        graph, targets, core_mask, _ = load_partition(base_dir / p["file"], compute_dtype=compute_dtype)
                        ep_key, step_key = jax.random.split(ep_key)
                        val_nll, _ = eval_step_single(gnn_params, flow_arrays, graph, targets, core_mask, step_key)
                        val_losses.append(float(val_nll))
                mean_val = float(np.mean(val_losses)) if val_losses else float("nan")
                history["val_nll"].append((epoch, mean_val))
                history["val_kind"].append((epoch, val_kind))
                if mean_val < best_val:
                    best_val = mean_val
                    if args.data_parallel:
                        best = (
                            jax.device_get(_take_first_replica(rep_gnn_params)),
                            jax.device_get(_take_first_replica(rep_flow_arrays)),
                        )
                    else:
                        best = (
                            jax.device_get(gnn_params),
                            jax.device_get(flow_arrays),
                        )
                if rank == 0:
                    print(f"Epoch {epoch:04d} | train_nll={mean_train_nll:.4f} | val_nll={mean_val:.4f}", flush=True)
            else:
                if rank == 0:
                    print(f"Epoch {epoch:04d} | train_nll={mean_train_nll:.4f}", flush=True)
    finally:
        if prefetch_pool is not None:
            prefetch_pool.shutdown(wait=False)

    elapsed = time.time() - t0
    if rank == 0:
        print(f"Training finished in {elapsed:.1f}s, best_val_nll={best_val:.4f}", flush=True)

    if best is None:
        if args.data_parallel:
            best = (
                jax.device_get(_take_first_replica(rep_gnn_params)),
                jax.device_get(_take_first_replica(rep_flow_arrays)),
            )
        else:
            best = (jax.device_get(gnn_params), jax.device_get(flow_arrays))
    if rank != 0:
        return

    best_gnn, best_flow_arrays = best
    best_flow = eqx.combine(best_flow_arrays, flow_static)
    ts = time.strftime("%Y%m%d_%H%M%S")
    flow_path = os.path.join(args.output_dir, f"partitioned_flow_seed_{args.seed}_{ts}.eqx")
    model_path = os.path.join(args.output_dir, f"partitioned_model_seed_{args.seed}_{ts}.pkl")
    logs_path = os.path.join(args.output_dir, f"partitioned_logs_seed_{args.seed}_{ts}.pkl")
    eqx.tree_serialise_leaves(flow_path, best_flow)
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "gnn_params": best_gnn,
                "flow_path": flow_path,
                "config": vars(args),
                "target_scaler": target_scaler,
                "stats": stats,
                "partition_manifest": args.partition_manifest,
                "rank": rank,
                "world_size": world,
            },
            f,
        )
    with open(logs_path, "wb") as f:
        pickle.dump(history, f)
    print(f"Saved flow: {flow_path}", flush=True)
    print(f"Saved model: {model_path}", flush=True)
    print(f"Saved logs: {logs_path}", flush=True)


if __name__ == "__main__":
    main(parse_args())
