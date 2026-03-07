"""Build partition mini-batches from an SBI-ready Abacus cache.

This script consumes the monolithic cache schema expected by the existing SBI
pipeline and emits partition artifacts for partition-aware training.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import pickle
from pathlib import Path

import numpy as np


SPLIT_CODE = {"train": 0, "val": 1, "test": 2}
SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}
GLOBAL_STATE: dict[str, object] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-cache-path", required=True, help="Path to SBI-ready monolithic cache (.pkl).")
    parser.add_argument("--output-dir", required=True, help="Directory for manifest and partition files.")
    parser.add_argument("--core-partition-size", type=int, default=250_000, help="Core nodes per partition.")
    parser.add_argument(
        "--adaptive-core-size",
        action="store_true",
        help="Adaptively choose core nodes per partition to satisfy node/edge budgets.",
    )
    parser.add_argument(
        "--min-core-nodes",
        type=int,
        default=5_000,
        help="Minimum core nodes per adaptive partition candidate.",
    )
    parser.add_argument(
        "--max-core-nodes",
        type=int,
        default=0,
        help="Maximum core nodes per adaptive partition candidate (0 -> --core-partition-size).",
    )
    parser.add_argument(
        "--target-total-nodes",
        type=int,
        default=0,
        help="Adaptive budget for total nodes (core + halo). 0 disables this budget.",
    )
    parser.add_argument(
        "--target-edges",
        type=int,
        default=0,
        help="Adaptive budget for induced edges. 0 disables this budget.",
    )
    parser.add_argument(
        "--halo-hops",
        type=int,
        default=0,
        help="Number of graph hops to add around core nodes (0 disables halo).",
    )
    parser.add_argument(
        "--edge-selection-chunk-size",
        type=int,
        default=10_000_000,
        help="Chunk size for scanning edges during induced-edge extraction.",
    )
    parser.add_argument(
        "--max-partitions-per-split",
        type=int,
        default=0,
        help="Optional cap for smoke testing. 0 means no cap.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "Parallel workers for partition construction. 0 = auto from "
            "SLURM_CPUS_PER_TASK/os.cpu_count()."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used only when randomizing core node order inside each split.",
    )
    parser.add_argument(
        "--shuffle-core-order",
        action="store_true",
        help="Shuffle core node order within each split before chunking.",
    )
    parser.add_argument(
        "--allow-login-node",
        action="store_true",
        help="Allow running outside Slurm. Intended for tiny smoke tests only.",
    )
    return parser.parse_args()


def _ensure_compute_node(args: argparse.Namespace) -> None:
    if "SLURM_JOB_ID" not in os.environ and not args.allow_login_node:
        raise RuntimeError(
            "build_abacus_partition_batches.py should run in a Slurm allocation. "
            "Use --allow-login-node only for tiny smoke tests."
        )


def _load_cache(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def _to_numpy_graph(cache: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    graph = cache["graph"]
    x = np.asarray(graph.nodes)
    edge_attr = np.asarray(graph.edges)
    senders = np.asarray(graph.senders, dtype=np.int64)
    receivers = np.asarray(graph.receivers, dtype=np.int64)
    targets = np.asarray(cache["regression_targets"])
    return x, edge_attr, senders, receivers, targets


def _neighbors_one_hop(frontier: np.ndarray, sorted_s: np.ndarray, sorted_r: np.ndarray, indptr: np.ndarray) -> np.ndarray:
    out = []
    for u in frontier:
        start = indptr[u]
        end = indptr[u + 1]
        if end > start:
            out.append(sorted_r[start:end])
    if not out:
        return np.empty((0,), dtype=np.int64)
    return np.unique(np.concatenate(out).astype(np.int64))


def _build_halo_nodes(core_nodes: np.ndarray, halo_hops: int, senders: np.ndarray, receivers: np.ndarray, n_nodes: int) -> np.ndarray:
    if halo_hops <= 0:
        return np.empty((0,), dtype=np.int64)

    order = np.argsort(senders, kind="mergesort")
    sorted_s = senders[order]
    sorted_r = receivers[order]
    deg = np.bincount(sorted_s, minlength=n_nodes)
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(deg, out=indptr[1:])

    visited = np.zeros(n_nodes, dtype=bool)
    visited[core_nodes] = True
    frontier = core_nodes
    for _ in range(halo_hops):
        neigh = _neighbors_one_hop(frontier, sorted_s, sorted_r, indptr)
        if neigh.size == 0:
            break
        new_nodes = neigh[~visited[neigh]]
        if new_nodes.size == 0:
            break
        visited[new_nodes] = True
        frontier = new_nodes

    halo_nodes = np.where(visited)[0]
    halo_nodes = halo_nodes[~np.isin(halo_nodes, core_nodes)]
    return halo_nodes.astype(np.int64)


def _induced_edge_indices(
    batch_node_mask: np.ndarray,
    senders: np.ndarray,
    receivers: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    hits = []
    n_edges = senders.shape[0]
    for start in range(0, n_edges, chunk_size):
        stop = min(start + chunk_size, n_edges)
        s = senders[start:stop]
        r = receivers[start:stop]
        m = batch_node_mask[s] & batch_node_mask[r]
        if np.any(m):
            local_idx = np.nonzero(m)[0].astype(np.int64)
            hits.append(local_idx + start)
    if not hits:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(hits)


def _save_partition_npz(
    path: Path,
    *,
    global_node_ids: np.ndarray,
    core_mask_local: np.ndarray,
    x_local: np.ndarray,
    y_local: np.ndarray,
    edge_index_local: np.ndarray,
    edge_attr_local: np.ndarray,
    split: str,
) -> None:
    np.savez_compressed(
        path,
        global_node_ids=global_node_ids.astype(np.int64),
        core_mask_local=core_mask_local.astype(bool),
        x=x_local.astype(np.float32),
        targets=y_local.astype(y_local.dtype),
        edge_index=edge_index_local.astype(np.int32),
        edge_attr=edge_attr_local.astype(np.float32),
        split_code=np.int8(SPLIT_CODE[split]),
    )


def _configure_global_state(
    *,
    x: np.ndarray,
    edge_attr: np.ndarray,
    senders: np.ndarray,
    receivers: np.ndarray,
    targets: np.ndarray,
    split_to_ids: dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    GLOBAL_STATE["x"] = x
    GLOBAL_STATE["edge_attr"] = edge_attr
    GLOBAL_STATE["senders"] = senders
    GLOBAL_STATE["receivers"] = receivers
    GLOBAL_STATE["targets"] = targets
    GLOBAL_STATE["split_to_ids"] = split_to_ids
    GLOBAL_STATE["out_dir"] = out_dir


def _build_partition_candidate(
    *,
    core_ids: np.ndarray,
    halo_hops: int,
    senders: np.ndarray,
    receivers: np.ndarray,
    n_nodes: int,
    edge_selection_chunk_size: int,
) -> dict:
    halo_ids = _build_halo_nodes(core_ids, halo_hops, senders, receivers, n_nodes)
    batch_ids = np.unique(np.concatenate([core_ids, halo_ids])).astype(np.int64)
    batch_node_mask = np.zeros(n_nodes, dtype=bool)
    batch_node_mask[batch_ids] = True
    edge_ids = _induced_edge_indices(batch_node_mask, senders, receivers, edge_selection_chunk_size)
    return {
        "core_ids": core_ids,
        "halo_ids": halo_ids,
        "batch_ids": batch_ids,
        "edge_ids": edge_ids,
        "n_core_nodes": int(core_ids.size),
        "n_halo_nodes": int(halo_ids.size),
        "n_total_nodes": int(batch_ids.size),
        "n_edges": int(edge_ids.size),
    }


def _fits_adaptive_budget(candidate: dict, args: argparse.Namespace) -> bool:
    nodes_ok = args.target_total_nodes <= 0 or candidate["n_total_nodes"] <= args.target_total_nodes
    edges_ok = args.target_edges <= 0 or candidate["n_edges"] <= args.target_edges
    return nodes_ok and edges_ok


def _choose_adaptive_candidate(
    *,
    core_ids_full: np.ndarray,
    start: int,
    split: str,
    args: argparse.Namespace,
    senders: np.ndarray,
    receivers: np.ndarray,
    n_nodes: int,
) -> tuple[dict, bool]:
    remain = core_ids_full.size - start
    max_core_nodes = args.max_core_nodes if args.max_core_nodes > 0 else args.core_partition_size
    hi = max(1, min(max_core_nodes, remain))
    lo = max(1, min(args.min_core_nodes, hi))

    cache: dict[int, dict] = {}

    def evaluate(size: int) -> dict:
        if size not in cache:
            stop = start + size
            core_ids = np.sort(core_ids_full[start:stop])
            cache[size] = _build_partition_candidate(
                core_ids=core_ids,
                halo_hops=args.halo_hops,
                senders=senders,
                receivers=receivers,
                n_nodes=n_nodes,
                edge_selection_chunk_size=args.edge_selection_chunk_size,
            )
        return cache[size]

    best = None
    left = lo
    right = hi
    while left <= right:
        mid = (left + right) // 2
        cand = evaluate(mid)
        if _fits_adaptive_budget(cand, args):
            best = cand
            left = mid + 1
        else:
            right = mid - 1

    if best is not None:
        return best, False

    # If even the minimum candidate violates the budget, still emit a partition to make progress.
    forced_size = lo
    forced = evaluate(forced_size)
    print(
        f"[{split}] WARNING adaptive budgets unmet even at min core size={forced_size:,}; "
        f"emitting oversized partition (nodes={forced['n_total_nodes']:,}, edges={forced['n_edges']:,})."
    )
    return forced, True


def _materialize_partition(
    *,
    split: str,
    part_tag: str,
    candidate: dict,
    oversized: bool,
) -> tuple[dict, str, tuple[int, int, int]]:
    x = GLOBAL_STATE["x"]
    edge_attr = GLOBAL_STATE["edge_attr"]
    senders = GLOBAL_STATE["senders"]
    receivers = GLOBAL_STATE["receivers"]
    targets = GLOBAL_STATE["targets"]
    out_dir = GLOBAL_STATE["out_dir"]

    core_ids = candidate["core_ids"]
    halo_ids = candidate["halo_ids"]
    batch_ids = candidate["batch_ids"]
    edge_ids = candidate["edge_ids"]
    core_mask_local = np.isin(batch_ids, core_ids)

    if edge_ids.size > 0:
        s_global = senders[edge_ids]
        r_global = receivers[edge_ids]
        s_local = np.searchsorted(batch_ids, s_global).astype(np.int32)
        r_local = np.searchsorted(batch_ids, r_global).astype(np.int32)
        edge_index_local = np.stack([s_local, r_local], axis=0)
        edge_attr_local = edge_attr[edge_ids]
    else:
        edge_index_local = np.zeros((2, 0), dtype=np.int32)
        edge_attr_local = np.zeros((0, edge_attr.shape[1]), dtype=edge_attr.dtype)

    x_local = x[batch_ids]
    y_local = targets[batch_ids]

    part_id = f"{split}_{part_tag}"
    part_file = f"partition_{part_id}.npz"
    _save_partition_npz(
        out_dir / part_file,
        global_node_ids=batch_ids,
        core_mask_local=core_mask_local,
        x_local=x_local,
        y_local=y_local,
        edge_index_local=edge_index_local,
        edge_attr_local=edge_attr_local,
        split=split,
    )

    entry = {
        "partition_id": part_id,
        "split": split,
        "file": part_file,
        "n_core_nodes": int(core_ids.size),
        "n_halo_nodes": int(halo_ids.size),
        "n_total_nodes": int(batch_ids.size),
        "n_edges": int(edge_index_local.shape[1]),
        "oversized_budget": bool(oversized),
    }
    log_line = (
        f"[{split}] {part_id}: core={core_ids.size:,}, halo={halo_ids.size:,}, "
        f"nodes={batch_ids.size:,}, edges={edge_index_local.shape[1]:,}"
    )
    return entry, log_line, (0, 0, 0)


def _build_fixed_task(
    split: str,
    part_count: int,
    start: int,
    stop: int,
    halo_hops: int,
    edge_selection_chunk_size: int,
) -> tuple[dict, str, tuple[int, int, int]]:
    split_to_ids = GLOBAL_STATE["split_to_ids"]
    senders = GLOBAL_STATE["senders"]
    receivers = GLOBAL_STATE["receivers"]
    n_nodes = int(GLOBAL_STATE["x"].shape[0])

    core_ids_full = split_to_ids[split]
    core_ids = np.sort(core_ids_full[start:stop])
    candidate = _build_partition_candidate(
        core_ids=core_ids,
        halo_hops=halo_hops,
        senders=senders,
        receivers=receivers,
        n_nodes=n_nodes,
        edge_selection_chunk_size=edge_selection_chunk_size,
    )
    entry, log_line, _ = _materialize_partition(
        split=split,
        part_tag=f"{part_count:06d}",
        candidate=candidate,
        oversized=False,
    )
    return entry, log_line, (0, int(part_count), 0)


def _build_adaptive_range_task(
    split: str,
    args: argparse.Namespace,
    start_idx: int,
    stop_idx: int,
    shard_id: int,
) -> list[tuple[dict, str, tuple[int, int, int]]]:
    split_to_ids = GLOBAL_STATE["split_to_ids"]
    senders = GLOBAL_STATE["senders"]
    receivers = GLOBAL_STATE["receivers"]
    n_nodes = int(GLOBAL_STATE["x"].shape[0])
    core_ids_full = split_to_ids[split][start_idx:stop_idx]

    out: list[tuple[dict, str, tuple[int, int, int]]] = []
    start = 0
    part_count = 0
    while start < core_ids_full.size:
        candidate, oversized = _choose_adaptive_candidate(
            core_ids_full=core_ids_full,
            start=start,
            split=split,
            args=args,
            senders=senders,
            receivers=receivers,
            n_nodes=n_nodes,
        )
        entry, log_line, _ = _materialize_partition(
            split=split,
            part_tag=f"s{shard_id:03d}_{part_count:06d}",
            candidate=candidate,
            oversized=oversized,
        )
        out.append((entry, log_line, (int(start_idx), int(part_count), int(candidate["n_core_nodes"]))))
        start += int(candidate["n_core_nodes"])
        part_count += 1
    return out


def _make_adaptive_shards(
    split_to_ids: dict[str, np.ndarray],
    splits: list[str],
    workers: int,
) -> list[tuple[str, int, int, int]]:
    total = sum(int(split_to_ids[s].size) for s in splits)
    if total == 0:
        return []
    tasks: list[tuple[str, int, int, int]] = []
    shard_uid = 0
    for split in splits:
        n = int(split_to_ids[split].size)
        if n == 0:
            continue
        split_workers = max(1, int(round(workers * n / total)))
        # Avoid excessive tiny shards; keep at least ~min-core sized chunks.
        max_by_size = max(1, n // 5000)
        split_workers = max(1, min(split_workers, max_by_size))
        bounds = np.linspace(0, n, num=split_workers + 1, dtype=np.int64)
        bounds = np.unique(bounds)
        for i in range(bounds.size - 1):
            start = int(bounds[i])
            stop = int(bounds[i + 1])
            if stop <= start:
                continue
            tasks.append((split, start, stop, shard_uid))
            shard_uid += 1
    return tasks


def _recommended_workers(requested: int) -> int:
    if requested > 0:
        return requested
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        try:
            return max(1, int(slurm_cpus))
        except ValueError:
            pass
    return max(1, os.cpu_count() or 1)


def main() -> None:
    args = parse_args()
    _ensure_compute_node(args)

    in_cache = Path(args.input_cache_path).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = _load_cache(in_cache)
    x, edge_attr, senders, receivers, targets = _to_numpy_graph(cache)
    train_mask, val_mask, test_mask = [np.asarray(m).astype(bool) for m in cache["masks"]]

    n_nodes = x.shape[0]
    n_edges = senders.shape[0]
    print(f"Loaded cache: nodes={n_nodes:,}, edges={n_edges:,}")

    split_to_ids = {
        "train": np.where(train_mask)[0].astype(np.int64),
        "val": np.where(val_mask)[0].astype(np.int64),
        "test": np.where(test_mask)[0].astype(np.int64),
    }
    if args.shuffle_core_order:
        rng = np.random.default_rng(args.seed)
        for k in split_to_ids:
            rng.shuffle(split_to_ids[k])

    _configure_global_state(
        x=x,
        edge_attr=edge_attr,
        senders=senders,
        receivers=receivers,
        targets=targets,
        split_to_ids=split_to_ids,
        out_dir=out_dir,
    )

    manifest = {
        "schema_version": 1,
        "source_cache_path": str(in_cache),
        "num_passes": None,
        "halo_hops": int(args.halo_hops),
        "core_partition_size": int(args.core_partition_size),
        "edge_selection_chunk_size": int(args.edge_selection_chunk_size),
        "n_nodes_global": int(n_nodes),
        "n_edges_global": int(n_edges),
        "target_dtype": str(targets.dtype),
        "feature_dtype": str(x.dtype),
        "adaptive_core_size": bool(args.adaptive_core_size),
        "min_core_nodes": int(args.min_core_nodes),
        "max_core_nodes": int(args.max_core_nodes if args.max_core_nodes > 0 else args.core_partition_size),
        "target_total_nodes": int(args.target_total_nodes),
        "target_edges": int(args.target_edges),
        "partitions": [],
    }

    if args.adaptive_core_size and args.target_total_nodes <= 0 and args.target_edges <= 0:
        raise ValueError(
            "--adaptive-core-size requires at least one budget: "
            "--target-total-nodes > 0 and/or --target-edges > 0."
        )

    workers = _recommended_workers(args.num_workers)
    print(f"Partition build workers: {workers}")

    mp_ctx = None
    if workers > 1:
        try:
            mp_ctx = mp.get_context("fork")
        except ValueError:
            print("WARNING: 'fork' multiprocessing context unavailable; falling back to sequential mode.")
            workers = 1

    built: list[tuple[dict, str, tuple[int, int, int]]] = []
    splits = [k for k in ("train", "val", "test") if split_to_ids[k].size > 0]
    max_parts = args.max_partitions_per_split if args.max_partitions_per_split > 0 else None

    if args.adaptive_core_size:
        if workers > 1 and max_parts is not None:
            print(
                "WARNING: --max-partitions-per-split with adaptive mode uses sequential execution "
                "to preserve exact per-split caps."
            )
            workers = 1

        if workers > 1:
            adaptive_tasks = _make_adaptive_shards(split_to_ids, splits, workers)
            print(f"Adaptive shard tasks: {len(adaptive_tasks)}")
            n_jobs = min(workers, len(adaptive_tasks))
            with cf.ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp_ctx) as ex:
                futures = [
                    ex.submit(_build_adaptive_range_task, split, args, start_idx, stop_idx, shard_id)
                    for (split, start_idx, stop_idx, shard_id) in adaptive_tasks
                ]
                for fut in cf.as_completed(futures):
                    for entry, log_line, sort_key in fut.result():
                        built.append((entry, log_line, sort_key))
                        print(log_line)
        else:
            for split in splits:
                core_ids_full = split_to_ids[split]
                stop_idx = int(core_ids_full.size)
                out = _build_adaptive_range_task(
                    split=split,
                    args=args,
                    start_idx=0,
                    stop_idx=stop_idx,
                    shard_id=0,
                )
                if max_parts is not None:
                    out = out[:max_parts]
                for entry, log_line, sort_key in out:
                    built.append((entry, log_line, sort_key))
                    print(log_line)
    else:
        tasks: list[tuple[str, int, int, int, int, int]] = []
        for split in splits:
            core_ids_full = split_to_ids[split]
            n = core_ids_full.size
            part_count = 0
            for start in range(0, n, args.core_partition_size):
                if max_parts is not None and part_count >= max_parts:
                    break
                stop = min(start + args.core_partition_size, n)
                tasks.append(
                    (
                        split,
                        part_count,
                        start,
                        stop,
                        args.halo_hops,
                        args.edge_selection_chunk_size,
                    )
                )
                part_count += 1
        if workers > 1 and tasks:
            with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as ex:
                futures = [ex.submit(_build_fixed_task, *t) for t in tasks]
                for fut in cf.as_completed(futures):
                    entry, log_line, sort_key = fut.result()
                    built.append((entry, log_line, sort_key))
                    print(log_line)
        else:
            for t in tasks:
                entry, log_line, sort_key = _build_fixed_task(*t)
                built.append((entry, log_line, sort_key))
                print(log_line)

    manifest["partitions"] = [
        x[0]
        for x in sorted(
            built,
            key=lambda p: (
                SPLIT_ORDER[p[0]["split"]],
                p[2][0],
                p[2][1],
                p[2][2],
            ),
        )
    ]

    manifest_path = out_dir / "partition_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest: {manifest_path}")
    print(f"Saved partitions: {len(manifest['partitions'])}")


if __name__ == "__main__":
    main()
