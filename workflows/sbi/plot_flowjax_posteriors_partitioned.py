"""
Posterior visualization for partitioned FlowJAX-SBI models.

This script is the partition-aware counterpart to plot_flowjax_posteriors.py:
- loads `partitioned_model_seed_*.pkl` (+ referenced `.eqx` flow),
- evaluates test split partitions from a partition manifest,
- samples posteriors and produces the same diagnostics:
  training curve, individual posterior plots, comparison plot,
  calibration histograms, and TARP coverage.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
import sys

import equinox as eqx
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import matplotlib.pyplot as plt
import numpy as np
from flowjax.distributions import Normal
from flowjax.flows import RationalQuadraticSpline, masked_autoregressive_flow

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.config_paths import CANONICAL_FIGURE_ROOT
from shared.eigenvalue_transformations import samples_to_raw_eigenvalues
from shared.graph_net_models import make_gnn_encoder

try:
    import tarp

    TARP_AVAILABLE = True
except ImportError:
    TARP_AVAILABLE = False


def _load_partition_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as d:
        x = np.asarray(d["x"], dtype=np.float32)
        edge_index = np.asarray(d["edge_index"], dtype=np.int32)
        edge_attr = np.asarray(d["edge_attr"], dtype=np.float32)
        targets = np.asarray(d["targets"], dtype=np.float32)
        core_mask = np.asarray(d["core_mask_local"], dtype=bool)
    return {
        "x": x,
        "senders": edge_index[0],
        "receivers": edge_index[1],
        "edge_attr": edge_attr,
        "targets": targets,
        "core_mask": core_mask,
        "n_nodes": np.int32(x.shape[0]),
        "n_edges": np.int32(edge_index.shape[1]),
    }


def _compute_dtype_from_config(config: dict) -> jnp.dtype:
    mode = str(config.get("mixed_precision", "none"))
    return jnp.bfloat16 if mode == "bf16" else jnp.float32


def _make_gnn_and_flow(config: dict, flow_path: str, graph_for_init: jraph.GraphsTuple, key: jax.Array):
    gnn_fn = make_gnn_encoder(
        num_passes=int(config["num_passes"]),
        latent_size=int(config["latent_size"]),
        num_heads=int(config["num_heads"]),
        dropout_rate=float(config.get("dropout", 0.2)),
    )
    gnn = hk.transform(gnn_fn)
    gnn_key, flow_key = jax.random.split(key)
    _ = gnn.init(gnn_key, graph_for_init, is_training=False)
    flow = masked_autoregressive_flow(
        flow_key,
        base_dist=Normal(jnp.zeros(3), jnp.ones(3)),
        cond_dim=int(config["latent_size"]),
        flow_layers=int(config.get("num_flow_layers", 5)),
        nn_width=int(config.get("flow_hidden_size", 128)),
        nn_depth=2,
        transformer=RationalQuadraticSpline(knots=int(config.get("num_bins", 8)), interval=12),
    )
    flow = eqx.tree_deserialise_leaves(flow_path, flow)
    return gnn, flow


def _plot_training_history(logs_path: Path, output_dir: Path) -> None:
    if not logs_path.exists():
        print(f"Logs not found at {logs_path}; skipping training plot.")
        return
    with logs_path.open("rb") as f:
        logs = pickle.load(f)
    train_hist = logs.get("train_nll", [])
    val_hist = logs.get("val_nll", [])
    if not train_hist:
        print("No train_nll history found; skipping training plot.")
        return
    epochs_t = [int(e) for e, _ in train_hist]
    vals_t = [float(v) for _, v in train_hist]
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_t, vals_t, label="Train NLL", alpha=0.8)
    if val_hist:
        epochs_v = [int(e) for e, _ in val_hist]
        vals_v = [float(v) for _, v in val_hist]
        plt.plot(epochs_v, vals_v, label="Val NLL", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("NLL Loss")
    plt.title("Flowjax Training History")
    plt.grid(alpha=0.3)
    plt.legend()
    out = output_dir / "flowjax_training.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def _sample_posterior(flow, embedding: np.ndarray, num_samples: int, key: jax.Array) -> np.ndarray:
    samples = flow.sample(key, (num_samples,), condition=jnp.asarray(embedding))
    return np.asarray(samples)


def _collect_test_samples(
    *,
    model_info: dict,
    partition_manifest_path: Path,
    gnn,
    gnn_params,
    compute_dtype: jnp.dtype,
    max_test_nodes: int,
    seed: int,
    test_partition_limit: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    with partition_manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    base_dir = partition_manifest_path.parent
    test_parts = [p for p in manifest["partitions"] if p.get("split") == "test"]
    if test_partition_limit > 0:
        test_parts = test_parts[:test_partition_limit]
    if not test_parts:
        raise ValueError("No test partitions found in manifest.")

    rng = np.random.default_rng(seed)
    per_part_keep = max(1, int(np.ceil(max_test_nodes / max(1, len(test_parts)) * 2.0)))
    all_emb: list[np.ndarray] = []
    all_tgt: list[np.ndarray] = []
    total_core = 0
    key = jax.random.key(seed)

    for i, part in enumerate(test_parts):
        arr = _load_partition_arrays(base_dir / part["file"])
        core_idx = np.flatnonzero(arr["core_mask"])
        total_core += int(core_idx.size)
        if core_idx.size == 0:
            continue
        keep = core_idx
        if core_idx.size > per_part_keep:
            keep = rng.choice(core_idx, size=per_part_keep, replace=False)

        graph = jraph.GraphsTuple(
            nodes=jnp.asarray(arr["x"], dtype=compute_dtype),
            edges=jnp.asarray(arr["edge_attr"], dtype=compute_dtype),
            senders=jnp.asarray(arr["senders"], dtype=jnp.int32),
            receivers=jnp.asarray(arr["receivers"], dtype=jnp.int32),
            n_node=jnp.asarray([arr["n_nodes"]], dtype=jnp.int32),
            n_edge=jnp.asarray([arr["n_edges"]], dtype=jnp.int32),
            globals=None,
        )
        key, step_key = jax.random.split(key)
        emb = np.asarray(gnn.apply(gnn_params, step_key, graph, is_training=False))
        all_emb.append(emb[keep])
        all_tgt.append(arr["targets"][keep])
        if (i + 1) % 10 == 0:
            print(f"Processed test partitions: {i + 1}/{len(test_parts)}")

    emb_all = np.concatenate(all_emb, axis=0)
    tgt_all = np.concatenate(all_tgt, axis=0)
    if emb_all.shape[0] > max_test_nodes:
        take = rng.choice(emb_all.shape[0], size=max_test_nodes, replace=False)
        emb_all = emb_all[take]
        tgt_all = tgt_all[take]
    return emb_all, tgt_all, total_core


def main(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path).expanduser().resolve()
    with model_path.open("rb") as f:
        model_info = pickle.load(f)

    config = model_info["config"]
    gnn_params = model_info["gnn_params"]
    flow_path = model_info.get("flow_path") or model_info.get("flow_filename")
    if not flow_path:
        raise KeyError("Model artifact missing flow_path/flow_filename.")
    flow_path = str(flow_path)
    target_scaler = model_info.get("target_scaler", model_info.get("eigenvalue_scaler"))
    use_transformed_eig = bool(model_info.get("use_transformed_eig", True))

    manifest_path = (
        Path(args.partition_manifest).expanduser().resolve()
        if args.partition_manifest
        else Path(model_info["partition_manifest"]).expanduser().resolve()
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FlowJAX Partitioned Posterior Visualization")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Flow : {flow_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Output: {output_dir}")

    first_part = None
    with manifest_path.open("r", encoding="utf-8") as f:
        parts = json.load(f)["partitions"]
    for p in parts:
        if p.get("split") == "test":
            first_part = p
            break
    if first_part is None:
        raise ValueError("No test partition in manifest.")
    arr0 = _load_partition_arrays(manifest_path.parent / first_part["file"])
    compute_dtype = _compute_dtype_from_config(config)
    graph0 = jraph.GraphsTuple(
        nodes=jnp.asarray(arr0["x"], dtype=compute_dtype),
        edges=jnp.asarray(arr0["edge_attr"], dtype=compute_dtype),
        senders=jnp.asarray(arr0["senders"], dtype=jnp.int32),
        receivers=jnp.asarray(arr0["receivers"], dtype=jnp.int32),
        n_node=jnp.asarray([arr0["n_nodes"]], dtype=jnp.int32),
        n_edge=jnp.asarray([arr0["n_edges"]], dtype=jnp.int32),
        globals=None,
    )

    gnn, flow = _make_gnn_and_flow(config, flow_path, graph0, jax.random.key(args.seed))
    test_embeddings, test_targets_scaled, total_test_core = _collect_test_samples(
        model_info=model_info,
        partition_manifest_path=manifest_path,
        gnn=gnn,
        gnn_params=gnn_params,
        compute_dtype=compute_dtype,
        max_test_nodes=args.max_test_nodes,
        seed=args.seed,
        test_partition_limit=args.test_partition_limit,
    )
    print(f"Collected sampled test nodes: {len(test_embeddings)} (total core nodes seen: {total_test_core})")

    if target_scaler is None:
        raise ValueError("Model artifact is missing target_scaler/eigenvalue_scaler.")
    true_trans = target_scaler.inverse_transform(test_targets_scaled)
    true_raw = samples_to_raw_eigenvalues(test_targets_scaled, target_scaler, use_transformed_eig)
    param_names_raw = [r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$"]
    param_names_trans = [r"$v_1$", r"$\Delta\lambda_2$", r"$\Delta\lambda_3$"] if use_transformed_eig else param_names_raw

    logs_path = Path(str(model_path).replace("partitioned_model_", "partitioned_logs_"))
    _plot_training_history(logs_path, output_dir)

    rng = np.random.default_rng(args.seed)
    key = jax.random.key(args.seed + 123)
    n = len(test_embeddings)
    n_individual = min(args.num_plots, n)
    idx_individual = rng.choice(n, size=n_individual, replace=False)

    for i, idx in enumerate(idx_individual):
        key, sk = jax.random.split(key)
        samples_scaled = _sample_posterior(flow, test_embeddings[idx], args.num_samples, sk)
        samples_trans = target_scaler.inverse_transform(samples_scaled)
        samples_raw = samples_to_raw_eigenvalues(samples_scaled, target_scaler, use_transformed_eig)
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        for j in range(3):
            axes[0, j].hist(samples_trans[:, j], bins=50, density=True, alpha=0.7, color="steelblue")
            axes[0, j].axvline(true_trans[idx, j], color="red", linewidth=2, label="True" if j == 0 else None)
            axes[0, j].axvline(
                float(np.mean(samples_trans[:, j])),
                color="green",
                linewidth=2,
                linestyle=":",
                label="Mean" if j == 0 else None,
            )
            axes[0, j].set_xlabel(param_names_trans[j])
            axes[1, j].hist(samples_raw[:, j], bins=50, density=True, alpha=0.7, color="darkorange")
            axes[1, j].axvline(true_raw[idx, j], color="red", linewidth=2)
            axes[1, j].axvline(float(np.mean(samples_raw[:, j])), color="green", linewidth=2, linestyle=":")
            axes[1, j].set_xlabel(param_names_raw[j])
        axes[0, 0].set_ylabel("Transformed Space")
        axes[1, 0].set_ylabel("Raw Eigenvalues")
        axes[0, 0].legend(fontsize=8)
        plt.suptitle(f"Posterior for sampled test node {int(idx)}")
        plt.tight_layout()
        out = output_dir / f"flowjax_dual_posterior_node_{int(idx)}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out} ({i+1}/{n_individual})")

    idx_cmp = idx_individual[: min(5, len(idx_individual))]
    fig, axes = plt.subplots(len(idx_cmp), 3, figsize=(12, 3 * len(idx_cmp)))
    if len(idx_cmp) == 1:
        axes = axes.reshape(1, -1)
    for r, idx in enumerate(idx_cmp):
        key, sk = jax.random.split(key)
        s = _sample_posterior(flow, test_embeddings[idx], args.num_samples, sk)
        for c in range(3):
            axes[r, c].hist(s[:, c], bins=40, density=True, alpha=0.7, color="steelblue")
            axes[r, c].axvline(
                test_targets_scaled[idx, c],
                color="red",
                linewidth=2,
                linestyle="--",
                label=f"True: {test_targets_scaled[idx, c]:.2f}",
            )
            mean_val = float(np.mean(s[:, c]))
            axes[r, c].axvline(mean_val, color="green", linewidth=2, linestyle=":", label=f"Mean: {mean_val:.2f}")
            axes[r, c].set_xlabel(param_names_raw[c])
            if c == 0:
                axes[r, c].set_ylabel(f"Node {int(idx)}")
            axes[r, c].legend(fontsize=8)
    plt.suptitle("FlowJAX Posterior Marginals Comparison")
    plt.tight_layout()
    out = output_dir / "flowjax_posterior_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    n_cal = min(args.num_calibration, n)
    idx_cal = rng.choice(n, size=n_cal, replace=False)
    ranks_raw = []
    ranks_trans = []
    for i, idx in enumerate(idx_cal):
        if (i + 1) % 200 == 0:
            print(f"Calibration sampling {i + 1}/{n_cal}...")
        key, sk = jax.random.split(key)
        s_scaled = _sample_posterior(flow, test_embeddings[idx], args.num_samples, sk)
        s_trans = target_scaler.inverse_transform(s_scaled)
        s_raw = samples_to_raw_eigenvalues(s_scaled, target_scaler, use_transformed_eig)
        ranks_trans.append(np.mean(s_trans < true_trans[idx], axis=0))
        ranks_raw.append(np.mean(s_raw < true_raw[idx], axis=0))
    ranks_trans = np.asarray(ranks_trans)
    ranks_raw = np.asarray(ranks_raw)

    for name, ranks, color in [
        ("flowjax_calibration_transformed.png", ranks_trans, "steelblue"),
        ("flowjax_calibration_raw_eig.png", ranks_raw, "darkorange"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for j in range(3):
            axes[j].hist(ranks[:, j], bins=20, density=True, alpha=0.7, color=color, edgecolor="white")
            axes[j].axhline(1.0, color="red", linestyle="--", linewidth=2, label="Uniform")
            label_set = param_names_trans if "transformed" in name else param_names_raw
            axes[j].set_xlabel(f"Rank for {label_set[j]}")
            axes[j].set_ylabel("Density" if j == 0 else "")
            axes[j].set_xlim(0, 1)
            axes[j].set_title(label_set[j])
            axes[j].legend()
        if "transformed" in name:
            plt.suptitle(f"Calibration (Transformed Space) - {n_cal} test points", fontsize=14)
        else:
            plt.suptitle(f"Calibration (Raw Eigenvalues) - {n_cal} test points", fontsize=14)
        plt.tight_layout()
        out = output_dir / name
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")

    if TARP_AVAILABLE:
        n_tarp = min(args.num_tarp, n)
        idx_tarp = rng.choice(n, size=n_tarp, replace=False)
        all_samples = []
        all_truth = []
        for i, idx in enumerate(idx_tarp):
            if (i + 1) % 100 == 0:
                print(f"TARP sampling {i + 1}/{n_tarp}...")
            key, sk = jax.random.split(key)
            all_samples.append(_sample_posterior(flow, test_embeddings[idx], args.num_samples, sk))
            all_truth.append(test_targets_scaled[idx])
        all_samples = np.asarray(all_samples)  # [N, S, 3]
        all_truth = np.asarray(all_truth)      # [N, 3]
        samples_tarp = np.transpose(all_samples, (1, 0, 2))
        ecp, alpha = tarp.get_tarp_coverage(samples_tarp, all_truth, norm=True, bootstrap=False)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Ideal")
        ax.plot(alpha, ecp, "b-", linewidth=2, label="TARP")
        ax.set_xlabel(r"Credibility Level $\alpha$", fontsize=12)
        ax.set_ylabel("Expected Coverage Probability")
        ax.set_title(f"Flowjax TARP Coverage Test ({n_tarp} test points)", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
        ax.text(
            0.05,
            0.95,
            "Above diagonal = under-confident\nBelow diagonal = over-confident",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        out = output_dir / "flowjax_tarp_coverage.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")
    else:
        print("TARP not available; skipping tarp coverage plot.")

    print("=" * 70)
    print(f"Done. Outputs written to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Partition-aware FlowJAX posterior diagnostics")
    parser.add_argument("--model-path", required=True, help="Path to partitioned_model_seed_*.pkl")
    parser.add_argument("--partition-manifest", default="", help="Optional override for partition_manifest.json")
    parser.add_argument(
        "--output-dir",
        default=f"{CANONICAL_FIGURE_ROOT}/sbi/flowjax_partitioned",
        help="Directory for output plots.",
    )
    parser.add_argument("--num-plots", type=int, default=5, help="Number of individual posterior plots.")
    parser.add_argument("--num-samples", type=int, default=2000, help="Posterior samples per evaluated node.")
    parser.add_argument("--max-test-nodes", type=int, default=5000, help="Max sampled test core nodes for diagnostics.")
    parser.add_argument("--num-calibration", type=int, default=1000, help="Number of nodes for calibration histograms.")
    parser.add_argument("--num-tarp", type=int, default=500, help="Number of nodes for TARP coverage.")
    parser.add_argument("--test-partition-limit", type=int, default=0, help="Optional cap on test partitions.")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())

