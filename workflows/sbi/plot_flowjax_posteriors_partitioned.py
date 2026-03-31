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
import subprocess
import tempfile
import time
from pathlib import Path
import sys

import equinox as eqx
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
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


def _apply_dark_plot_theme() -> None:
    """Apply a consistent dark plotting theme across all figures."""
    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "figure.facecolor": "#0f1117",
            "axes.facecolor": "#0f1117",
            "savefig.facecolor": "#0f1117",
            "savefig.edgecolor": "#0f1117",
            "axes.edgecolor": "#7d8590",
            "axes.labelcolor": "#e6edf3",
            "axes.titlecolor": "#e6edf3",
            "text.color": "#e6edf3",
            "xtick.color": "#c9d1d9",
            "ytick.color": "#c9d1d9",
            "grid.color": "#30363d",
            "grid.alpha": 0.35,
            "legend.facecolor": "#161b22",
            "legend.edgecolor": "#30363d",
            "legend.framealpha": 0.9,
        }
    )


def _load_partition_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as d:
        x = np.asarray(d["x"], dtype=np.float32)
        edge_index = np.asarray(d["edge_index"], dtype=np.int32)
        edge_attr = np.asarray(d["edge_attr"], dtype=np.float32)
        targets = np.asarray(d["targets"], dtype=np.float32)
        core_mask = np.asarray(d["core_mask_local"], dtype=bool)
        global_node_ids = np.asarray(d["global_node_ids"], dtype=np.int64)
    return {
        "x": x,
        "senders": edge_index[0],
        "receivers": edge_index[1],
        "edge_attr": edge_attr,
        "targets": targets,
        "core_mask": core_mask,
        "global_node_ids": global_node_ids,
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


def _config_to_dict(config) -> dict:
    if isinstance(config, dict):
        return config
    return vars(config)


def _load_sbi_scaler_stats_cpu(cache_path: Path) -> tuple[object, object]:
    """Load (target_scaler, stats) from a monolithic SBI cache without touching GPU.

    The cache may contain JAX arrays; unpickling them can trigger device_put on GPU.
    We avoid that by loading in a CPU-only subprocess and writing a tiny pickle.
    """
    cache_path = cache_path.expanduser().resolve()
    if not cache_path.exists():
        raise FileNotFoundError(f"SBI cache not found: {cache_path}")
    with tempfile.TemporaryDirectory() as td:
        out_pkl = Path(td) / "scaler_stats.pkl"
        code = r"""
import os, pickle, sys
from pathlib import Path

# Force CPU backend during unpickle to avoid GPU device_put.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

cache_path = Path(sys.argv[1]).expanduser().resolve()
out_path = Path(sys.argv[2]).expanduser().resolve()
with cache_path.open("rb") as f:
    obj = pickle.load(f)
payload = {
    "target_scaler": obj.get("target_scaler"),
    "stats": obj.get("stats"),
}
with out_path.open("wb") as f:
    pickle.dump(payload, f)
"""
        proc = subprocess.run(
            [sys.executable, "-c", code, str(cache_path), str(out_pkl)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "JAX_PLATFORM_NAME": "cpu", "JAX_PLATFORMS": "cpu"},
        )
        _ = proc.stdout  # keep for debugging if needed
        with out_pkl.open("rb") as f:
            payload = pickle.load(f)
    return payload.get("target_scaler"), payload.get("stats")


def _make_gnn_and_flow_from_checkpoint_arrays(
    flow_arrays,
    config: dict,
    graph_for_init: jraph.GraphsTuple,
    key: jax.Array,
):
    """Rebuild the trained flow from checkpoint `flow_arrays` + a fresh static template."""
    gnn_fn = make_gnn_encoder(
        num_passes=int(config["num_passes"]),
        latent_size=int(config["latent_size"]),
        num_heads=int(config["num_heads"]),
        dropout_rate=float(config.get("dropout", 0.2)),
    )
    gnn = hk.transform(gnn_fn)
    gnn_key, flow_key = jax.random.split(key)
    _ = gnn.init(gnn_key, graph_for_init, is_training=False)
    flow_template = masked_autoregressive_flow(
        flow_key,
        base_dist=Normal(jnp.zeros(3), jnp.ones(3)),
        cond_dim=int(config["latent_size"]),
        flow_layers=int(config.get("num_flow_layers", 5)),
        nn_width=int(config.get("flow_hidden_size", 128)),
        nn_depth=2,
        transformer=RationalQuadraticSpline(knots=int(config.get("num_bins", 8)), interval=12),
    )
    _, flow_static = eqx.partition(flow_template, eqx.is_inexact_array)
    flow = eqx.combine(flow_arrays, flow_static)
    return gnn, flow


def _plot_training_history_logs(logs: dict, output_dir: Path) -> None:
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


def _plot_training_history(logs_path: Path, output_dir: Path) -> None:
    if not logs_path.exists():
        print(f"Logs not found at {logs_path}; skipping training plot.")
        return
    with logs_path.open("rb") as f:
        logs = pickle.load(f)
    _plot_training_history_logs(logs, output_dir)


def _sample_posterior(flow, embedding: np.ndarray, num_samples: int, key: jax.Array) -> np.ndarray:
    samples = flow.sample(key, (num_samples,), condition=jnp.asarray(embedding))
    return np.asarray(samples)


def _sample_posterior_batch(
    flow,
    embeddings: np.ndarray,
    num_samples: int,
    key: jax.Array,
) -> np.ndarray:
    """Sample posteriors for a batch of node embeddings via vmap."""
    cond = jnp.asarray(embeddings)
    keys = jax.random.split(key, cond.shape[0])

    def _one(k, c):
        return flow.sample(k, (num_samples,), condition=c)

    samples = jax.vmap(_one)(keys, cond)  # [B, S, 3]
    return np.asarray(samples)


def _bootstrap_tarp_bands(
    samples_tarp: np.ndarray,
    truth: np.ndarray,
    *,
    num_resamples: int,
    ci_percent: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap confidence bands for ECP(alpha) by resampling test points."""
    if num_resamples <= 0:
        ecp, alpha = tarp.get_tarp_coverage(samples_tarp, truth, norm=True, bootstrap=False)
        return np.asarray(alpha), np.asarray(ecp), np.asarray(ecp)

    rng = np.random.default_rng(seed)
    n_points = int(truth.shape[0])
    ecp0, alpha = tarp.get_tarp_coverage(samples_tarp, truth, norm=True, bootstrap=False)
    alpha = np.asarray(alpha)
    ecp0 = np.asarray(ecp0)
    boot = np.empty((num_resamples, ecp0.shape[0]), dtype=np.float32)
    for b in range(num_resamples):
        idx = rng.integers(0, n_points, size=n_points)
        res_samples = samples_tarp[:, idx, :]
        res_truth = truth[idx]
        ecp_b, _ = tarp.get_tarp_coverage(res_samples, res_truth, norm=True, bootstrap=False)
        boot[b] = np.asarray(ecp_b, dtype=np.float32)
    lo_q = 50.0 - ci_percent / 2.0
    hi_q = 50.0 + ci_percent / 2.0
    lo = np.percentile(boot, lo_q, axis=0)
    hi = np.percentile(boot, hi_q, axis=0)
    return alpha, lo, hi


def _collect_structural_stats(
    *,
    flow,
    embeddings: np.ndarray,
    truths_scaled: np.ndarray,
    num_samples: int,
    batch_size: int,
    coverage_alpha: float,
    key: jax.Array,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, jax.Array]:
    """Collect posterior mean/std and interval coverage flags in batches."""
    n = embeddings.shape[0]
    means = []
    stds = []
    covered = []
    q_lo = (1.0 - coverage_alpha) / 2.0
    q_hi = 1.0 - q_lo
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        key, sk = jax.random.split(key)
        s = _sample_posterior_batch(flow, embeddings[start:end], num_samples, sk)  # [B, S, 3]
        m = np.mean(s, axis=1)
        sd = np.std(s, axis=1)
        lo = np.quantile(s, q_lo, axis=1)
        hi = np.quantile(s, q_hi, axis=1)
        cov = (truths_scaled[start:end] >= lo) & (truths_scaled[start:end] <= hi)
        means.append(m)
        stds.append(sd)
        covered.append(cov.astype(np.float32))
    return np.concatenate(means, axis=0), np.concatenate(stds, axis=0), np.concatenate(covered, axis=0), key


def _collect_calibration_ranks_batched(
    *,
    flow,
    embeddings: np.ndarray,
    true_trans: np.ndarray,
    true_raw: np.ndarray,
    target_scaler,
    use_transformed_eig: bool,
    num_samples: int,
    batch_size: int,
    key: jax.Array,
) -> tuple[np.ndarray, np.ndarray, jax.Array]:
    """Compute SBC-style ranks in batches to improve throughput."""
    n = embeddings.shape[0]
    ranks_trans = []
    ranks_raw = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        if end % 200 == 0 or end == n:
            print(f"Calibration sampling {end}/{n}...")
        key, sk = jax.random.split(key)
        s_scaled = _sample_posterior_batch(flow, embeddings[start:end], num_samples, sk)  # [B, S, 3]
        bsz = s_scaled.shape[0]
        s_scaled_2d = s_scaled.reshape(bsz * num_samples, -1)
        s_trans_2d = target_scaler.inverse_transform(s_scaled_2d)
        s_raw_2d = samples_to_raw_eigenvalues(s_scaled_2d, target_scaler, use_transformed_eig)
        s_trans = s_trans_2d.reshape(bsz, num_samples, -1)
        s_raw = s_raw_2d.reshape(bsz, num_samples, -1)
        ranks_trans.append(np.mean(s_trans < true_trans[start:end, None, :], axis=1))
        ranks_raw.append(np.mean(s_raw < true_raw[start:end, None, :], axis=1))
    return np.concatenate(ranks_trans, axis=0), np.concatenate(ranks_raw, axis=0), key


def _plot_pred_vs_true(
    true_vals: np.ndarray,
    pred_vals: np.ndarray,
    unc_vals: np.ndarray,
    param_names: list[str],
    out_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for j in range(3):
        x = true_vals[:, j]
        y = pred_vals[:, j]
        c = unc_vals[:, j]
        vmin = min(np.min(x), np.min(y))
        vmax = max(np.max(x), np.max(y))
        sc = axes[j].scatter(x, y, c=c, s=10, alpha=0.55, cmap="viridis")
        axes[j].plot([vmin, vmax], [vmin, vmax], "r--", linewidth=1.5, label="Ideal")
        axes[j].set_xlabel(f"True {param_names[j]}")
        axes[j].set_ylabel(f"Predicted {param_names[j]}")
        axes[j].set_title(param_names[j])
        axes[j].grid(alpha=0.25)
        axes[j].legend(loc="upper left", fontsize=8)
        cbar = plt.colorbar(sc, ax=axes[j], fraction=0.046, pad=0.04)
        cbar.set_label("Posterior std (scaled)")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def _plot_residual_vs_degree(
    degree: np.ndarray,
    residuals: np.ndarray,
    param_names: list[str],
    out_path: Path,
) -> None:
    log_deg = np.log10(1.0 + degree)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    n_bins = 12
    bins = np.linspace(np.min(log_deg), np.max(log_deg), n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    for j in range(3):
        axes[j].scatter(log_deg, residuals[:, j], s=6, alpha=0.15, color="steelblue")
        binned = np.full(n_bins, np.nan, dtype=np.float32)
        for b in range(n_bins):
            mask = (log_deg >= bins[b]) & (log_deg < bins[b + 1])
            if np.any(mask):
                binned[b] = float(np.mean(residuals[mask, j]))
        axes[j].plot(centers, binned, color="darkorange", linewidth=2.0, label="Binned mean residual")
        axes[j].axhline(0.0, color="red", linestyle="--", linewidth=1.5)
        axes[j].set_xlabel(r"$\log_{10}(1+\mathrm{degree})$")
        axes[j].set_ylabel(f"Residual ({param_names[j]})")
        axes[j].set_title(param_names[j])
        axes[j].grid(alpha=0.25)
        axes[j].legend(fontsize=8)
    plt.suptitle("Residual vs Local Degree")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def _plot_coverage_vs_degree(
    degree: np.ndarray,
    covered: np.ndarray,
    nominal_alpha: float,
    param_names: list[str],
    out_path: Path,
) -> None:
    log_deg = np.log10(1.0 + degree)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    n_bins = 12
    bins = np.linspace(np.min(log_deg), np.max(log_deg), n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    for j in range(3):
        cov = np.full(n_bins, np.nan, dtype=np.float32)
        counts = np.zeros(n_bins, dtype=np.int32)
        for b in range(n_bins):
            mask = (log_deg >= bins[b]) & (log_deg < bins[b + 1])
            counts[b] = int(np.sum(mask))
            if counts[b] > 0:
                cov[b] = float(np.mean(covered[mask, j]))
        axes[j].plot(centers, cov, "o-", color="steelblue", label="Empirical coverage")
        axes[j].axhline(nominal_alpha, color="red", linestyle="--", linewidth=1.5, label=f"Nominal {nominal_alpha:.2f}")
        axes[j].set_xlabel(r"$\log_{10}(1+\mathrm{degree})$")
        axes[j].set_ylabel("Coverage")
        axes[j].set_ylim(0.0, 1.0)
        axes[j].set_title(param_names[j])
        axes[j].grid(alpha=0.25)
        axes[j].legend(fontsize=8)
    plt.suptitle(f"Coverage vs Local Degree ({nominal_alpha:.2f} interval)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def _set_equal_3d_axes(ax, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    """Set equal scaling for 3D axes to avoid visual distortion."""
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    z_min, z_max = float(np.min(z)), float(np.max(z))
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    x_mid = 0.5 * (x_max + x_min)
    y_mid = 0.5 * (y_max + y_min)
    z_mid = 0.5 * (z_max + z_min)
    half = 0.5 * max_range if max_range > 0 else 1.0
    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)
    ax.set_zlim(z_mid - half, z_mid + half)


def _plot_eigenvalue_vector_field(
    positions_xyz: np.ndarray,
    eig_true_raw: np.ndarray,
    eig_pred_raw: np.ndarray,
    out_path: Path,
    *,
    vector_magnification: float,
    vector_linewidth: float,
) -> None:
    """Visualize galaxies as 3D eigenvalue vectors, colored by ||lambda||."""
    mag_true = np.linalg.norm(eig_true_raw, axis=1)
    mag_pred = np.linalg.norm(eig_pred_raw, axis=1)
    x = positions_xyz[:, 0]
    y = positions_xyz[:, 1]
    z = positions_xyz[:, 2]
    u_t, v_t, w_t = eig_true_raw[:, 0], eig_true_raw[:, 1], eig_true_raw[:, 2]
    u_p, v_p, w_p = eig_pred_raw[:, 0], eig_pred_raw[:, 1], eig_pred_raw[:, 2]
    fig = plt.figure(figsize=(16, 7))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    norm = plt.Normalize(vmin=float(min(np.min(mag_true), np.min(mag_pred))), vmax=float(max(np.max(mag_true), np.max(mag_pred))))
    colors_true = plt.cm.magma(norm(mag_true))
    colors_pred = plt.cm.magma(norm(mag_pred))
    q1 = ax1.quiver(
        x,
        y,
        z,
        u_t,
        v_t,
        w_t,
        color=colors_true,
        linewidth=vector_linewidth,
        arrow_length_ratio=0.35,
        length=vector_magnification,
        normalize=False,
        alpha=0.95,
    )
    q2 = ax2.quiver(
        x,
        y,
        z,
        u_p,
        v_p,
        w_p,
        color=colors_pred,
        linewidth=vector_linewidth,
        arrow_length_ratio=0.35,
        length=vector_magnification,
        normalize=False,
        alpha=0.95,
    )
    ax1.set_title("True Eigenvalue Vector Field (3D)")
    ax2.set_title("Predicted Eigenvalue Vector Field (3D)")
    for ax in (ax1, ax2):
        ax.set_xlabel("Position X")
        ax.set_ylabel("Position Y")
        ax.set_zlabel("Position Z")
        ax.grid(False)
        _set_equal_3d_axes(ax, x, y, z)
    cbar1 = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="magma"), ax=ax1, fraction=0.046, pad=0.04)
    cbar2 = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="magma"), ax=ax2, fraction=0.046, pad=0.04)
    cbar1.set_label(r"$||\lambda||$")
    cbar2.set_label(r"$||\lambda||$")
    plt.suptitle(r"Galaxy Eigenvalue Vectors in True Cartesian 3D: $(\lambda_1,\lambda_2,\lambda_3)$, color $||\lambda||$")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
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
    all_x: list[np.ndarray] = []
    all_degree: list[np.ndarray] = []
    all_global_ids: list[np.ndarray] = []
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
        degree = np.bincount(
            np.concatenate([arr["senders"], arr["receivers"]]),
            minlength=int(arr["n_nodes"]),
        ).astype(np.float32)

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
        all_x.append(arr["x"][keep])
        all_degree.append(degree[keep])
        all_global_ids.append(arr["global_node_ids"][keep])
        if (i + 1) % 10 == 0:
            print(f"Processed test partitions: {i + 1}/{len(test_parts)}")

    emb_all = np.concatenate(all_emb, axis=0)
    tgt_all = np.concatenate(all_tgt, axis=0)
    x_all = np.concatenate(all_x, axis=0)
    degree_all = np.concatenate(all_degree, axis=0)
    global_ids_all = np.concatenate(all_global_ids, axis=0)
    if emb_all.shape[0] > max_test_nodes:
        take = rng.choice(emb_all.shape[0], size=max_test_nodes, replace=False)
        emb_all = emb_all[take]
        tgt_all = tgt_all[take]
        x_all = x_all[take]
        degree_all = degree_all[take]
        global_ids_all = global_ids_all[take]
    return emb_all, tgt_all, x_all, degree_all, global_ids_all, total_core


def main(args: argparse.Namespace) -> None:
    _apply_dark_plot_theme()
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve() if args.checkpoint_path else None
    model_path = Path(args.model_path).expanduser().resolve() if args.model_path else None

    if checkpoint_path is not None:
        with checkpoint_path.open("rb") as f:
            ckpt = pickle.load(f)
        config = _config_to_dict(ckpt["config"])
        if args.checkpoint_weights == "best":
            if ckpt.get("best") is not None:
                gnn_params, flow_arrays = ckpt["best"]
                print("Using best-validation weights from checkpoint.")
            else:
                gnn_params, flow_arrays = ckpt["gnn_params"], ckpt["flow_arrays"]
                print("No best weights stored in checkpoint; using last-epoch weights.")
        else:
            gnn_params, flow_arrays = ckpt["gnn_params"], ckpt["flow_arrays"]
            print("Using last-epoch weights from checkpoint.")
        sbi_cache_path = Path(config["sbi_cache_path"]).expanduser().resolve()
        target_scaler, stats = _load_sbi_scaler_stats_cpu(sbi_cache_path)
        # Abacus SBI caches may not store an explicit flag; transformed-eig caches include `stats`.
        use_transformed_eig = bool(stats is not None)
        model_info = {
            "gnn_params": gnn_params,
            "config": config,
            "target_scaler": target_scaler,
            "stats": stats,
            "partition_manifest": config["partition_manifest"],
            "use_transformed_eig": use_transformed_eig,
        }
        flow_path = ""
    else:
        assert model_path is not None
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

    if not isinstance(config, dict):
        config = _config_to_dict(config)

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
    if checkpoint_path is not None:
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Flow : (from checkpoint arrays + static template)")
    else:
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

    init_key = jax.random.key(args.seed)
    if checkpoint_path is not None:
        gnn, flow = _make_gnn_and_flow_from_checkpoint_arrays(flow_arrays, config, graph0, init_key)
    else:
        gnn, flow = _make_gnn_and_flow(config, flow_path, graph0, init_key)

    if args.export_model_dir:
        exp_dir = Path(args.export_model_dir).expanduser().resolve()
        exp_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        flow_export = exp_dir / f"partitioned_flow_export_{ts}.eqx"
        eqx.tree_serialise_leaves(flow_export, flow)
        model_export = exp_dir / f"partitioned_model_export_{ts}.pkl"
        with model_export.open("wb") as f:
            pickle.dump(
                {
                    "gnn_params": gnn_params,
                    "flow_path": str(flow_export),
                    "config": config,
                    "target_scaler": target_scaler,
                    "stats": model_info.get("stats"),
                    "partition_manifest": str(manifest_path),
                    "use_transformed_eig": use_transformed_eig,
                },
                f,
            )
        print(f"Exported flow: {flow_export}")
        print(f"Exported model: {model_export}")

    test_embeddings, test_targets_scaled, test_node_features, test_node_degree, test_global_ids, total_test_core = _collect_test_samples(
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

    if checkpoint_path is not None:
        _plot_training_history_logs(ckpt.get("history", {}), output_dir)
    else:
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
    ranks_trans, ranks_raw, key = _collect_calibration_ranks_batched(
        flow=flow,
        embeddings=test_embeddings[idx_cal],
        true_trans=true_trans[idx_cal],
        true_raw=true_raw[idx_cal],
        target_scaler=target_scaler,
        use_transformed_eig=use_transformed_eig,
        num_samples=args.num_samples,
        batch_size=max(1, int(args.calibration_batch_size)),
        key=key,
    )

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

    # Structural diagnostics: prediction quality and environment-conditioned reliability.
    n_struct = min(args.structural_num_nodes, n)
    if n_struct > 0:
        idx_struct = rng.choice(n, size=n_struct, replace=False)
        struct_emb = test_embeddings[idx_struct]
        struct_tgt_scaled = test_targets_scaled[idx_struct]
        struct_feat = test_node_features[idx_struct]
        struct_deg = test_node_degree[idx_struct]
        struct_global_ids = test_global_ids[idx_struct]
        pred_mean_scaled, pred_std_scaled, covered_scaled, key = _collect_structural_stats(
            flow=flow,
            embeddings=struct_emb,
            truths_scaled=struct_tgt_scaled,
            num_samples=args.num_samples,
            batch_size=max(1, int(args.structural_batch_size)),
            coverage_alpha=float(args.structural_coverage_alpha),
            key=key,
        )
        true_trans_struct = target_scaler.inverse_transform(struct_tgt_scaled)
        pred_trans_struct = target_scaler.inverse_transform(pred_mean_scaled)
        true_raw_struct = samples_to_raw_eigenvalues(struct_tgt_scaled, target_scaler, use_transformed_eig)
        pred_raw_struct = samples_to_raw_eigenvalues(pred_mean_scaled, target_scaler, use_transformed_eig)

        _plot_pred_vs_true(
            true_trans_struct,
            pred_trans_struct,
            pred_std_scaled,
            param_names_trans,
            output_dir / "flowjax_pred_vs_true_transformed.png",
            "Predicted vs True (Transformed Space)",
        )
        _plot_pred_vs_true(
            true_raw_struct,
            pred_raw_struct,
            pred_std_scaled,
            param_names_raw,
            output_dir / "flowjax_pred_vs_true_raw_eig.png",
            "Predicted vs True (Raw Eigenvalues)",
        )
        _plot_residual_vs_degree(
            struct_deg,
            pred_raw_struct - true_raw_struct,
            param_names_raw,
            output_dir / "flowjax_residual_vs_degree_raw_eig.png",
        )
        _plot_coverage_vs_degree(
            struct_deg,
            covered_scaled,
            float(args.structural_coverage_alpha),
            param_names_trans,
            output_dir / "flowjax_coverage_vs_degree.png",
        )

        points_xyz_path = Path(args.points_xyz_path).expanduser().resolve()
        if points_xyz_path.exists():
            points_xyz = np.load(points_xyz_path)
            n_vec = min(int(args.vector_max_points), n_struct)
            vec_take = rng.choice(n_struct, size=n_vec, replace=False)
            vec_global_ids = struct_global_ids[vec_take].astype(np.int64)
            if np.max(vec_global_ids) >= points_xyz.shape[0] or np.min(vec_global_ids) < 0:
                print(
                    f"Skipping eigenvalue vector field: global ids out of bounds for "
                    f"{points_xyz_path} with shape {points_xyz.shape}."
                )
            else:
                pos_xyz = np.asarray(points_xyz[vec_global_ids, :3], dtype=np.float32)
                _plot_eigenvalue_vector_field(
                    pos_xyz,
                    true_raw_struct[vec_take],
                    pred_raw_struct[vec_take],
                    output_dir / "flowjax_eigenvalue_vector_field.png",
                    vector_magnification=float(args.vector_magnification),
                    vector_linewidth=float(args.vector_linewidth),
                )
        else:
            print(
                f"Skipping eigenvalue vector field: points_xyz_path does not exist: {points_xyz_path}"
            )

    if TARP_AVAILABLE:
        n_tarp = min(args.num_tarp, n)
        idx_tarp = rng.choice(n, size=n_tarp, replace=False)
        all_samples = []
        all_truth = []
        batch_size = max(1, int(args.tarp_batch_size))
        for start in range(0, n_tarp, batch_size):
            end = min(start + batch_size, n_tarp)
            if end % 100 == 0 or end == n_tarp:
                print(f"TARP sampling {end}/{n_tarp}...")
            sel = idx_tarp[start:end]
            key, sk = jax.random.split(key)
            batch_samples = _sample_posterior_batch(flow, test_embeddings[sel], args.num_samples, sk)
            all_samples.append(batch_samples)
            all_truth.append(test_targets_scaled[sel])
        all_samples = np.concatenate(all_samples, axis=0)  # [N, S, 3]
        all_truth = np.concatenate(all_truth, axis=0)      # [N, 3]
        samples_tarp = np.transpose(all_samples, (1, 0, 2))  # [S, N, 3]
        ecp, alpha = tarp.get_tarp_coverage(samples_tarp, all_truth, norm=True, bootstrap=False)
        alpha = np.asarray(alpha)
        ecp = np.asarray(ecp)
        alpha_b, ecp_lo, ecp_hi = _bootstrap_tarp_bands(
            samples_tarp,
            all_truth,
            num_resamples=int(args.tarp_bootstrap_resamples),
            ci_percent=float(args.tarp_bootstrap_ci),
            seed=int(args.seed + 999),
        )
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot([0, 1], [0, 1], "--", color="#9aa4b2", linewidth=2, label="Ideal")
        ax.plot(alpha, ecp, "b-", linewidth=2, label="TARP")
        if int(args.tarp_bootstrap_resamples) > 0:
            ax.fill_between(
                alpha_b,
                ecp_lo,
                ecp_hi,
                color="steelblue",
                alpha=0.25,
                label=f"{float(args.tarp_bootstrap_ci):.1f}% bootstrap band",
            )
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
            bbox=dict(boxstyle="round", facecolor="#1f6feb", edgecolor="#58a6ff", alpha=0.25),
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
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--model-path", default=None, help="Path to partitioned_model_seed_*.pkl (finished training export).")
    src.add_argument(
        "--checkpoint-path",
        default=None,
        help="Path to checkpoint_latest.pkl or checkpoint_epoch_*.pkl from jraph_sbi_flowjax_partitioned.py.",
    )
    parser.add_argument(
        "--checkpoint-weights",
        choices=("best", "last"),
        default="best",
        help="With --checkpoint-path: use best-validation snapshot (if present) or last-epoch weights.",
    )
    parser.add_argument(
        "--export-model-dir",
        default="",
        help="If set, write partitioned_model_export_*.pkl + partitioned_flow_export_*.eqx here (for reuse as --model-path).",
    )
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
    parser.add_argument("--calibration-batch-size", type=int, default=256, help="Batch size for calibration rank sampling.")
    parser.add_argument("--num-tarp", type=int, default=500, help="Number of nodes for TARP coverage.")
    parser.add_argument("--structural-num-nodes", type=int, default=4000, help="Number of nodes for structural diagnostics.")
    parser.add_argument("--structural-batch-size", type=int, default=128, help="Batch size for structural diagnostics sampling.")
    parser.add_argument(
        "--structural-coverage-alpha",
        type=float,
        default=0.9,
        help="Nominal interval level for coverage-vs-degree diagnostics.",
    )
    parser.add_argument(
        "--points-xyz-path",
        default="/pscratch/sd/d/dkololgi/abacus/graph_constructions/abacus_alpha_points_xyz.npy",
        help="Path to global Cartesian positions array (N,3) used for vector-field plotting.",
    )
    parser.add_argument(
        "--vector-pos-cols",
        default="0,1,2",
        help="Deprecated; ignored. Vector field now always uses --points-xyz-path and global_node_ids.",
    )
    parser.add_argument("--vector-max-points", type=int, default=3000, help="Max points for eigenvalue vector-field plot.")
    parser.add_argument(
        "--vector-magnification",
        type=float,
        default=2.5,
        help="Global length multiplier for 3D vector arrows.",
    )
    parser.add_argument(
        "--vector-linewidth",
        type=float,
        default=1.2,
        help="Line width for 3D vector arrows.",
    )
    parser.add_argument("--tarp-batch-size", type=int, default=128, help="Batch size for posterior sampling in TARP loop.")
    parser.add_argument(
        "--tarp-bootstrap-resamples",
        type=int,
        default=200,
        help="Bootstrap resamples for TARP uncertainty band (0 disables).",
    )
    parser.add_argument(
        "--tarp-bootstrap-ci",
        type=float,
        default=95.0,
        help="Central confidence interval percent for TARP bootstrap band.",
    )
    parser.add_argument("--test-partition-limit", type=int, default=0, help="Optional cap on test partitions.")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())

