#!/usr/bin/env python3
"""Evaluate a trained Jraph regression checkpoint (or raw params .pkl) against an SBI/Jraph cache.

Loads the same artifacts as ``jraph_pipeline.py`` (``--cache_path``), rebuilds the graph net with
matching architecture flags, runs a single forward pass, then writes:

- metrics report (test set), matching the transformed-eig path when applicable
- predictions pickle (same schema as the training script's post-hoc block)
- optional scatter PNGs (pred vs true) for physical eigenvalues on train/val/test

Example::

  python workflows/jraph/jraph_regression_eval_from_checkpoint.py \\
    --cache_path /pscratch/.../abacus_delaunay_cube_..._sbi_cache.pkl \\
    --model_path /pscratch/.../cube_regression/checkpoints/ckpt_..._best_epoch_009346.pkl \\
    --output_dir /pscratch/.../cube_regression/analysis_epoch_9346 \\
    --latent_size 96 --dropout 0.15 --num_passes 8 --num_heads 8 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# CPU-friendly default for analysis on login nodes; override with CUDA_VISIBLE_DEVICES on GPU nodes.
os.environ.setdefault("JAX_PLATFORMS", os.environ.get("JAX_PLATFORMS", "cpu"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import haiku as hk
import jraph
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.eigenvalue_transformations import increments_to_eigenvalues
from shared.graph_net_models import make_graph_network

# Matches `abacus_graph_features_cugraph.py` / `build_abacus_sbi_cache` node feature order when present.
DEFAULT_NODE_FEATURE_NAMES = (
    "Degree",
    "Clustering",
    "Density",
    "Neigh Density",
    "I_eig1",
    "I_eig2",
    "I_eig3",
)


def _load_params(model_path: Path) -> tuple[object, int | None]:
    with model_path.open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "params" in obj:
        return obj["params"], obj.get("epoch")
    return obj, None


def _eig_from_15d_linear(raw15: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Reconstruct (λ1, λ2, λ3) from 15-d linear increment targets (channels 0–2)."""
    raw15 = np.asarray(raw15, dtype=np.float64)
    l1 = raw15[:, 0]
    d2 = np.maximum(raw15[:, 1], eps)
    d3 = np.maximum(raw15[:, 2], eps)
    l2 = l1 + d2
    l3 = l2 + d3
    return np.stack([l1, l2, l3], axis=-1)


def _metrics_block(
    *,
    preds_output: np.ndarray,
    targets_output: np.ndarray,
    use_transformed_eig: bool,
    target_scaler,
    eigenvalues_raw: np.ndarray | None,
    mask: np.ndarray,
) -> dict:
    """Return dict of scalar metrics for one mask (train/val/test)."""
    if use_transformed_eig:
        if target_scaler is not None:
            preds_t = target_scaler.inverse_transform(preds_output)
            targets_t = target_scaler.inverse_transform(targets_output)
        else:
            preds_t = preds_output
            targets_t = targets_output
        preds_eig = np.asarray(increments_to_eigenvalues(jnp.asarray(preds_t)))
        te_full = np.asarray(eigenvalues_raw, dtype=np.float64)
        pt = preds_eig[mask]
        tt = te_full[mask]
        mse_shape = float(np.mean((preds_t[mask] - targets_t[mask]) ** 2))
        mae_shape = float(np.mean(np.abs(preds_t[mask] - targets_t[mask])))
        mse_eig = float(np.mean((pt - tt) ** 2))
        mae_eig = float(np.mean(np.abs(pt - tt)))
        r2_eig = _r2_per_column(tt, pt)
        return {
            "mse_transformed": mse_shape,
            "mae_transformed": mae_shape,
            "mse_eigenvalues": mse_eig,
            "mae_eigenvalues": mae_eig,
            "r2_lambda1": float(r2_eig[0]),
            "r2_lambda2": float(r2_eig[1]),
            "r2_lambda3": float(r2_eig[2]),
            "r2_eigenvalues_mean": float(np.mean(r2_eig)),
        }
    if target_scaler is not None:
        preds_r = target_scaler.inverse_transform(preds_output)
        targets_r = target_scaler.inverse_transform(targets_output)
    else:
        preds_r = preds_output
        targets_r = targets_output
    preds_r = np.asarray(preds_r, dtype=np.float64)
    targets_r = np.asarray(targets_r, dtype=np.float64)
    # 15-d caches: channels 0–2 are (λ1, Δλ2, Δλ3), not physical λ2/λ3.
    is_15d_linear = (
        preds_r.ndim == 2
        and targets_r.ndim == 2
        and preds_r.shape[1] == targets_r.shape[1]
        and preds_r.shape[1] == 15
    )
    if is_15d_linear:
        pe = _eig_from_15d_linear(preds_r)
        if eigenvalues_raw is not None:
            te = np.asarray(eigenvalues_raw, dtype=np.float64)
        else:
            te = _eig_from_15d_linear(targets_r)
        pt = pe[mask]
        tt = te[mask]
        mse_shape = float(np.mean((preds_r[mask] - targets_r[mask]) ** 2))
        mae_shape = float(np.mean(np.abs(preds_r[mask] - targets_r[mask])))
        mse_eig = float(np.mean((pt - tt) ** 2))
        mae_eig = float(np.mean(np.abs(pt - tt)))
        r2_eig = _r2_per_column(tt, pt)
        return {
            "mse_raw_15d": mse_shape,
            "mae_raw_15d": mae_shape,
            "mse_eigenvalues": mse_eig,
            "mae_eigenvalues": mae_eig,
            "r2_lambda1": float(r2_eig[0]),
            "r2_lambda2": float(r2_eig[1]),
            "r2_lambda3": float(r2_eig[2]),
            "r2_eigenvalues_mean": float(np.mean(r2_eig)),
        }
    pr = preds_r[mask]
    tr = targets_r[mask]
    mse = float(np.mean((pr - tr) ** 2))
    mae = float(np.mean(np.abs(pr - tr)))
    r2_e = _r2_per_column(tr, pr)
    return {
        "mse_raw_eigenvalues": mse,
        "mae_raw_eigenvalues": mae,
        "r2_lambda1": float(r2_e[0]),
        "r2_lambda2": float(r2_e[1]),
        "r2_lambda3": float(r2_e[2]),
        "r2_eigenvalues_mean": float(np.mean(r2_e)),
    }


def _r2_per_column(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """R² per target dimension (columns), shape (d,)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    y_mean = np.mean(y_true, axis=0)
    ss_tot = np.sum((y_true - y_mean) ** 2, axis=0)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def _pearson_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or b.size < 2 or a.shape != b.shape:
        return float("nan")
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa < 1e-15 or sb < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _node_feature_names(n_cols: int) -> list[str]:
    if n_cols == len(DEFAULT_NODE_FEATURE_NAMES):
        return list(DEFAULT_NODE_FEATURE_NAMES)
    return [f"node_feat_{j}" for j in range(n_cols)]


def _pearson_eigs_vs_node_features(
    true_eig: np.ndarray,
    node_x: np.ndarray,
    *,
    mask: np.ndarray,
    feature_names: list[str],
) -> list[list[float | str]]:
    """Rows: one per (λk, feature); columns: eigenvalue_idx, feature_name, pearson_r."""
    te = np.asarray(true_eig, dtype=np.float64)[mask]
    x = np.asarray(node_x, dtype=np.float64)[mask]
    rows: list[list[float | str]] = []
    lam_names = ("lambda1", "lambda2", "lambda3")
    for k in range(min(3, te.shape[1])):
        for j, fname in enumerate(feature_names):
            if j >= x.shape[1]:
                break
            r = _pearson_1d(te[:, k], x[:, j])
            rows.append([lam_names[k], fname, r])
    return rows


def _write_pearson_csv(path: Path, rows: list[list[float | str]], split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "eigenvalue", "node_feature", "pearson_r"])
        for row in rows:
            w.writerow([split, row[0], row[1], f"{row[2]:.6g}" if row[2] == row[2] else "nan"])


def _scatter_eigs(preds_eig: np.ndarray, targets_eig: np.ndarray, out_png: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    names = ("λ₁", "λ₂", "λ₃")
    r2_all = _r2_per_column(targets_eig, preds_eig)
    for i, ax in enumerate(axes):
        ax.scatter(targets_eig[:, i], preds_eig[:, i], s=2, alpha=0.25, rasterized=True)
        lo = float(min(targets_eig[:, i].min(), preds_eig[:, i].min()))
        hi = float(max(targets_eig[:, i].max(), preds_eig[:, i].max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlabel(f"true {names[i]}")
        ax.set_ylabel(f"pred {names[i]}")
        ax.set_title(f"$R^2$ = {r2_all[i]:.4f}")
    fig.suptitle(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache_path", type=Path, required=True)
    p.add_argument("--model_path", type=Path, required=True, help="Checkpoint .pkl or final params .pkl")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--latent_size", type=int, default=80)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_passes", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--no_transformed_eig", action="store_true")
    p.add_argument(
        "--heteroscedastic_15d",
        action="store_true",
        help="If cache targets are 15-d, build network with output_dim=30 (mean+logvar) and evaluate using mean head.",
    )
    p.add_argument("--no_plots", action="store_true")
    p.add_argument(
        "--allow_login_node",
        action="store_true",
        help="Skip SLURM GPU guard (use CPU JAX or ensure you are not on a shared login).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ and not args.allow_login_node:
        from shared.resource_requirements import require_gpu_slurm

        require_gpu_slurm("jraph_regression_eval_from_checkpoint.py", min_gpus=1)

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"JAX backend: {jax.default_backend()} | devices: {jax.devices()}")

    with args.cache_path.expanduser().resolve().open("rb") as f:
        data = pickle.load(f)
    graph = data["graph"]
    targets = data["regression_targets"]
    target_scaler = data.get("target_scaler")
    eigenvalues_raw = data.get("eigenvalues_raw")
    stats = data.get("stats")
    train_mask, val_mask, test_mask = data["masks"]
    train_mask_np = np.asarray(train_mask, dtype=bool)
    val_mask_np = np.asarray(val_mask, dtype=bool)
    test_mask_np = np.asarray(test_mask, dtype=bool)
    node_x = np.asarray(graph.nodes, dtype=np.float64)
    node_feature_names = _node_feature_names(node_x.shape[1])

    targets_arr = np.asarray(targets)
    output_dim = int(targets_arr.shape[1]) if targets_arr.ndim == 2 else 3
    # 15-d caches store (λ1, Δλ2, Δλ3, …) in raw space before scaling — not legacy softplus increments.
    use_transformed_eig = not args.no_transformed_eig
    if output_dim == 15:
        use_transformed_eig = False
        if args.heteroscedastic_15d:
            output_dim = 30

    params, ckpt_epoch = _load_params(args.model_path.expanduser().resolve())
    meta = {
        "cache_path": str(args.cache_path),
        "model_path": str(args.model_path),
        "checkpoint_epoch": ckpt_epoch,
        "eval_timestamp": ts,
        "use_transformed_eig": use_transformed_eig,
        "latent_size": args.latent_size,
        "num_heads": args.num_heads,
        "num_passes": args.num_passes,
        "dropout": args.dropout,
        "seed": args.seed,
        "node_feature_names": node_feature_names,
    }
    with (out_dir / f"eval_meta_{ts}.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    net_fn = make_graph_network(
        num_passes=args.num_passes,
        latent_size=args.latent_size,
        num_heads=args.num_heads,
        dropout_rate=args.dropout,
        output_dim=output_dim,
    )
    net = hk.transform(net_fn)

    @jax.jit
    def predict(p, g, rng):
        return net.apply(p, rng, g, is_training=False).nodes

    rng = jax.random.PRNGKey(args.seed + 999)
    outputs = predict(params, graph, rng)
    preds_output_full = np.asarray(outputs)
    # If heteroscedastic, keep only mean head for metric/scatter comparisons.
    if preds_output_full.ndim == 2 and preds_output_full.shape[1] == 30 and targets_arr.ndim == 2 and targets_arr.shape[1] == 15:
        preds_output = preds_output_full[:, :15]
    else:
        preds_output = preds_output_full
    targets_output = np.asarray(targets)

    # Per-split metrics
    splits = {
        "train": train_mask_np,
        "val": val_mask_np,
        "test": test_mask_np,
    }
    metrics_all: dict[str, dict] = {}
    for name, m in splits.items():
        metrics_all[name] = _metrics_block(
            preds_output=preds_output,
            targets_output=targets_output,
            use_transformed_eig=use_transformed_eig,
            target_scaler=target_scaler,
            eigenvalues_raw=np.asarray(eigenvalues_raw) if eigenvalues_raw is not None else None,
            mask=m,
        )

    report_lines = [
        f"jraph_regression_eval_from_checkpoint  {ts}",
        f"model_path={args.model_path}",
        f"checkpoint_epoch={ckpt_epoch}",
        "",
        "Per-split metrics:",
    ]
    for name, d in metrics_all.items():
        report_lines.append(f"  [{name}]")
        for k, v in d.items():
            report_lines.append(f"    {k}: {v}")

    # Pearson: true eigenvalues vs graph node features (input signal check).
    report_lines.extend(["", "Pearson r: true eigenvalue vs node feature (per split):"])
    if eigenvalues_raw is not None:
        te_true = np.asarray(eigenvalues_raw, dtype=np.float64)
        for split_name, m in splits.items():
            prow = _pearson_eigs_vs_node_features(
                te_true, node_x, mask=m, feature_names=node_feature_names
            )
            csv_path = out_dir / f"eval_eig_vs_node_features_pearson_{split_name}_{ts}.csv"
            _write_pearson_csv(csv_path, prow, split_name)
            report_lines.append(f"  [{split_name}] wrote {csv_path.name}")
            lam_labels = ("λ1", "λ2", "λ3")
            for k, lab in enumerate(lam_labels):
                lam_key = f"lambda{k + 1}"
                sub = [r for r in prow if r[0] == lam_key and r[2] == r[2]]
                if not sub:
                    continue
                best = max(sub, key=lambda row: abs(float(row[2])))
                report_lines.append(
                    f"    {lab}: max|corr|={abs(float(best[2])):.4f} (with {best[1]})"
                )
    else:
        report_lines.append("  (skipped: cache has no eigenvalues_raw)")

    report_path = out_dir / f"eval_metrics_{ts}.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(report_path.read_text())

    # Predictions pickle (aligned with training script keys)
    if use_transformed_eig:
        if target_scaler is not None:
            preds_transformed_eig = target_scaler.inverse_transform(preds_output)
            targets_transformed_eig = target_scaler.inverse_transform(targets_output)
        else:
            preds_transformed_eig = preds_output
            targets_transformed_eig = targets_output
        preds_eigenvalues = np.asarray(increments_to_eigenvalues(preds_transformed_eig))
        targets_eigenvalues = np.asarray(eigenvalues_raw)
        preds_data = {
            "preds_transformed_eig": preds_transformed_eig,
            "targets_transformed_eig": targets_transformed_eig,
            "preds_eigenvalues": preds_eigenvalues,
            "targets_eigenvalues": targets_eigenvalues,
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
            "stats": stats,
            "use_transformed_eig": True,
            "metrics_by_split": metrics_all,
            "node_feature_names": node_feature_names,
        }
    else:
        if target_scaler is not None:
            preds_raw = target_scaler.inverse_transform(preds_output)
            targets_raw = target_scaler.inverse_transform(targets_output)
        else:
            preds_raw = preds_output
            targets_raw = targets_output
        preds_data = {
            "preds_scaled": preds_output,
            "preds_raw": preds_raw,
            "targets_scaled": targets_output,
            "targets_raw": targets_raw,
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
            "use_transformed_eig": False,
            "metrics_by_split": metrics_all,
            "node_feature_names": node_feature_names,
        }
        if targets_arr.ndim == 2 and targets_arr.shape[1] == 15:
            preds_data["preds_eigenvalues"] = _eig_from_15d_linear(np.asarray(preds_raw, dtype=np.float64))
            preds_data["targets_eigenvalues"] = (
                np.asarray(eigenvalues_raw, dtype=np.float64)
                if eigenvalues_raw is not None
                else _eig_from_15d_linear(np.asarray(targets_raw, dtype=np.float64))
            )

    preds_path = out_dir / f"eval_predictions_{ts}.pkl"
    with preds_path.open("wb") as f:
        pickle.dump(preds_data, f)
    print(f"Wrote predictions: {preds_path}")

    if not args.no_plots and eigenvalues_raw is not None:
        te = np.asarray(eigenvalues_raw)
        if use_transformed_eig:
            pe = preds_eigenvalues
        elif targets_arr.ndim == 2 and targets_arr.shape[1] == 15:
            pr = (
                target_scaler.inverse_transform(preds_output)
                if target_scaler is not None
                else preds_output
            )
            pe = _eig_from_15d_linear(np.asarray(pr, dtype=np.float64))
        else:
            pe = None

        if pe is not None:
            for name, m in splits.items():
                _scatter_eigs(
                    pe[m],
                    te[m],
                    out_dir / f"eval_scatter_eigenvalues_{name}_{ts}.png",
                    title=f"Pred vs true eigenvalues ({name}) n={int(m.sum())}",
                )
            print(f"Wrote scatter plots under {out_dir}")


if __name__ == "__main__":
    main()
