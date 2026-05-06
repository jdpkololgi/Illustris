#!/usr/bin/env python3
"""Evaluate a trained Jraph classification checkpoint (or raw params .pkl) against an SBI/Jraph cache.

Loads the same artifacts as ``jraph_pipeline.py`` (``--cache_path``), rebuilds the graph net with
matching architecture flags, runs a single forward pass, then writes:

- metrics report (train/val/test): accuracy + mean cross-entropy loss
- optional confusion-matrix PNGs
- predictions pickle (logits + preds + targets + masks + meta)

Example::

  python workflows/jraph/jraph_classification_eval_from_checkpoint.py \\
    --cache_path /pscratch/.../abacus_delaunay_cube_..._sbi_cache.pkl \\
    --model_path /pscratch/.../cube_classification/checkpoints/ckpt_classification_..._best.pkl \\
    --output_dir /pscratch/.../cube_classification/analysis_eval_best_YYYYMMDD_HHMMSS \\
    --latent_size 96 --dropout 0.15 --num_passes 8 --num_heads 8 --seed 42
"""

from __future__ import annotations

import argparse
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
import haiku as hk
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from shared.graph_net_models import make_graph_network


def _load_params(model_path: Path) -> tuple[object, int | None]:
    with model_path.open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "params" in obj:
        return obj["params"], obj.get("epoch")
    return obj, None


def _softmax_cross_entropy_mean(logits: np.ndarray, y: np.ndarray) -> float:
    """Mean softmax cross entropy for integer labels."""
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D [N,C], got {logits.shape}")
    if y.ndim != 1 or y.shape[0] != logits.shape[0]:
        raise ValueError(f"labels must be 1D [N], got {y.shape} for logits {logits.shape}")
    max_l = np.max(logits, axis=1, keepdims=True)
    lse = np.log(np.sum(np.exp(logits - max_l), axis=1, keepdims=True)) + max_l
    log_probs = logits - lse
    n = y.shape[0]
    return float(-np.mean(log_probs[np.arange(n), y]))


def _accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.int64).ravel()
    y = np.asarray(y, dtype=np.int64).ravel()
    if pred.shape != y.shape:
        return float("nan")
    if pred.size == 0:
        return float("nan")
    return float(np.mean(pred == y))


def _class_balance_lines(y: np.ndarray, mask: np.ndarray | None, *, num_classes: int, label: str) -> list[str]:
    yy = np.asarray(y, dtype=np.int64).ravel()
    if mask is not None:
        mm = np.asarray(mask, dtype=bool).ravel()
        yy = yy[mm]
    counts = np.bincount(yy, minlength=int(num_classes)).astype(np.int64) if yy.size else np.zeros(int(num_classes), dtype=np.int64)
    total = int(counts.sum())
    fracs = counts / max(total, 1)
    lines = [f"{label}: total={total}"]
    for k in range(int(num_classes)):
        lines.append(f"  class {k}: n={int(counts[k])}  frac={float(fracs[k]):.6f}")
    return lines


def _plot_confusion(cm: np.ndarray, out_png: Path, title: str, *, normalize: bool = False) -> None:
    """Plot confusion matrix.

    If normalize=True, row-normalize by true class (each row sums to 1 when possible).
    """
    cm_int = np.asarray(cm, dtype=np.int64)
    if normalize:
        denom = cm_int.sum(axis=1, keepdims=True).astype(np.float64)
        cm_plot = np.divide(cm_int, np.maximum(denom, 1.0), dtype=np.float64)
    else:
        cm_plot = cm_int
    fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.4))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("pred")
    ax.set_ylabel("true")
    ax.set_xticks(range(cm_plot.shape[1]))
    ax.set_yticks(range(cm_plot.shape[0]))
    # annotate
    thresh = float(np.nanmax(cm_plot) * 0.6) if cm_plot.size else 0.0
    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            v = cm_plot[i, j]
            if normalize:
                txt = f"{v:.2f}"
            else:
                txt = str(int(cm_int[i, j]))
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color="white" if v > thresh else "black",
                fontsize=9,
            )
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
    p.add_argument("--no_plots", action="store_true", help="Disable confusion matrix plots.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = args.cache_path.expanduser().resolve()
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")

    with cache_path.open("rb") as f:
        data = pickle.load(f)
    graph = data["graph"]
    y = np.asarray(data["classification_labels"], dtype=np.int64)
    train_mask, val_mask, test_mask = data["masks"]
    train_mask_np = np.asarray(train_mask, dtype=bool)
    val_mask_np = np.asarray(val_mask, dtype=bool)
    test_mask_np = np.asarray(test_mask, dtype=bool)

    num_classes = int(np.max(y)) + 1 if y.size else 4
    if num_classes < 2:
        num_classes = 4
    num_classes = int(num_classes)

    params, ckpt_epoch = _load_params(args.model_path.expanduser().resolve())
    meta = {
        "cache_path": str(cache_path),
        "model_path": str(args.model_path),
        "checkpoint_epoch": ckpt_epoch,
        "eval_timestamp": ts,
        "prediction_mode": "classification",
        "num_classes": int(num_classes),
        "latent_size": args.latent_size,
        "num_heads": args.num_heads,
        "num_passes": args.num_passes,
        "dropout": args.dropout,
        "seed": args.seed,
    }
    with (out_dir / f"eval_meta_{ts}.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    net_fn = make_graph_network(
        num_passes=args.num_passes,
        latent_size=args.latent_size,
        num_heads=args.num_heads,
        dropout_rate=args.dropout,
        output_dim=int(num_classes),
    )
    net = hk.transform(net_fn)

    @jax.jit
    def predict(p, g, rng):
        return net.apply(p, rng, g, is_training=False).nodes

    rng = jax.random.PRNGKey(args.seed + 999)
    logits = np.asarray(predict(params, graph, rng))
    if logits.ndim != 2:
        raise ValueError(f"Unexpected logits shape {logits.shape} (expected [N,C])")
    preds = np.asarray(np.argmax(logits, axis=1), dtype=np.int64)

    splits = {
        "train": train_mask_np,
        "val": val_mask_np,
        "test": test_mask_np,
    }

    metrics_by_split: dict[str, dict[str, float]] = {}
    reports: dict[str, str] = {}
    cms: dict[str, np.ndarray] = {}
    for name, m in splits.items():
        yy = y[m]
        pp = preds[m]
        ll = logits[m]
        metrics_by_split[name] = {
            "accuracy": _accuracy(pp, yy),
            "xent": _softmax_cross_entropy_mean(ll, yy) if yy.size else float("nan"),
            "n": float(int(yy.size)),
        }
        if yy.size:
            cms[name] = confusion_matrix(yy, pp, labels=list(range(int(num_classes))))
            reports[name] = classification_report(
                yy,
                pp,
                labels=list(range(int(num_classes))),
                zero_division=0,
            )
        else:
            cms[name] = np.zeros((int(num_classes), int(num_classes)), dtype=np.int64)
            reports[name] = ""

    # Human-readable metrics file.
    report_lines = [
        f"jraph_classification_eval_from_checkpoint  {ts}",
        f"model_path={args.model_path}",
        f"checkpoint_epoch={ckpt_epoch}",
        f"cache_path={cache_path}",
        f"num_classes={num_classes}",
        "",
        "Class balance (fractions):",
        *_class_balance_lines(y, None, num_classes=num_classes, label="  [all]"),
        *_class_balance_lines(y, train_mask_np, num_classes=num_classes, label="  [train]"),
        *_class_balance_lines(y, val_mask_np, num_classes=num_classes, label="  [val]"),
        *_class_balance_lines(y, test_mask_np, num_classes=num_classes, label="  [test]"),
        "",
        "Per-split metrics:",
    ]
    for name in ("train", "val", "test"):
        d = metrics_by_split[name]
        report_lines.append(f"  [{name}]")
        report_lines.append(f"    accuracy: {d['accuracy']:.6f}")
        report_lines.append(f"    xent: {d['xent']:.6f}")
        report_lines.append(f"    n: {int(d['n'])}")
    report_path = out_dir / f"eval_metrics_{ts}.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(report_path.read_text())

    # Save detailed classification report (test split).
    test_report_path = out_dir / f"eval_classification_report_test_{ts}.txt"
    test_report_path.write_text(reports.get("test", "") + "\n", encoding="utf-8")
    print(f"Wrote classification report: {test_report_path}")

    # Predictions pickle.
    preds_data = {
        "logits": logits,
        "preds": preds,
        "targets": y,
        "train_mask": train_mask_np,
        "val_mask": val_mask_np,
        "test_mask": test_mask_np,
        "metrics_by_split": metrics_by_split,
        "meta": meta,
    }
    preds_path = out_dir / f"eval_predictions_{ts}.pkl"
    with preds_path.open("wb") as f:
        pickle.dump(preds_data, f)
    print(f"Wrote predictions: {preds_path}")

    if not args.no_plots:
        for name in ("train", "val", "test"):
            cm = cms[name]
            _plot_confusion(cm, out_dir / f"eval_confusion_{name}_{ts}.png", title=f"Confusion ({name})")
            _plot_confusion(
                cm,
                out_dir / f"eval_confusion_{name}_{ts}_normalized.png",
                title=f"Confusion ({name}) normalized (per true class)",
                normalize=True,
            )
        print(f"Wrote confusion plots under {out_dir}")


if __name__ == "__main__":
    main()

