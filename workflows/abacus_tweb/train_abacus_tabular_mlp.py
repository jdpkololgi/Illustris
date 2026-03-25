#!/usr/bin/env python3
"""Train a simple PyTorch MLP on Abacus tabular graph metrics.

Default dataset pairing:
1) Node features parquet from cuGraph feature export:
   /pscratch/sd/d/dkololgi/abacus/graph_constructions/abacus_alpha_cugraph_node_features.parquet
2) Eigenvalue targets from the source annotated CutSky FITS referenced by
   abacus_alpha_metadata.json (typically .../cutsky_..._with_tweb.fits)

Rows are assumed to be aligned by construction from the same filtered catalog.
The script validates row counts before training.

Supports easy multi-GPU via torch.nn.DataParallel (--multi-gpu).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import fitsio
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.eigenvalue_transformations import eigenvalues_to_increments


DEFAULT_GNN_META = "/pscratch/sd/d/dkololgi/abacus/graph_constructions/abacus_alpha_cugraph_gnn_metadata.json"
DEFAULT_OUTDIR = "/pscratch/sd/d/dkololgi/tng_illustris/outputs/abacus_tabular_mlp"


def _resolve_col(table, candidates: Iterable[str]) -> str:
    names_upper = {name.upper(): name for name in table.dtype.names}
    for candidate in candidates:
        resolved = names_upper.get(candidate.upper())
        if resolved is not None:
            return resolved
    raise KeyError(
        f"None of candidate columns {list(candidates)} found. "
        f"Available columns include: {table.dtype.names[:20]}..."
    )


def _apply_optional_y1y5_filter(table: np.ndarray) -> np.ndarray:
    names_upper = {name.upper(): name for name in table.dtype.names}
    in_y1 = names_upper.get("IN_Y1")
    in_y5 = names_upper.get("IN_Y5")
    if in_y1 is None or in_y5 is None:
        return np.ones(len(table), dtype=bool)
    return (table[in_y1] == 1) | (table[in_y5] == 1)


class TabularMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], out_dim: int = 3, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gnn-metadata-path", default=DEFAULT_GNN_META, help="Path to *_cugraph_gnn_metadata.json")
    p.add_argument("--catalog-path", default="", help="Optional override for source FITS catalog path.")
    p.add_argument("--output-dir", default=DEFAULT_OUTDIR, help="Directory for metrics/model outputs.")
    p.add_argument(
        "--feature-columns",
        default="Degree,Clustering,Density,Neigh Density,I_eig1,I_eig2,I_eig3",
        help="Comma-separated node feature columns from node_features parquet.",
    )
    p.add_argument(
        "--target-mode",
        choices=("raw", "transformed"),
        default="raw",
        help="Train target parameterization: raw λ or transformed (v1, Δλ2, Δλ3).",
    )
    p.add_argument("--apply-y1y5-filter", action="store_true", default=True)
    p.add_argument("--no-apply-y1y5-filter", dest="apply_y1y5_filter", action="store_false")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--hidden-layers", default="512,512,256,256", help="Comma-separated hidden layer sizes.")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=8, help="DataLoader workers.")
    p.add_argument("--mixed-precision", action="store_true", help="Enable AMP training (CUDA only).")
    p.add_argument("--multi-gpu", action="store_true", help="Use torch.nn.DataParallel across visible GPUs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gnn_meta_path = Path(args.gnn_metadata_path).expanduser().resolve()
    with gnn_meta_path.open("r", encoding="utf-8") as f:
        gnn_meta = json.load(f)

    node_parquet = Path(gnn_meta["outputs"]["node_features"]).expanduser().resolve()
    input_meta_path = Path(gnn_meta["input_metadata_path"]).expanduser().resolve()
    with input_meta_path.open("r", encoding="utf-8") as f:
        input_meta = json.load(f)
    source_catalog = (
        Path(args.catalog_path).expanduser().resolve()
        if args.catalog_path
        else Path(input_meta["source_path"]).expanduser().resolve()
    )

    feat_cols = [c.strip() for c in args.feature_columns.split(",") if c.strip()]
    print(f"Loading features from {node_parquet}")
    node_df = pd.read_parquet(node_parquet, columns=feat_cols)
    x = node_df.to_numpy(dtype=np.float32)

    print(f"Loading targets from {source_catalog}")
    tab = fitsio.read(str(source_catalog))
    mask = np.ones(len(tab), dtype=bool)
    if args.apply_y1y5_filter:
        mask &= _apply_optional_y1y5_filter(tab)

    l1_col = _resolve_col(tab, ("LAMBDA1", "L1", "EIG1", "LAM1", "LAMBDA_1"))
    l2_col = _resolve_col(tab, ("LAMBDA2", "L2", "EIG2", "LAM2", "LAMBDA_2"))
    l3_col = _resolve_col(tab, ("LAMBDA3", "L3", "EIG3", "LAM3", "LAMBDA_3"))
    y_raw = np.stack([tab[l1_col][mask], tab[l2_col][mask], tab[l3_col][mask]], axis=-1).astype(np.float32)

    if x.shape[0] != y_raw.shape[0]:
        raise ValueError(
            f"Row mismatch: features={x.shape[0]:,}, targets={y_raw.shape[0]:,}. "
            "This indicates feature/target construction misalignment."
        )

    if args.target_mode == "transformed":
        y = np.asarray(eigenvalues_to_increments(y_raw), dtype=np.float32)
    else:
        y = y_raw.copy()

    n = x.shape[0]
    idx = np.arange(n)
    train_frac = 1.0 - args.val_frac - args.test_frac
    if train_frac <= 0:
        raise ValueError("val-frac + test-frac must be < 1.0")

    train_idx, rem_idx = train_test_split(idx, test_size=(1.0 - train_frac), random_state=args.seed, shuffle=True)
    val_rel = args.val_frac / (args.val_frac + args.test_frac)
    val_idx, test_idx = train_test_split(rem_idx, train_size=val_rel, random_state=args.seed, shuffle=True)

    # Train-only scaling.
    x_mu = x[train_idx].mean(axis=0, keepdims=True)
    x_sd = x[train_idx].std(axis=0, keepdims=True) + 1e-8
    y_mu = y[train_idx].mean(axis=0, keepdims=True)
    y_sd = y[train_idx].std(axis=0, keepdims=True) + 1e-8
    x_s = (x - x_mu) / x_sd
    y_s = (y - y_mu) / y_sd

    def _make_loader(indices: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.from_numpy(x_s[indices]).float(),
            torch.from_numpy(y_s[indices]).float(),
        )
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    train_loader = _make_loader(train_idx, shuffle=True)
    val_loader = _make_loader(val_idx, shuffle=False)
    test_loader = _make_loader(test_idx, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden = [int(x.strip()) for x in args.hidden_layers.split(",") if x.strip()]
    model = TabularMLP(in_dim=x.shape[1], hidden=hidden, out_dim=3, dropout=args.dropout)
    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision and device.type == "cuda"))

    best_val = float("inf")
    best_state = None
    history: list[dict[str, float]] = []

    def _eval(loader: DataLoader) -> tuple[float, float]:
        model.eval()
        losses = []
        ys_true = []
        ys_pred = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                losses.append(float(loss.detach().cpu()))
                ys_true.append(yb.detach().cpu().numpy())
                ys_pred.append(pred.detach().cpu().numpy())
        y_true = np.concatenate(ys_true, axis=0)
        y_pred = np.concatenate(ys_pred, axis=0)
        mae = float(np.mean(np.abs(y_true - y_pred)))
        return float(np.mean(losses)), mae

    for epoch in range(args.epochs):
        model.train()
        tr_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(args.mixed_precision and device.type == "cuda")):
                pred = model(xb)
                loss = loss_fn(pred, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tr_losses.append(float(loss.detach().cpu()))

        train_mse = float(np.mean(tr_losses))
        val_mse, val_mae = _eval(val_loader)
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse, "val_mae": val_mae})
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        if epoch % max(1, args.epochs // 20) == 0 or epoch == args.epochs - 1:
            print(f"epoch={epoch:04d} train_mse={train_mse:.6f} val_mse={val_mse:.6f} val_mae={val_mae:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test metrics in original target units.
    model.eval()
    all_true = []
    all_pred = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device, non_blocking=True)
            pred = model(xb).detach().cpu().numpy()
            all_pred.append(pred)
            all_true.append(yb.numpy())

    y_true_s = np.concatenate(all_true, axis=0)
    y_pred_s = np.concatenate(all_pred, axis=0)
    y_true = y_true_s * y_sd + y_mu
    y_pred = y_pred_s * y_sd + y_mu

    metrics = {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2_mean": float(r2_score(y_true, y_pred, multioutput="uniform_average")),
        "r2_per_dim": [float(v) for v in r2_score(y_true, y_pred, multioutput="raw_values")],
        "n_total": int(n),
        "n_train": int(train_idx.size),
        "n_val": int(val_idx.size),
        "n_test": int(test_idx.size),
        "target_mode": args.target_mode,
        "feature_columns": feat_cols,
        "node_parquet": str(node_parquet),
        "source_catalog": str(source_catalog),
    }

    report_path = out_dir / f"abacus_tabular_mlp_report_seed{args.seed}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "metrics": metrics, "history_tail": history[-20:]}, f, indent=2)

    model_path = out_dir / f"abacus_tabular_mlp_model_seed{args.seed}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "x_mu": x_mu,
            "x_sd": x_sd,
            "y_mu": y_mu,
            "y_sd": y_sd,
            "feature_columns": feat_cols,
            "target_mode": args.target_mode,
            "hidden_layers": hidden,
        },
        model_path,
    )

    print("=" * 72)
    print("Abacus Tabular MLP done")
    print("=" * 72)
    print(f"Dataset rows: total={n:,}, train={train_idx.size:,}, val={val_idx.size:,}, test={test_idx.size:,}")
    print(f"Target mode: {args.target_mode}")
    print(f"Test MSE={metrics['mse']:.6f} MAE={metrics['mae']:.6f} R2_mean={metrics['r2_mean']:.6f}")
    print(f"R2 per dim: {metrics['r2_per_dim']}")
    print(f"Model: {model_path}")
    print(f"Report: {report_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()

