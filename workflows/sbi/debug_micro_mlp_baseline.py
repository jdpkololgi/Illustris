#!/usr/bin/env python3
"""Quick tabular baselines for micro overfit diagnostics.

Given a partition NPZ with `x` and `targets`, this script fits:
1) mean predictor
2) linear regression (least squares)
3) nonlinear MLP regressor (scikit-learn)

It reports train MSE/MAE so we can compare against GNN/SBI runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def _mse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return mse, mae


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--partition-npz",
        required=True,
        help="Path to partition .npz containing x and targets.",
    )
    p.add_argument(
        "--hidden-layers",
        default="256,256",
        help="Comma-separated hidden layer sizes for MLPRegressor.",
    )
    p.add_argument("--max-iter", type=int, default=2000, help="Max iterations for MLPRegressor.")
    p.add_argument("--alpha", type=float, default=1e-6, help="L2 regularization for MLPRegressor.")
    p.add_argument("--learning-rate-init", type=float, default=1e-3, help="Initial learning rate for MLPRegressor.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--output-json",
        default="",
        help="Optional output JSON report path. If empty, writes next to partition NPZ.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    part = Path(args.partition_npz).expanduser().resolve()
    if not part.exists():
        raise FileNotFoundError(f"Partition file not found: {part}")

    hidden_layers = tuple(int(x.strip()) for x in args.hidden_layers.split(",") if x.strip())
    if not hidden_layers:
        raise ValueError("hidden-layers must include at least one size.")

    with np.load(part) as d:
        x = np.asarray(d["x"], dtype=np.float64)
        y = np.asarray(d["targets"], dtype=np.float64)

    # Mean baseline
    y_mean = np.repeat(np.mean(y, axis=0, keepdims=True), repeats=y.shape[0], axis=0)
    mse_mean, mae_mean = _mse_mae(y, y_mean)

    # Linear regression baseline (closed-form least squares with intercept)
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    w = np.linalg.pinv(x_aug) @ y
    y_lin = x_aug @ w
    mse_lin, mae_lin = _mse_mae(y, y_lin)

    # Nonlinear MLP baseline with standardized inputs/targets
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_s = x_scaler.fit_transform(x)
    y_s = y_scaler.fit_transform(y)
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=float(args.alpha),
        learning_rate_init=float(args.learning_rate_init),
        max_iter=int(args.max_iter),
        random_state=int(args.seed),
    )
    mlp.fit(x_s, y_s)
    y_mlp = y_scaler.inverse_transform(mlp.predict(x_s))
    mse_mlp, mae_mlp = _mse_mae(y, y_mlp)

    payload = {
        "partition_npz": str(part),
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "hidden_layers": list(hidden_layers),
        "max_iter": int(args.max_iter),
        "alpha": float(args.alpha),
        "learning_rate_init": float(args.learning_rate_init),
        "seed": int(args.seed),
        "metrics": {
            "mean_predictor": {"mse": mse_mean, "mae": mae_mean},
            "linear_regression": {"mse": mse_lin, "mae": mae_lin},
            "mlp_regressor": {"mse": mse_mlp, "mae": mae_mlp},
        },
    }

    out = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else part.with_name(part.stem + "_mlp_baseline_report.json")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print("=" * 72)
    print("Micro Baseline Report")
    print("=" * 72)
    print(f"Partition: {part}")
    print(f"Samples/features: {x.shape[0]} / {x.shape[1]}")
    print(f"Mean baseline     MSE={mse_mean:.6f}  MAE={mae_mean:.6f}")
    print(f"Linear regression MSE={mse_lin:.6f}  MAE={mae_lin:.6f}")
    print(f"MLP regressor     MSE={mse_mlp:.6f}  MAE={mae_mlp:.6f}")
    print(f"Saved report: {out}")
    print("=" * 72)


if __name__ == "__main__":
    main()

