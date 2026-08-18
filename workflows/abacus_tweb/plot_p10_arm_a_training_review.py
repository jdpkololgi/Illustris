#!/usr/bin/env python3
"""Plot the P10 Arm-A learning curves against the P8 single-phase controls.

This is a read-only analysis utility.  It deliberately reads only the P10
training phases, the held-out ph006 validation products, and the historical P8
same-phase development runs.  It never opens ph001 products.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


P10_ROOT = Path(
    "/pscratch/sd/d/dkololgi/abacus/p10_multiphase/arm_a_training/"
    "arm_a_r0_v1"
)
P8_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p8_recovery_v1")
DEFAULT_OUTPUT = Path("docs/figures/p10_multiphase_review_20260818")

MODELS = ("unet", "graph")
MODEL_LABEL = {"unet": "U-PATCH", "graph": "G-PATCH"}
MODEL_COLOR = {"unet": "#0072B2", "graph": "#D55E00"}
SHELLS = ("0p15_0p25", "0p25_0p35", "0p35_0p45", "0p45_0p55")
SHELL_LABEL = {
    "0p15_0p25": r"$0.15<z<0.25$",
    "0p25_0p35": r"$0.25<z<0.35$",
    "0p35_0p45": r"$0.35<z<0.45$",
    "0p45_0p55": r"$0.45<z<0.55$",
}
SHELL_COLOR = {
    "0p15_0p25": "#0072B2",
    "0p25_0p35": "#009E73",
    "0p35_0p45": "#E69F00",
    "0p45_0p55": "#CC79A7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def p10_history(model: str) -> list[dict[str, Any]]:
    return read_jsonl(P10_ROOT / model / "seed_42" / "epoch_history.jsonl")


def p8_history(model: str, rotation: int) -> list[dict[str, Any]]:
    base = read_jsonl(
        P8_ROOT
        / "recovery_v1"
        / model
        / f"rotation_{rotation}"
        / "seed_42"
        / "epoch_history.jsonl"
    )
    extension = read_jsonl(
        P8_ROOT
        / "convergence_extension_v1"
        / model
        / f"rotation_{rotation}"
        / "seed_42"
        / "epoch_history.jsonl"
    )
    step_offset = int(base[-1]["global_step"])
    epoch_offset = int(base[-1]["epoch"])
    merged: list[dict[str, Any]] = []
    for row in base:
        item = dict(row)
        item["comparison_update"] = int(row["global_step"])
        item["comparison_epoch"] = int(row["epoch"])
        merged.append(item)
    for row in extension:
        item = dict(row)
        item["comparison_update"] = step_offset + int(row["global_step"])
        item["comparison_epoch"] = int(row.get("effective_epoch", epoch_offset + int(row["epoch"])))
        merged.append(item)
    return merged


def values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([row[key] for row in rows], dtype=float)


def nested_values(rows: list[dict[str, Any]], outer: str, inner: str) -> np.ndarray:
    return np.asarray([row[outer][inner] for row in rows], dtype=float)


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "figure.titlesize": 13,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "lines.linewidth": 2.0,
            "savefig.bbox": "tight",
        }
    )


def save_both(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi)
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def plot_p10_overview(
    histories: dict[str, list[dict[str, Any]]], output_dir: Path, dpi: int
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.4), constrained_layout=True)

    ax = axes[0, 0]
    for model in MODELS:
        rows = histories[model]
        ax.plot(
            values(rows, "epoch"),
            values(rows, "equal_phase_optimizer_objective_mean"),
            marker="o",
            ms=4,
            color=MODEL_COLOR[model],
            label=MODEL_LABEL[model],
        )
    ax.set(title="Training objective", xlabel="P10 epoch", ylabel="Equal-phase weighted MSE")
    ax.legend()

    ax = axes[0, 1]
    for model in MODELS:
        rows = histories[model]
        ax.plot(
            values(rows, "epoch"),
            values(rows, "primary_macro_r2_lambda1"),
            marker="o",
            ms=4,
            color=MODEL_COLOR[model],
            label=MODEL_LABEL[model],
        )
    ax.axhline(0.520, color="0.35", ls="--", lw=1.4, label="P8 CIC first-three reference")
    ax.set(title="Held-out ph006 performance", xlabel="P10 epoch", ylabel=r"Macro $R^2(\lambda_1)$")
    ax.legend()

    ax = axes[1, 0]
    for model in MODELS:
        rows = histories[model]
        ax.plot(
            values(rows, "epoch"),
            nested_values(rows, "validation", "all_rows_scaled_mse"),
            marker="o",
            ms=4,
            color=MODEL_COLOR[model],
            label=MODEL_LABEL[model],
        )
    ax.set(title="Held-out ph006 loss", xlabel="P10 epoch", ylabel="All-row scaled MSE")
    ax.legend()

    ax = axes[1, 1]
    for model in MODELS:
        rows = histories[model]
        ax.plot(
            values(rows, "global_step"),
            values(rows, "learning_rate"),
            marker="o",
            ms=4,
            color=MODEL_COLOR[model],
            label=MODEL_LABEL[model],
        )
    ax.set(title="Registered cosine schedule", xlabel="Optimizer updates", ylabel="Learning rate")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
    ax.legend()

    fig.suptitle("P10 Arm-A multi-phase training and ph006 validation")
    save_both(fig, output_dir, "training_curves_p10_overview", dpi)


def plot_p10_shells(
    histories: dict[str, list[dict[str, Any]]], output_dir: Path, dpi: int
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True, constrained_layout=True)
    for ax, model in zip(axes, MODELS, strict=True):
        rows = histories[model]
        epochs = values(rows, "epoch")
        for shell in SHELLS:
            ax.plot(
                epochs,
                nested_values(rows, "per_shell_lambda1_r2", shell),
                marker="o",
                ms=3.5,
                color=SHELL_COLOR[shell],
                label=SHELL_LABEL[shell],
            )
        ax.plot(
            epochs,
            values(rows, "primary_macro_r2_lambda1"),
            color="black",
            lw=2.5,
            label="Four-shell macro",
        )
        ax.set(title=MODEL_LABEL[model], xlabel="P10 epoch")
    axes[0].set_ylabel(r"Held-out ph006 $R^2(\lambda_1)$")
    axes[1].legend(loc="lower right")
    fig.suptitle("P10 per-shell generalisation reveals the information-density gradient")
    save_both(fig, output_dir, "training_curves_p10_per_shell", dpi)


def plot_p8_p10_comparison(
    p10: dict[str, list[dict[str, Any]]],
    p8: dict[str, dict[int, list[dict[str, Any]]]],
    output_dir: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.4), constrained_layout=True)

    for col, model in enumerate(MODELS):
        ax = axes[0, col]
        for rotation, rows in p8[model].items():
            ax.plot(
                values(rows, "comparison_epoch"),
                values(rows, "primary_macro_r2_lambda1"),
                color="0.45",
                alpha=0.78,
                ls="--" if rotation == 0 else ":",
                label=f"P8 ph000 rotation {rotation}",
            )
        ax.plot(
            values(p10[model], "epoch"),
            values(p10[model], "primary_macro_r2_lambda1"),
            color=MODEL_COLOR[model],
            marker="o",
            ms=3.8,
            label="P10 ph006",
        )
        ax.set(title=f"{MODEL_LABEL[model]}: nominal epoch", xlabel="Epoch", ylabel=r"Macro $R^2(\lambda_1)$")
        ax.legend()

        ax = axes[1, col]
        for rotation, rows in p8[model].items():
            ax.plot(
                values(rows, "comparison_update"),
                values(rows, "primary_macro_r2_lambda1"),
                color="0.45",
                alpha=0.78,
                ls="--" if rotation == 0 else ":",
                label=f"P8 ph000 rotation {rotation}",
            )
        ax.plot(
            values(p10[model], "global_step"),
            values(p10[model], "primary_macro_r2_lambda1"),
            color=MODEL_COLOR[model],
            marker="o",
            ms=3.8,
            label="P10 ph006",
        )
        ax.set(title=f"{MODEL_LABEL[model]}: optimizer exposure", xlabel="Optimizer updates", ylabel=r"Macro $R^2(\lambda_1)$")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(5, 5))
        ax.legend()

    fig.suptitle(
        "P8 same-phase and P10 fresh-phase curves (validation targets differ; compare trends, not identical estimands)"
    )
    save_both(fig, output_dir, "training_curves_p8_p10_comparison", dpi)


def trailing_gain(rows: list[dict[str, Any]], field: str, window: int = 3) -> float | None:
    if len(rows) <= window:
        return None
    return float(rows[-1][field] - rows[-1 - window][field])


def write_metrics(
    p10: dict[str, list[dict[str, Any]]],
    p8: dict[str, dict[int, list[dict[str, Any]]]],
    output_dir: Path,
) -> None:
    summary: dict[str, Any] = {
        "scope": {
            "p10_train_phases": ["ph000", "ph002", "ph003", "ph004", "ph005"],
            "p10_validation_phase": "ph006",
            "blind_phase_accessed": False,
            "warning": "P8 and P10 validation targets differ; the comparison is diagnostic, not paired.",
        },
        "p10": {},
        "p8": {},
    }
    for model in MODELS:
        rows = p10[model]
        metric = values(rows, "primary_macro_r2_lambda1")
        best_index = int(np.argmax(metric))
        best = rows[best_index]
        latest = rows[-1]
        summary["p10"][model] = {
            "completed_epochs": len(rows),
            "latest_epoch": int(latest["epoch"]),
            "latest_updates": int(latest["global_step"]),
            "best_epoch": int(best["epoch"]),
            "best_updates": int(best["global_step"]),
            "best_macro_r2_lambda1": float(best["primary_macro_r2_lambda1"]),
            "latest_macro_r2_lambda1": float(latest["primary_macro_r2_lambda1"]),
            "latest_equal_phase_objective": float(latest["equal_phase_optimizer_objective_mean"]),
            "latest_validation_scaled_mse": float(latest["validation"]["all_rows_scaled_mse"]),
            "latest_learning_rate": float(latest["learning_rate"]),
            "three_epoch_macro_gain": trailing_gain(rows, "primary_macro_r2_lambda1", 3),
            "three_epoch_objective_change": trailing_gain(rows, "equal_phase_optimizer_objective_mean", 3),
            "latest_per_shell_lambda1_r2": latest["per_shell_lambda1_r2"],
        }
        summary["p8"][model] = {}
        for rotation, old_rows in p8[model].items():
            old_metric = values(old_rows, "primary_macro_r2_lambda1")
            old_best = old_rows[int(np.argmax(old_metric))]
            summary["p8"][model][f"rotation_{rotation}"] = {
                "latest_effective_epoch": int(old_rows[-1]["comparison_epoch"]),
                "latest_total_updates": int(old_rows[-1]["comparison_update"]),
                "best_macro_r2_lambda1": float(old_best["primary_macro_r2_lambda1"]),
                "best_total_updates": int(old_best["comparison_update"]),
            }

    path = output_dir / "training_curves_metrics.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()

    p10 = {model: p10_history(model) for model in MODELS}
    p8 = {
        model: {rotation: p8_history(model, rotation) for rotation in (0, 2)}
        for model in MODELS
    }

    plot_p10_overview(p10, args.output_dir, args.dpi)
    plot_p10_shells(p10, args.output_dir, args.dpi)
    plot_p8_p10_comparison(p10, p8, args.output_dir, args.dpi)
    write_metrics(p10, p8, args.output_dir)

    for path in sorted(args.output_dir.glob("training_curves_*")):
        print(path)


if __name__ == "__main__":
    main()
