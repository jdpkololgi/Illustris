#!/usr/bin/env python3
"""Plot frozen P10 Arm-A ph006 predictions against authoritative truth.

This script deliberately supports only the visible validation phase, ``ph006``.
It validates the P10 parent-row contract before plotting and never discovers or
opens the sealed ph001 truth product.

Outputs include pooled predicted-versus-true and residual diagnostics for all
three ordered eigenvalues, plus redshift-shell facets with shared axis limits.
The fitted lines are descriptive fits on ph006 itself; they are not applied as
calibrations and must not be interpreted as test-set corrections.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from sklearn.metrics import r2_score


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.plot_style import TEXT_COLOR, apply_style


DEFAULT_CONTRACT = Path(
    "/pscratch/sd/d/dkololgi/abacus/p10_multiphase/training_contract/"
    "phases/ph006/phase_contract.json"
)
DEFAULT_RUN_ROOT = Path(
    "/pscratch/sd/d/dkololgi/abacus/p10_multiphase/arm_a_training/"
    "arm_a_r0_v1"
)
DEFAULT_OUTPUT = REPO_ROOT / "docs/figures/p10_multiphase_review_20260818"
MODELS = ("unet", "graph")
MODEL_LABELS = {"unet": "U-PATCH", "graph": "G-PATCH"}
SHELL_NAMES = ("0p15_0p25", "0p25_0p35", "0p35_0p45", "0p45_0p55")
SHELL_LABELS = ("0.15 < z < 0.25", "0.25 < z < 0.35", "0.35 < z < 0.45", "0.45 < z < 0.55")
LAMBDA_LABELS = (r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$")


def sha256(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bins", type=int, default=180)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def authoritative_mask(assignment) -> np.ndarray:
    return np.asarray(assignment["supervised_eligible"], dtype=bool)


def load_validation_contract(contract_path: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    contract = load_json(contract_path)
    if contract.get("phase") != "ph006" or contract.get("role") != "validation_and_selection":
        raise RuntimeError("plotting contract must be the visible ph006 validation contract")
    target_path = Path(contract["target"]["path"])
    assignment_path = Path(contract["inputs"]["assignment"])
    if "ph001" in str(target_path) or "ph001" in str(assignment_path):
        raise RuntimeError("sealed ph001 path is forbidden")
    if sha256(target_path) != contract["target"]["sha256"]:
        raise RuntimeError("ph006 target checksum differs from the frozen contract")
    if sha256(assignment_path) != contract["inputs"]["assignment_sha256"]:
        raise RuntimeError("ph006 assignment checksum differs from the frozen contract")

    truth_by_parent = np.load(target_path, mmap_mode="r")
    assignment = np.load(assignment_path, mmap_mode="r")
    rows = np.flatnonzero(authoritative_mask(assignment))
    parent = np.asarray(assignment["parent_node_id"][rows], dtype=np.int64)
    shell = np.asarray(assignment["shell"][rows], dtype=np.int8)
    if len(parent) != int(contract["authoritative_rows"]):
        raise RuntimeError("authoritative-row count differs from phase contract")
    if len(np.unique(parent)) != len(parent):
        raise RuntimeError("authoritative ph006 parent IDs are not unique")
    truth = np.asarray(truth_by_parent[parent], dtype=np.float64)
    if truth.shape != (len(parent), 3) or not np.all(np.isfinite(truth)):
        raise RuntimeError("invalid authoritative ph006 truth array")
    if np.any(truth[:, 1] < truth[:, 0]) or np.any(truth[:, 2] < truth[:, 1]):
        raise RuntimeError("ph006 truth violates the ordered-eigenvalue contract")
    assignment.close()
    return contract, parent, shell, truth


def align_model(run_root: Path, model: str, seed: int, required_parent: np.ndarray, truth: np.ndarray) -> dict:
    root = run_root / model / f"seed_{seed}"
    parent_path = root / "best_validation_parent_node_id.npy"
    prediction_path = root / "best_validation_eigenvalues.npy"
    report_path = root / "best_validation_report.json"
    parent = np.asarray(np.load(parent_path, mmap_mode="r"), dtype=np.int64)
    prediction = np.asarray(np.load(prediction_path, mmap_mode="r"), dtype=np.float64)
    report = load_json(report_path)

    if report.get("validation_phase") != "ph006" or not report.get("complete_phase_coverage"):
        raise RuntimeError(f"{model} artifact is not a complete ph006 evaluation")
    if len(parent) != len(required_parent) or len(np.unique(parent)) != len(parent):
        raise RuntimeError(f"{model} parent-row coverage or uniqueness failed")
    order = np.argsort(parent)
    sorted_parent = parent[order]
    lookup = np.searchsorted(sorted_parent, required_parent)
    if np.any(lookup == len(sorted_parent)) or not np.array_equal(sorted_parent[lookup], required_parent):
        raise RuntimeError(f"{model} parent IDs do not match authoritative ph006 rows")
    aligned = prediction[order][lookup]
    if aligned.shape != truth.shape or not np.all(np.isfinite(aligned)):
        raise RuntimeError(f"{model} prediction shape/finite check failed")
    if np.any(aligned[:, 1] < aligned[:, 0]) or np.any(aligned[:, 2] < aligned[:, 1]):
        raise RuntimeError(f"{model} predictions violate ordered eigenvalues")

    # Independent array-level reproduction of the stored headline metrics.
    measured_r2 = [float(r2_score(truth[:, i], aligned[:, i])) for i in range(3)]
    reported_r2 = [float(report["pooled"][f"lambda{i + 1}"]["r2"]) for i in range(3)]
    if not np.allclose(measured_r2, reported_r2, rtol=0.0, atol=2e-10):
        raise RuntimeError(f"{model} stored R2 does not reproduce from aligned rows")
    return {
        "root": root,
        "prediction": aligned,
        "report": report,
        "parent_path": parent_path,
        "prediction_path": prediction_path,
        "report_path": report_path,
    }


def robust_limits(truth: np.ndarray, predictions: dict[str, dict]) -> list[tuple[float, float]]:
    limits = []
    for index in range(3):
        candidates = [truth[:, index]] + [predictions[m]["prediction"][:, index] for m in MODELS]
        low = min(float(np.quantile(values, 0.001)) for values in candidates)
        high = max(float(np.quantile(values, 0.999)) for values in candidates)
        pad = 0.04 * max(high - low, 1e-6)
        limits.append((low - pad, high + pad))
    return limits


def histogram_image(ax, x: np.ndarray, y: np.ndarray, limits: tuple[float, float], bins: int):
    hist, xedges, yedges = np.histogram2d(
        x, y, bins=bins, range=(limits, limits)
    )
    shown = np.ma.masked_where(hist.T <= 0, hist.T)
    return ax.imshow(
        shown,
        origin="lower",
        extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]),
        aspect="equal",
        cmap="magma",
        norm=LogNorm(vmin=1, vmax=max(float(hist.max()), 2.0)),
        interpolation="nearest",
    )


def residual_image(
    ax,
    truth: np.ndarray,
    residual: np.ndarray,
    xlimits: tuple[float, float],
    ylimit: float,
    bins: int,
):
    hist, xedges, yedges = np.histogram2d(
        truth,
        residual,
        bins=bins,
        range=(xlimits, (-ylimit, ylimit)),
    )
    shown = np.ma.masked_where(hist.T <= 0, hist.T)
    image = ax.imshow(
        shown,
        origin="lower",
        extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]),
        aspect="auto",
        cmap="magma",
        norm=LogNorm(vmin=1, vmax=max(float(hist.max()), 2.0)),
        interpolation="nearest",
    )
    ax.axhline(0.0, color=TEXT_COLOR, lw=1.0, ls="--", alpha=0.8)
    centers = 0.5 * (xedges[:-1] + xedges[1:])
    # The density image uses every authoritative row.  Limit only the visual
    # binned-median overlay so it does not scan 4.9 million rows once per bin.
    stride = max(1, len(truth) // 250_000)
    median_truth = truth[::stride]
    median_residual = residual[::stride]
    digit = np.digitize(median_truth, xedges) - 1
    median = np.full(len(centers), np.nan)
    for bin_id in range(len(centers)):
        selected = digit == bin_id
        if selected.sum() >= 25:
            median[bin_id] = np.median(median_residual[selected])
    ax.plot(centers, median, color="#4E84F7", lw=1.6, label="binned median")
    return image


def pooled_figure(
    output_dir: Path,
    model: str,
    truth: np.ndarray,
    prediction: np.ndarray,
    report: dict,
    limits: list[tuple[float, float]],
    bins: int,
    dpi: int,
) -> list[Path]:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.4), constrained_layout=True)
    for index in range(3):
        x = truth[:, index]
        y = prediction[:, index]
        residual = y - x
        lo, hi = limits[index]
        slope, intercept = np.polyfit(x, y, 1)
        stored = report["pooled"][f"lambda{index + 1}"]
        rho = float(stored["spearman"])
        r2 = float(stored["r2"])
        top = axes[0, index]
        histogram_image(top, x, y, (lo, hi), bins)
        top.plot((lo, hi), (lo, hi), color=TEXT_COLOR, lw=1.2, ls="--", label="1:1")
        top.plot((lo, hi), (slope * lo + intercept, slope * hi + intercept),
                 color="#3A86FF", lw=1.5, label="ph006 descriptive fit")
        top.set_xlim(lo, hi)
        top.set_ylim(lo, hi)
        top.set_xlabel(f"True {LAMBDA_LABELS[index]}")
        top.set_ylabel(f"Predicted {LAMBDA_LABELS[index]}")
        top.set_title(LAMBDA_LABELS[index])
        top.text(
            0.03,
            0.97,
            f"$R^2$ = {r2:.3f}\n$\\rho_s$ = {rho:.3f}\nfit: $y={slope:.3f}x{intercept:+.3f}$",
            transform=top.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.72},
        )
        top.legend(loc="lower right", fontsize=8)

        bottom = axes[1, index]
        residual_limit = float(np.quantile(np.abs(residual), 0.995))
        residual_image(bottom, x, residual, (lo, hi), residual_limit, bins)
        bottom.set_xlim(lo, hi)
        bottom.set_ylim(-residual_limit, residual_limit)
        bottom.set_xlabel(f"True {LAMBDA_LABELS[index]}")
        bottom.set_ylabel(f"Predicted $-$ true {LAMBDA_LABELS[index]}")
        bottom.legend(loc="lower left", fontsize=8)

    epoch = report["runtime"]["epoch"]
    fig.suptitle(
        f"{MODEL_LABELS[model]} on independent ph006 (best epoch {epoch}; N={len(truth):,})\n"
        "Fits are descriptive on validation data only; no affine correction is applied",
        fontsize=18,
        fontweight="bold",
    )
    paths = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"predicted_vs_true_{model}_pooled.{suffix}"
        fig.savefig(path, dpi=dpi if suffix == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def shell_figure(
    output_dir: Path,
    model: str,
    shell: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    report: dict,
    limits: list[tuple[float, float]],
    bins: int,
    dpi: int,
) -> list[Path]:
    fig, axes = plt.subplots(4, 3, figsize=(14.5, 17), constrained_layout=True)
    for shell_id, shell_name in enumerate(SHELL_NAMES):
        selected = shell == shell_id
        for index in range(3):
            ax = axes[shell_id, index]
            x = truth[selected, index]
            y = prediction[selected, index]
            lo, hi = limits[index]
            histogram_image(ax, x, y, (lo, hi), bins)
            slope, intercept = np.polyfit(x, y, 1)
            stored = report["per_shell"][shell_name][f"lambda{index + 1}"]
            ax.plot((lo, hi), (lo, hi), color=TEXT_COLOR, lw=1.0, ls="--")
            ax.plot((lo, hi), (slope * lo + intercept, slope * hi + intercept),
                    color="#3A86FF", lw=1.25)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.text(
                0.03,
                0.97,
                f"$R^2$={stored['r2']:.3f}  $\\rho_s$={stored['spearman']:.3f}\n"
                f"slope={slope:.3f}  N={selected.sum():,}",
                transform=ax.transAxes,
                va="top",
                fontsize=9,
                bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.72},
            )
            if shell_id == 0:
                ax.set_title(LAMBDA_LABELS[index])
            if shell_id == 3:
                ax.set_xlabel(f"True {LAMBDA_LABELS[index]}")
            if index == 0:
                ax.set_ylabel(f"{SHELL_LABELS[shell_id]}\nPredicted")

    fig.suptitle(
        f"{MODEL_LABELS[model]} predicted versus true eigenvalues by ph006 shell\n"
        "Each eigenvalue uses shared axes across all shells and both models",
        fontsize=18,
        fontweight="bold",
    )
    paths = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"predicted_vs_true_{model}_shell_facets.{suffix}"
        fig.savefig(path, dpi=dpi if suffix == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def main() -> None:
    args = parse_args()
    apply_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract, required_parent, shell, truth = load_validation_contract(args.contract)
    models = {
        model: align_model(args.run_root, model, args.seed, required_parent, truth)
        for model in MODELS
    }
    limits = robust_limits(truth, models)
    outputs: list[Path] = []
    for model in MODELS:
        row = models[model]
        outputs.extend(
            pooled_figure(
                args.output_dir,
                model,
                truth,
                row["prediction"],
                row["report"],
                limits,
                args.bins,
                args.dpi,
            )
        )
        outputs.extend(
            shell_figure(
                args.output_dir,
                model,
                shell,
                truth,
                row["prediction"],
                row["report"],
                limits,
                args.bins,
                args.dpi,
            )
        )

    summary = {
        "schema_version": "p10-ph006-predicted-vs-true-plots-v1",
        "validation_phase": "ph006",
        "blind_phase_opened": False,
        "contract_path": str(args.contract),
        "contract_sha256": sha256(args.contract),
        "assignment_sha256": contract["inputs"]["assignment_sha256"],
        "target_sha256": contract["target"]["sha256"],
        "n_authoritative": int(len(required_parent)),
        "shared_limits": {
            f"lambda{index + 1}": [float(value) for value in limits[index]]
            for index in range(3)
        },
        "models": {
            model: {
                "label": MODEL_LABELS[model],
                "best_epoch": int(models[model]["report"]["runtime"]["epoch"]),
                "macro_r2_lambda1": float(models[model]["report"]["primary_macro_r2_lambda1"]),
                "pooled": models[model]["report"]["pooled"],
                "parent_ids": str(models[model]["parent_path"]),
                "predictions": str(models[model]["prediction_path"]),
                "report": str(models[model]["report_path"]),
                "parent_alignment_complete": True,
            }
            for model in MODELS
        },
        "outputs": [str(path) for path in outputs],
        "note": "All fitted lines are descriptive ph006 validation fits; predictions were not recalibrated.",
    }
    summary_path = args.output_dir / "predicted_vs_true_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "outputs": [str(p) for p in outputs]}, indent=2))


if __name__ == "__main__":
    main()
