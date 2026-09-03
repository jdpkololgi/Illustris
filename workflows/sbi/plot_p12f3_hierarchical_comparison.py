#!/usr/bin/env python3
"""Render matched P12-F3 training, field, spectrum, and TARP diagnostics."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.sbi.p12f_common_evaluator import load_core_record
from workflows.sbi.p12f_dependency_rescue_evaluator import (
    tarp_curve,
    tidal_eigenvalues_at_galaxies,
)


METHODS = (
    "g1_wide_crop_h8",
    "g1_wide_h24",
    "hybrid_local_h8",
    "hybrid_wide_h24",
)
LABELS = {
    "g1_wide_crop_h8": "G1 wide → local FFT",
    "g1_wide_h24": "G1 wide",
    "hybrid_local_h8": "hybrid: local low-mode condition",
    "hybrid_wide_h24": "hybrid: wide low-mode condition",
}
COLORS = {
    "g1_wide_crop_h8": "#777777",
    "g1_wide_h24": "#111111",
    "hybrid_local_h8": "#2878b5",
    "hybrid_wide_h24": "#d1495b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument(
        "--method-evaluation-root",
        action="append",
        default=[],
        metavar="METHOD=PATH",
        help="Override the evaluation root for one method (repeatable).",
    )
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_trace(path: Path) -> dict[str, np.ndarray]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {
        "update": np.asarray([row["update"] for row in rows]),
        "loss": np.asarray([row["loss"] for row in rows]),
        "mean": np.asarray([row["mean_loss"] for row in rows]),
    }


def moving_average(values: np.ndarray, width: int = 9) -> np.ndarray:
    if len(values) < width:
        return values
    half = width // 2
    padded = np.pad(values, (half, width - 1 - half), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def spectral_accumulate(
    samples: np.ndarray,
    truth: np.ndarray,
    support: np.ndarray,
    core_bounds: np.ndarray,
    edges: np.ndarray,
    voxel_mpc_h: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    core = tuple(slice(int(a), int(b)) for a, b in zip(core_bounds[0], core_bounds[1]))
    sample = samples[(slice(None),) + core]
    target = truth[core]
    mask = support[core].astype(np.float32)
    mean = sample.mean(axis=0)
    innovation = (target - mean) * mask
    residual = (sample - mean[None]) * mask[None]
    truth_k = np.fft.rfftn(innovation, axes=(-3, -2, -1), norm="ortho")
    draw_k = np.fft.rfftn(residual, axes=(-3, -2, -1), norm="ortho")
    shape = target.shape
    kx = 2 * np.pi * np.fft.fftfreq(shape[0], d=voxel_mpc_h)[:, None, None]
    ky = 2 * np.pi * np.fft.fftfreq(shape[1], d=voxel_mpc_h)[None, :, None]
    kz = 2 * np.pi * np.fft.rfftfreq(shape[2], d=voxel_mpc_h)[None, None, :]
    radius = np.sqrt(kx * kx + ky * ky + kz * kz).ravel()
    label = np.searchsorted(edges[1:-1], radius, side="right")
    non_dc = radius > 0
    truth_power = np.square(np.abs(truth_k)).ravel()
    posterior_power = np.mean(np.square(np.abs(draw_k)), axis=0).ravel()
    return (
        np.bincount(label[non_dc], weights=truth_power[non_dc], minlength=len(edges) - 1),
        np.bincount(label[non_dc], weights=posterior_power[non_dc], minlength=len(edges) - 1),
        np.bincount(label[non_dc], minlength=len(edges) - 1),
    )


def analyze_archive(manifest_path: Path, *, device: str) -> tuple[dict, dict]:
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != "p12f-sample-archive-v1"
        or manifest.get("phase") != "ph006"
        or manifest.get("ph001_opened")
        or int(manifest.get("draws", -1)) != 64
    ):
        raise RuntimeError("P12-F3 visualization received an unsafe archive")
    lambda_draws, lambda_truth = [], []
    cutoff = 0.1813799364234218
    maximum = np.sqrt(3.0) * np.pi / 5.0
    # The first two bins exactly reproduce the two non-DC causal-autopsy bands.
    edges = np.concatenate(([0.0, cutoff / 2.0], np.linspace(cutoff, maximum, 17)))
    truth_power = np.zeros(len(edges) - 1)
    draw_power = np.zeros(len(edges) - 1)
    mode_count = np.zeros(len(edges) - 1, dtype=np.int64)
    example = None
    for ordinal, entry in enumerate(manifest["entries"]):
        if sha256(Path(entry["path"])) != entry["sha256"]:
            raise RuntimeError("P12-F3 visualization archive changed")
        record = load_core_record(entry, 64)
        samples = np.asarray(record["delta_samples"], dtype=np.float32)
        truth = np.asarray(record["delta_truth"], dtype=np.float32)
        coordinates = np.asarray(record["galaxy_frac_index_local"], dtype=np.float32)
        with torch.inference_mode():
            draw_lambda, _ = tidal_eigenvalues_at_galaxies(
                torch.from_numpy(samples).to(device), coordinates
            )
            true_lambda, _ = tidal_eigenvalues_at_galaxies(
                torch.from_numpy(truth).to(device), coordinates
            )
        if len(true_lambda):
            lambda_draws.append(draw_lambda)
            lambda_truth.append(true_lambda)
        tp, dp, count = spectral_accumulate(
            samples,
            truth,
            np.asarray(record["support"], dtype=bool),
            np.asarray(record["core_bounds"], dtype=np.int64),
            edges,
        )
        truth_power += tp
        draw_power += dp
        mode_count += count
        if example is None and ordinal >= len(manifest["entries"]) // 2:
            bounds = np.asarray(record["core_bounds"], dtype=np.int64)
            core = tuple(slice(int(a), int(b)) for a, b in zip(bounds[0], bounds[1]))
            sample_core = samples[(slice(None),) + core]
            truth_core = truth[core]
            example = {
                "core_id": int(entry["core_id"]),
                "truth": truth_core,
                "mean": sample_core.mean(axis=0),
                "std": sample_core.std(axis=0),
            }
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"visual physics {manifest['method']} {ordinal+1}/{len(manifest['entries'])}", flush=True)
    lam = np.concatenate(lambda_draws, axis=1)
    target = np.concatenate(lambda_truth, axis=0)
    gaps = lam[..., 1:] - lam[..., :-1]
    truth_gaps = target[..., 1:] - target[..., :-1]
    eigen = tarp_curve(lam, target, seed=42)
    gap = tarp_curve(gaps, truth_gaps, seed=43)
    ratio = np.divide(draw_power, truth_power, out=np.full_like(draw_power, np.nan), where=truth_power > 0)
    return {
        "method": manifest["method"],
        "cores": len(manifest["entries"]),
        "galaxies": len(target),
        "eigen_tarp": eigen,
        "gap_tarp": gap,
        "spectral_k_edges_h_mpc": edges.tolist(),
        "truth_innovation_power": np.divide(truth_power, mode_count, out=np.zeros_like(truth_power), where=mode_count > 0).tolist(),
        "posterior_residual_power": np.divide(draw_power, mode_count, out=np.zeros_like(draw_power), where=mode_count > 0).tolist(),
        "posterior_to_truth_power": ratio.tolist(),
    }, example


def plot_summary(summary: dict, traces: dict, reports: dict, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for arm, label, color in (
        ("local_h8", "local h8", COLORS["hybrid_local_h8"]),
        ("wide_h24", "wide h24", COLORS["hybrid_wide_h24"]),
    ):
        trace = traces[arm]
        axes[0, 0].plot(trace["update"], moving_average(trace["loss"]), color=color, alpha=.8, label=label)
    axes[0, 0].set(xlabel="optimizer update", ylabel="low-mode flow loss", title="A. Matched training curves")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=.2)
    for method in METHODS:
        row = summary[method]
        edges = np.asarray(row["spectral_k_edges_h_mpc"])
        center = .5 * (edges[:-1] + edges[1:])
        axes[0, 1].plot(center, row["posterior_to_truth_power"], color=COLORS[method], label=LABELS[method])
    axes[0, 1].axhline(1, color="black", linestyle="--", linewidth=1)
    axes[0, 1].axvspan(0, 0.1813799364, color="#f2c14e", alpha=.15, label="trained low-mode band")
    axes[0, 1].set(xlabel=r"$k\ [h\,\mathrm{Mpc}^{-1}]$", ylabel="posterior / truth residual power", title="B. Conditional residual power", ylim=(0, 2.5))
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=.2)
    for axis, key, title in (
        (axes[1, 0], "eigen_tarp", "C. Joint ordered-eigenvalue TARP"),
        (axes[1, 1], "gap_tarp", "D. Joint eigengap TARP"),
    ):
        axis.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, label="ideal")
        for method in METHODS:
            row = summary[method][key]
            axis.plot(row["alpha"], row["expected_coverage_probability"], color=COLORS[method], label=f"{LABELS[method]} ({row['maximum_deviation']:.3f})")
        axis.set(xlabel="nominal coverage", ylabel="empirical coverage", title=title, xlim=(0, 1), ylim=(0, 1))
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    figure.suptitle("P12-F3 hierarchical low-mode field posterior — held-out ph006", fontsize=15)
    figure.savefig(output.with_suffix(".png"), dpi=180)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def plot_examples(examples: dict, output: Path) -> None:
    figure, axes = plt.subplots(3, len(METHODS), figsize=(16, 10), constrained_layout=True)
    all_truth = np.concatenate([row["truth"].ravel() for row in examples.values()])
    limit = np.quantile(np.abs(all_truth), .99)
    for column, method in enumerate(METHODS):
        row = examples[method]
        z = row["truth"].shape[2] // 2
        for axis, values, label in zip(
            axes[:, column],
            (row["truth"][:, :, z], row["mean"][:, :, z], row["std"][:, :, z]),
            ("truth", "posterior mean", "posterior std"),
        ):
            if label == "posterior std":
                image = axis.imshow(values.T, origin="lower", cmap="magma", vmin=0)
            else:
                image = axis.imshow(values.T, origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit)
            axis.set(xticks=[], yticks=[])
            if column == 0:
                axis.set_ylabel(label)
            figure.colorbar(image, ax=axis, fraction=.046)
        axes[0, column].set_title(f"{LABELS[method]}\ncore {row['core_id']}")
    figure.suptitle("Authoritative-core density slices (same panel; method-specific patch physics)", fontsize=14)
    figure.savefig(output.with_suffix(".png"), dpi=180)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def plot_gate_metrics(summary: dict, reports: dict, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    x = np.arange(len(METHODS))
    labels = [LABELS[method] for method in METHODS]
    colors = [COLORS[method] for method in METHODS]
    eigen = [summary[method]["eigen_tarp"]["maximum_deviation"] for method in METHODS]
    gaps = [summary[method]["gap_tarp"]["maximum_deviation"] for method in METHODS]
    axes[0, 0].bar(x - .18, eigen, width=.36, color=colors, alpha=.55, label="eigenvalues")
    axes[0, 0].bar(x + .18, gaps, width=.36, color=colors, label="eigengaps")
    axes[0, 0].axhline(.05, color="black", linestyle="--", label="gate 0.05")
    axes[0, 0].set(ylabel="maximum |ECP − nominal|", title="A. Joint TARP deviation")
    axes[0, 0].legend(fontsize=8)
    global_error = [max(map(float, reports[m]["global_coverage_error"].values())) for m in METHODS]
    conditional = [float(reports[m]["maximum_conditional_coverage_error"]) for m in METHODS]
    axes[0, 1].bar(x - .18, global_error, width=.36, color=colors, alpha=.55, label="global")
    axes[0, 1].bar(x + .18, conditional, width=.36, color=colors, label="conditional max")
    axes[0, 1].axhline(.05, color="black", linestyle="--", linewidth=1)
    axes[0, 1].axhline(.10, color="black", linestyle=":", linewidth=1)
    axes[0, 1].set(ylabel="absolute coverage error", title="B. Coverage gates")
    axes[0, 1].legend(fontsize=8)
    energy = [float(reports[m]["proper_scores"]["energy"]) for m in METHODS]
    variogram = [float(reports[m]["proper_scores"]["variogram_p0p5"]) for m in METHODS]
    axes[1, 0].bar(x, energy, color=colors)
    axes[1, 0].set(ylabel="energy score (lower is better)", title="C. Joint field score")
    axes[1, 1].bar(x, variogram, color=colors)
    axes[1, 1].set(ylabel="variogram score (lower is better)", title="D. Spatial-dependence score")
    for axis in axes.ravel():
        axis.set_xticks(x, labels, rotation=20, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=.2)
    figure.suptitle("P12-F3 registered calibration and proper-score gates", fontsize=15)
    figure.savefig(output.with_suffix(".png"), dpi=180)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("P12-F3 physics visualization requires a compute GPU")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_roots = {method: args.evaluation_root for method in METHODS}
    for item in args.method_evaluation_root:
        try:
            method, raw_path = item.split("=", 1)
        except ValueError as error:
            raise RuntimeError("--method-evaluation-root must be METHOD=PATH") from error
        if method not in METHODS or not raw_path:
            raise RuntimeError(f"invalid method evaluation-root override: {item}")
        evaluation_roots[method] = Path(raw_path)
    summaries, examples, reports = {}, {}, {}
    for method in METHODS:
        root = evaluation_roots[method]
        manifest = root / "archives" / method / "P12F_SAMPLE_ARCHIVE.json"
        report_path = root / "reports" / f"{method}.json"
        summaries[method], examples[method] = analyze_archive(manifest, device=args.device)
        reports[method] = json.loads(report_path.read_text())
    traces = {
        arm: read_trace(args.training_root / arm / "loss_trace.jsonl")
        for arm in ("local_h8", "wide_h24")
    }
    plot_summary(summaries, traces, reports, args.output_dir / "p12f3_hierarchical_summary")
    plot_examples(examples, args.output_dir / "p12f3_field_examples")
    plot_gate_metrics(summaries, reports, args.output_dir / "p12f3_gate_metrics")
    compact = {
        "schema_version": "p12f3-hierarchical-visual-audit-v1",
        "created_utc": utc_now(),
        "phase": "ph006",
        "evaluation_roots": {
            method: str(evaluation_roots[method].resolve()) for method in METHODS
        },
        "methods": summaries,
        "common_evaluation": {
            method: {
                "tarp_maximum_deviation": reports[method]["tarp_maximum_deviation"],
                "global_coverage_error": reports[method]["global_coverage_error"],
                "maximum_conditional_coverage_error": reports[method]["maximum_conditional_coverage_error"],
                "proper_scores": reports[method]["proper_scores"],
            }
            for method in METHODS
        },
        "figures": [
            "p12f3_hierarchical_summary.png",
            "p12f3_field_examples.png",
            "p12f3_gate_metrics.png",
        ],
        "ph001_opened": False,
    }
    evidence = (
        args.output_dir / "P12F3_VISUAL_AUDIT.json"
        if args.evidence_output is None
        else args.evidence_output
    )
    evidence.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(evidence, compact)
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
