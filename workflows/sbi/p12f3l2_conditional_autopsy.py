#!/usr/bin/env python3
"""Visual, block-aware autopsy of the residual P12-F3-L2 coverage error.

This is a diagnostic, not a recalibration path.  It never writes adjusted
posterior samples and it rejects any phase other than ph006.  In particular,
the true-density quartiles used below are evaluation strata and are not
deployable conditioning features.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


METHODS = ("g1_wide_h24", "fourier_gaussian_h24", "fourier_flow_h24")
LABELS = {
    "g1_wide_h24": "G1 stationary residual",
    "fourier_gaussian_h24": "Fourier Gaussian",
    "fourier_flow_h24": "conditional Fourier flow",
}
COLORS = {
    "g1_wide_h24": "#555555",
    "fourier_gaussian_h24": "#2878b5",
    "fourier_flow_h24": "#d1495b",
}
LEVELS = (0.68, 0.90)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_core_record(entry: dict[str, Any], draws: int) -> dict[str, np.ndarray]:
    required = {
        "delta_samples", "delta_truth", "support", "angular_response",
        "boundary_distance_mpc", "tracer_density", "core_bounds",
        "galaxy_frac_index_local",
    }
    with np.load(entry["path"], allow_pickle=False) as values:
        if not required.issubset(values.files):
            raise RuntimeError(f"sample core is missing {sorted(required-set(values.files))}")
        record = {name: np.asarray(values[name]) for name in required}
    if record["delta_samples"].shape[0] != draws:
        raise RuntimeError("sample core draw count mismatch")
    return record


def parse_keyed(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        key, separator, path = value.partition("=")
        if not separator or key in result:
            raise ValueError("archives must be unique METHOD=PATH values")
        result[key] = Path(path)
    if set(result) != set(METHODS):
        raise RuntimeError("conditional autopsy requires the registered method triplet")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser.parse_args()


def quantile_labels(values: np.ndarray, bins: int = 4) -> tuple[np.ndarray, np.ndarray]:
    edges = np.quantile(values, np.linspace(0.0, 1.0, bins + 1))
    edges = np.maximum.accumulate(edges)
    labels = np.searchsorted(edges[1:-1], values, side="right").astype(np.int8)
    return labels, edges


def cdf_deviation(ranks: np.ndarray) -> float:
    values = np.sort(np.asarray(ranks, dtype=np.float64))
    if not len(values):
        return float("nan")
    upper = np.arange(1, len(values) + 1) / len(values)
    lower = np.arange(len(values)) / len(values)
    return float(max(np.max(np.abs(upper - values)), np.max(np.abs(values - lower))))


def interval_arrays(draws: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
    mean = draws.mean(axis=0)
    std = draws.std(axis=0, ddof=1)
    below = np.sum(draws < truth[None], axis=0)
    equal = np.sum(draws == truth[None], axis=0)
    # Mid-ranks make method comparisons deterministic; the finite-draw sweep
    # below explicitly diagnoses ensemble-size sensitivity.
    rank = (below + 0.5 * (equal + 1.0)) / (draws.shape[0] + 1.0)
    result = {"truth": truth, "mean": mean, "std": std, "rank": rank}
    for level in LEVELS:
        tail = (1.0 - level) / 2.0
        low, high = np.quantile(draws, (tail, 1.0 - tail), axis=0)
        key = str(int(100 * level))
        result[f"low{key}"] = low
        result[f"high{key}"] = high
        result[f"inside{key}"] = (truth >= low) & (truth <= high)
    return result


def validate_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(path.read_text())
    if (
        manifest.get("schema_version") != "p12f-sample-archive-v1"
        or manifest.get("method") not in METHODS
        or manifest.get("phase") != "ph006"
        or manifest.get("ph001_opened")
        or manifest.get("truth_files_read")
        not in (["ph006"], ["ph006 density/T-web"])
        or int(manifest.get("draws", -1)) != 64
    ):
        raise RuntimeError(f"unsafe F3-L2 archive: {path}")
    entries = list(manifest.get("entries", []))
    ids = [int(row["core_id"]) for row in entries]
    if len(entries) != 256 or len(set(ids)) != len(ids):
        raise RuntimeError("F3-L2 autopsy requires the frozen 256-core panel")
    for row in entries:
        artifact = Path(row["path"])
        if "ph001" in str(artifact).lower() or sha256(artifact) != row["sha256"]:
            raise RuntimeError("F3-L2 core artifact is unsafe or changed")
    return manifest, entries


def load_method(path: Path, metadata: dict[int, dict[str, Any]]) -> dict[str, np.ndarray]:
    manifest, entries = validate_manifest(path)
    fields: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "truth", "mean", "std", "rank", "low68", "high68", "inside68",
            "low90", "high90", "inside90", "shell", "response", "boundary",
            "tracer", "core",
        )
    }
    draw_sweeps: dict[int, list[np.ndarray]] = {16: [], 32: [], 64: []}
    for ordinal, entry in enumerate(entries):
        record = load_core_record(entry, 64)
        bounds = np.asarray(record["core_bounds"], dtype=np.int64)
        core = tuple(slice(int(a), int(b)) for a, b in zip(bounds[0], bounds[1], strict=True))
        support = np.asarray(record["support"], dtype=bool)[core]
        valid = np.flatnonzero(support.ravel())
        if not len(valid):
            raise RuntimeError("autopsy core has no supported voxels")
        if len(valid) > 2048:
            valid = valid[np.linspace(0, len(valid) - 1, 2048, dtype=np.int64)]
        draws = np.asarray(record["delta_samples"], dtype=np.float32)[(slice(None),) + core]
        draws = draws.reshape(64, -1)[:, valid]
        truth = np.asarray(record["delta_truth"], dtype=np.float32)[core].ravel()[valid]
        arrays = interval_arrays(draws, truth)
        for key, value in arrays.items():
            fields[key].append(np.asarray(value))
        fields["response"].append(np.asarray(record["angular_response"])[core].ravel()[valid])
        fields["boundary"].append(np.asarray(record["boundary_distance_mpc"])[core].ravel()[valid])
        fields["tracer"].append(np.asarray(record["tracer_density"])[core].ravel()[valid])
        core_id = int(entry["core_id"])
        fields["core"].append(np.full(len(valid), core_id, dtype=np.int64))
        fields["shell"].append(np.full(len(valid), int(metadata[core_id]["shell"]), dtype=np.int8))
        for count in draw_sweeps:
            draw_sweeps[count].append(interval_arrays(draws[:count], truth)["inside68"])
        print(json.dumps({"method": manifest["method"], "core": ordinal + 1, "total": len(entries)}), flush=True)
    output = {key: np.concatenate(value) for key, value in fields.items()}
    output["draw_sweep_16"] = np.concatenate(draw_sweeps[16])
    output["draw_sweep_32"] = np.concatenate(draw_sweeps[32])
    output["draw_sweep_64"] = np.concatenate(draw_sweeps[64])
    output["method"] = np.asarray(manifest["method"])
    return output


def subset_metrics(arrays: dict[str, np.ndarray], selected: np.ndarray) -> dict[str, float]:
    truth = arrays["truth"][selected]
    mean = arrays["mean"][selected]
    std = np.maximum(arrays["std"][selected], 1e-8)
    error = mean - truth
    bias = float(np.mean(error))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    median_std = float(np.median(std))
    result = {
        "rows": int(np.sum(selected)),
        "bias": bias,
        "rmse": rmse,
        "median_posterior_std": median_std,
        "median_std_to_rmse": median_std / max(rmse, 1e-12),
        "mean_standardized_truth_minus_mean": float(np.mean(-error / std)),
        "rms_standardized_truth_minus_mean": float(np.sqrt(np.mean(np.square(error / std)))),
        "rank_mean": float(np.mean(arrays["rank"][selected])),
        "rank_cdf_maximum_deviation": cdf_deviation(arrays["rank"][selected]),
    }
    for level in LEVELS:
        key = str(int(100 * level))
        inside = arrays[f"inside{key}"][selected]
        result[f"coverage{key}"] = float(np.mean(inside))
        # Oracle quantities are explicitly diagnostic: first remove only the
        # mean group location error, then ask what interval scale would be
        # needed to reach the nominal coverage.
        corrected_mean = mean - bias
        low_width = np.maximum(mean - arrays[f"low{key}"][selected], 1e-8)
        high_width = np.maximum(arrays[f"high{key}"][selected] - mean, 1e-8)
        required = np.where(
            truth < corrected_mean,
            (corrected_mean - truth) / low_width,
            (truth - corrected_mean) / high_width,
        )
        result[f"oracle_recentered_scale_to_{key}"] = float(np.quantile(required, level))
        recentered_inside = (
            (truth >= corrected_mean - low_width)
            & (truth <= corrected_mean + high_width)
        )
        result[f"oracle_recentered_coverage{key}"] = float(np.mean(recentered_inside))
    return result


def core_bootstrap_coverage(
    arrays: dict[str, np.ndarray], labels: np.ndarray, *, repeats: int, seed: int
) -> dict[str, Any]:
    unique_cores = np.unique(arrays["core"])
    unique_bins = np.unique(labels)
    counts = np.zeros((len(unique_cores), len(unique_bins)), dtype=np.int64)
    success = {level: np.zeros_like(counts) for level in (68, 90)}
    for ci, core in enumerate(unique_cores):
        core_mask = arrays["core"] == core
        for bi, label in enumerate(unique_bins):
            chosen = core_mask & (labels == label)
            counts[ci, bi] = np.sum(chosen)
            for level in success:
                success[level][ci, bi] = np.sum(arrays[f"inside{level}"][chosen])
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(unique_cores), size=(repeats, len(unique_cores)))
    result: dict[str, Any] = {"spatial_blocks": int(len(unique_cores)), "bins": {}}
    max_error = np.zeros(repeats, dtype=np.float64)
    for bi, label in enumerate(unique_bins):
        row: dict[str, Any] = {}
        denominator = counts[sampled, bi].sum(axis=1)
        for level in (68, 90):
            numerator = success[level][sampled, bi].sum(axis=1)
            coverage = np.divide(numerator, denominator, out=np.full(repeats, np.nan), where=denominator > 0)
            finite = coverage[np.isfinite(coverage)]
            nominal = level / 100.0
            row[str(level)] = {
                "coverage_quantiles_05_50_95": np.quantile(finite, (0.05, 0.5, 0.95)).tolist(),
                "absolute_error_quantiles_05_50_95": np.quantile(np.abs(finite - nominal), (0.05, 0.5, 0.95)).tolist(),
            }
            valid = np.isfinite(coverage)
            max_error[valid] = np.maximum(max_error[valid], np.abs(coverage[valid] - nominal))
        result["bins"][str(int(label))] = row
    result["maximum_bin_error_quantiles_05_50_95"] = np.quantile(max_error, (0.05, 0.5, 0.95)).tolist()
    return result


def global_scale_scan(arrays: dict[str, np.ndarray], environment: np.ndarray) -> dict[str, Any]:
    scales = np.linspace(0.55, 1.45, 181)
    maximum = np.zeros(len(scales), dtype=np.float64)
    rows = np.zeros((len(scales), 4, 2), dtype=np.float64)
    mean = arrays["mean"]
    truth = arrays["truth"]
    for label in range(4):
        selected = environment == label
        for column, level in enumerate((68, 90)):
            low_width = np.maximum(mean[selected] - arrays[f"low{level}"][selected], 1.0e-8)
            high_width = np.maximum(arrays[f"high{level}"][selected] - mean[selected], 1.0e-8)
            required_scale = np.where(
                truth[selected] < mean[selected],
                (mean[selected] - truth[selected]) / low_width,
                (truth[selected] - mean[selected]) / high_width,
            )
            required_scale.sort()
            coverage = np.searchsorted(required_scale, scales, side="right") / len(required_scale)
            rows[:, label, column] = coverage
            maximum = np.maximum(maximum, np.abs(coverage - level / 100.0))
    best = int(np.argmin(maximum))
    return {
        "scale": scales.tolist(),
        "maximum_environment_coverage_error": maximum.tolist(),
        "coverage_by_environment_68_90": rows.tolist(),
        "best_scale": float(scales[best]),
        "best_maximum_error": float(maximum[best]),
    }


def plot_conditional(report: dict[str, Any], output: Path) -> None:
    variables = ("shell", "response", "boundary", "environment")
    figure, axes = plt.subplots(len(variables), 2, figsize=(15, 16), constrained_layout=True)
    for row, variable in enumerate(variables):
        for column, level in enumerate((68, 90)):
            axis = axes[row, column]
            nominal = level / 100.0
            axis.axhline(nominal, color="black", linestyle="--", linewidth=1)
            axis.axhspan(nominal - 0.10, nominal + 0.10, color="#8bc34a", alpha=0.10)
            for method in METHODS:
                bins = report["methods"][method]["strata"][variable]
                x = np.asarray(sorted(int(key) for key in bins if key.isdigit()))
                y = np.asarray([bins[str(value)][f"coverage{level}"] for value in x])
                axis.plot(x, y, marker="o", color=COLORS[method], label=LABELS[method])
            axis.set(
                xticks=range(4), ylim=(0.45, 1.0),
                xlabel=f"{variable} stratum", ylabel="empirical coverage",
                title=f"{variable}: nominal {level}%",
            )
            axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("F3-L2 conditional coverage autopsy — held-out ph006", fontsize=16)
    figure.savefig(output.with_suffix(".png"), dpi=180)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def plot_location_scale(report: dict[str, Any], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    x = np.arange(4)
    for method in METHODS:
        rows = report["methods"][method]["strata"]["environment"]
        axes[0, 0].plot(x, [rows[str(i)]["rank_mean"] for i in x], marker="o", color=COLORS[method], label=LABELS[method])
        axes[0, 1].plot(x, [rows[str(i)]["median_std_to_rmse"] for i in x], marker="o", color=COLORS[method])
        axes[1, 0].plot(x, [rows[str(i)]["oracle_recentered_scale_to_68"] for i in x], marker="o", color=COLORS[method])
        axes[1, 1].plot(x, [rows[str(i)]["oracle_recentered_coverage68"] for i in x], marker="o", color=COLORS[method])
    axes[0, 0].axhline(.5, color="black", linestyle="--"); axes[0, 0].set(title="A. Posterior rank mean (location)", ylabel="mean rank")
    axes[0, 1].axhline(1, color="black", linestyle="--"); axes[0, 1].set(title="B. Posterior spread / mean RMSE", ylabel="median std / RMSE")
    axes[1, 0].axhline(1, color="black", linestyle="--"); axes[1, 0].set(title="C. Truth-assisted scale needed after recentering", ylabel="scale for 68% coverage")
    axes[1, 1].axhline(.68, color="black", linestyle="--"); axes[1, 1].set(title="D. Coverage after location-only correction", ylabel="empirical 68% coverage")
    for axis in axes.ravel():
        axis.set(xticks=x, xlabel="true-density quartile (evaluation only)"); axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Location versus conditional-width failure", fontsize=16)
    figure.savefig(output.with_suffix(".png"), dpi=180)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def plot_crossed(report: dict[str, Any], output: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for row, cross in enumerate(("environment_x_shell", "environment_x_response")):
        for column, method in enumerate(METHODS):
            matrix = np.asarray(report["methods"][method]["crossed"][cross]["coverage68_error"])
            artist = axes[row, column].imshow(matrix, origin="lower", cmap="coolwarm", vmin=-.18, vmax=.18)
            axes[row, column].set(
                xticks=range(4), yticks=range(4),
                xlabel=cross.split("_x_")[1] + " stratum", ylabel="true-density quartile",
                title=f"{LABELS[method]}\n68% coverage error",
            )
            figure.colorbar(artist, ax=axes[row, column], fraction=.046)
    figure.suptitle("Where the conditional error changes sign", fontsize=16)
    figure.savefig(output.with_suffix(".png"), dpi=180)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def plot_scale_and_convergence(report: dict[str, Any], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for method in METHODS:
        row = report["methods"][method]["global_scale_scan"]
        axes[0].plot(row["scale"], row["maximum_environment_coverage_error"], color=COLORS[method], label=f"{LABELS[method]} (best {row['best_scale']:.2f})")
        sweep = report["methods"][method]["draw_count_convergence"]
        axes[1].plot((16, 32, 64), [sweep[str(n)]["maximum_environment_coverage_error68"] for n in (16, 32, 64)], marker="o", color=COLORS[method], label=LABELS[method])
    axes[0].axhline(.10, color="black", linestyle="--", label="conditional gate")
    axes[0].axvline(1, color="black", linestyle=":")
    axes[0].set(xlabel="global interval-width multiplier (diagnostic only)", ylabel="max environment coverage error", title="A. One global scale cannot hide heterogeneity")
    axes[1].axhline(.10, color="black", linestyle="--")
    axes[1].set(xticks=(16, 32, 64), xlabel="posterior draws", ylabel="max 68% environment error", title="B. Finite-ensemble convergence")
    for axis in axes: axis.grid(alpha=.2); axis.legend(fontsize=8)
    figure.savefig(output.with_suffix(".png"), dpi=180)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def crossed_report(arrays: dict[str, np.ndarray], first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    matrix = np.full((4, 4), np.nan)
    rows = np.zeros((4, 4), dtype=np.int64)
    for left in range(4):
        for right in range(4):
            selected = (first == left) & (second == right)
            rows[left, right] = int(np.sum(selected))
            if rows[left, right] >= 64:
                matrix[left, right] = float(np.mean(arrays["inside68"][selected]) - .68)
    return {"coverage68_error": matrix.tolist(), "rows": rows.tolist()}


def build_report(archives: dict[str, Path], repeats: int, seed: int) -> dict[str, Any]:
    manifests = {method: validate_manifest(path)[0] for method, path in archives.items()}
    identities = {method: [int(row["core_id"]) for row in manifest["entries"]] for method, manifest in manifests.items()}
    if any(value != identities[METHODS[0]] for value in identities.values()):
        raise RuntimeError("autopsy archives do not share exact core identities")
    panel_path = Path(manifests[METHODS[0]]["panel_marker"])
    panel = json.loads(panel_path.read_text())
    metadata = {int(row["core_id"]): row for row in panel["selected_core_metadata"]}
    arrays = {method: load_method(path, metadata) for method, path in archives.items()}
    reference = arrays[METHODS[0]]
    for method in METHODS[1:]:
        for key in ("truth", "shell", "response", "boundary", "tracer", "core"):
            if not np.array_equal(reference[key], arrays[method][key]):
                raise RuntimeError(f"autopsy covariate mismatch for {method}:{key}")
    environment, environment_edges = quantile_labels(reference["truth"])
    response, response_edges = quantile_labels(reference["response"])
    boundary, boundary_edges = quantile_labels(reference["boundary"])
    labels = {
        "shell": reference["shell"], "response": response,
        "boundary": boundary, "environment": environment,
    }
    result: dict[str, Any] = {
        "schema_version": "p12f3l2-conditional-autopsy-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "ph006", "cores": 256, "draws": 64,
        "quantile_edges": {"environment": environment_edges.tolist(), "response": response_edges.tolist(), "boundary": boundary_edges.tolist()},
        "methods": {},
        "artifacts": {method: {"path": str(path.resolve()), "sha256": sha256(path)} for method, path in archives.items()},
        "truth_files_read": ["ph006 density/T-web"], "ph001_opened": False,
        "scientific_scope": "diagnostic only; no ph006-fitted posterior is exported",
    }
    for mi, method in enumerate(METHODS):
        row: dict[str, Any] = {"strata": {}, "crossed": {}}
        for vi, (name, label) in enumerate(labels.items()):
            row["strata"][name] = {
                str(value): subset_metrics(arrays[method], label == value)
                for value in np.unique(label)
            }
            row["strata"][name]["block_bootstrap"] = core_bootstrap_coverage(
                arrays[method], label, repeats=repeats, seed=seed + 100 * mi + vi
            )
        row["crossed"]["environment_x_shell"] = crossed_report(arrays[method], environment, labels["shell"])
        row["crossed"]["environment_x_response"] = crossed_report(arrays[method], environment, response)
        row["global_scale_scan"] = global_scale_scan(arrays[method], environment)
        row["draw_count_convergence"] = {}
        for count in (16, 32, 64):
            inside = arrays[method][f"draw_sweep_{count}"]
            errors = [abs(float(np.mean(inside[environment == value])) - .68) for value in range(4)]
            row["draw_count_convergence"][str(count)] = {
                "coverage68_by_environment": [float(np.mean(inside[environment == value])) for value in range(4)],
                "maximum_environment_coverage_error68": max(errors),
            }
        result["methods"][method] = row
    flow = result["methods"]["fourier_flow_h24"]
    environment_scales = [flow["strata"]["environment"][str(i)]["oracle_recentered_scale_to_68"] for i in range(4)]
    bootstrap = flow["strata"]["environment"]["block_bootstrap"]
    result["diagnosis"] = {
        "environment_scale_range": [float(min(environment_scales)), float(max(environment_scales))],
        "environment_scale_span": float(max(environment_scales) - min(environment_scales)),
        "block_bootstrap_max_environment_error_05_50_95": bootstrap["maximum_bin_error_quantiles_05_50_95"],
        "global_scale_best_remaining_error": float(flow["global_scale_scan"]["best_maximum_error"]),
        "finite_draw_change_32_to_64": float(
            flow["draw_count_convergence"]["64"]["maximum_environment_coverage_error68"]
            - flow["draw_count_convergence"]["32"]["maximum_environment_coverage_error68"]
        ),
        "interpretation": (
            "conditional location/scale heterogeneity if environment scale span exceeds 0.15; "
            "a single post-hoc temperature is not a production correction"
        ),
    }
    return result


def main() -> None:
    args = parse_args()
    archives = parse_keyed(args.archive)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(archives, args.bootstrap_repeats, args.seed)
    output = args.output_dir / "P12F3L2_CONDITIONAL_AUTOPSY.json"
    atomic_json(output, report)
    plot_conditional(report, args.output_dir / "p12f3l2_conditional_coverage")
    plot_location_scale(report, args.output_dir / "p12f3l2_location_scale")
    plot_crossed(report, args.output_dir / "p12f3l2_crossed_strata")
    plot_scale_and_convergence(report, args.output_dir / "p12f3l2_scale_draw_convergence")
    print(json.dumps({"output": str(output), "sha256": sha256(output), "diagnosis": report["diagnosis"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
