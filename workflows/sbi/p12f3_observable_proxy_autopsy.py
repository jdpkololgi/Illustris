#!/usr/bin/env python3
"""Observable-proxy and calibrand audit for frozen P12-F3-L2b samples.

This module is evaluation-only.  It reads the frozen ph006 archive, never fits a
calibration map, and never opens ph001.  The split-draw reference asks how much
non-flat coverage versus a realised field value is expected even when a draw is
exchangeable with the posterior ensemble used to construct the interval.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.sbi.p12f3_export_hybrid_archive import lowpass_numpy
from workflows.sbi.p12f3l2_conditional_autopsy import (
    cdf_deviation,
    quantile_labels,
    validate_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/p12f3_conditional_calibration_v1.json"
LEVELS = (0.68, 0.90)
PROXY_ORDER = (
    "shell",
    "response",
    "boundary",
    "tracer",
    "predicted_density",
    "posterior_width",
    "predicted_shear",
    "predicted_web_class",
    "true_density_diagnostic",
)
PROXY_LABELS = {
    "shell": "redshift shell",
    "response": "random response",
    "boundary": "boundary distance",
    "tracer": "BRIGHT tracer density",
    "predicted_density": "posterior-mean density",
    "posterior_width": "posterior width",
    "predicted_shear": "predicted shear amplitude",
    "predicted_web_class": "predicted web class",
    "true_density_diagnostic": "true density (diagnostic)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("schema_version") != "p12f3-conditional-calibration-v1":
        raise RuntimeError("unexpected conditional-calibration contract")
    if (
        value["roles"]["validation"] != "ph006"
        or value["roles"]["sealed_blind_test"] != "ph001"
        or value["scope"].get("ph001_opened")
        or "ph001" in json.dumps(value["sources"]).lower()
    ):
        raise PermissionError("observable-proxy audit phase contract changed")
    return value


def load_record(entry: dict[str, Any], draws: int) -> dict[str, np.ndarray]:
    required = {
        "delta_samples", "delta_truth", "support", "angular_response",
        "boundary_distance_mpc", "tracer_density", "core_bounds",
    }
    path = Path(entry["path"])
    if sha256(path) != entry["sha256"] or "ph001" in str(path).lower():
        raise RuntimeError("unsafe or changed archive core")
    with np.load(path, allow_pickle=False) as values:
        if not required.issubset(values.files):
            raise RuntimeError(f"archive core is missing {sorted(required-set(values.files))}")
        output = {name: np.asarray(values[name]) for name in required}
    if output["delta_samples"].shape[0] != draws:
        raise RuntimeError("draw count changed")
    return output


def tidal_features(field: np.ndarray, core: tuple[slice, slice, slice], valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return traceless-shear norm and deterministic web class at selected voxels."""
    delta = np.asarray(field, dtype=np.float64)
    shape = delta.shape
    k = [
        np.fft.fftfreq(shape[axis]).reshape(tuple(shape[axis] if i == axis else 1 for i in range(3)))
        for axis in range(3)
    ]
    k2 = k[0] ** 2 + k[1] ** 2 + k[2] ** 2
    safe = np.where(k2 > 0, k2, 1.0)
    spectrum = np.fft.fftn(delta)
    components: dict[tuple[int, int], np.ndarray] = {}
    for left in range(3):
        for right in range(left, 3):
            kernel = np.where(k2 > 0, k[left] * k[right] / safe, 0.0)
            values = np.fft.ifftn(spectrum * kernel).real[core].ravel()[valid]
            components[(left, right)] = values
    matrix = np.zeros((len(valid), 3, 3), dtype=np.float64)
    for (left, right), values in components.items():
        matrix[:, left, right] = values
        matrix[:, right, left] = values
    trace = np.trace(matrix, axis1=1, axis2=2) / 3.0
    shear = matrix - trace[:, None, None] * np.eye(3)[None]
    amplitude = np.sqrt(np.sum(shear * shear, axis=(1, 2)))
    eigen = np.linalg.eigvalsh(matrix)
    web_class = np.sum(eigen > 0.2, axis=1).astype(np.int8)
    return amplitude.astype(np.float32), web_class


def _subsample(support: np.ndarray, maximum: int = 2048) -> np.ndarray:
    valid = np.flatnonzero(np.asarray(support, dtype=bool).ravel())
    if not len(valid):
        raise RuntimeError("frozen core has no supported voxels")
    if len(valid) > maximum:
        valid = valid[np.linspace(0, len(valid) - 1, maximum, dtype=np.int64)]
    return valid


def assemble(manifest: dict[str, Any], entries: list[dict[str, Any]], maximum_k: float) -> dict[str, np.ndarray]:
    metadata = {
        int(row["core_id"]): row
        for row in json.loads(Path(manifest["panel_marker"]).read_text())["selected_core_metadata"]
    }
    pieces: dict[str, list[np.ndarray]] = {name: [] for name in (
        "draws", "truth", "core", "shell", "response", "boundary", "tracer",
        "predicted_density", "posterior_width", "predicted_shear",
        "predicted_web_class", "low_variance", "high_variance",
    )}
    for ordinal, entry in enumerate(entries):
        record = load_record(entry, int(manifest["draws"]))
        bounds = np.asarray(record["core_bounds"], dtype=np.int64)
        core = tuple(slice(int(a), int(b)) for a, b in zip(bounds[0], bounds[1], strict=True))
        support = np.asarray(record["support"], dtype=bool)[core]
        valid = _subsample(support)
        draws_full = np.asarray(record["delta_samples"], dtype=np.float32)
        truth_full = np.asarray(record["delta_truth"], dtype=np.float32)
        draws = draws_full[(slice(None),) + core].reshape(len(draws_full), -1)[:, valid]
        truth = truth_full[core].ravel()[valid]
        mean_full = draws_full.mean(axis=0)
        mean = mean_full[core].ravel()[valid]
        width = draws.std(axis=0, ddof=1)
        centered = draws_full - mean_full[None]
        low = lowpass_numpy(centered, voxel_mpc_h=5.0, maximum_k=maximum_k)
        high = centered - low
        low_variance = np.var(low[(slice(None),) + core].reshape(len(low), -1)[:, valid], axis=0, ddof=1)
        high_variance = np.var(high[(slice(None),) + core].reshape(len(high), -1)[:, valid], axis=0, ddof=1)
        shear, web = tidal_features(mean_full, core, valid)
        core_id = int(entry["core_id"])
        count = len(valid)
        pieces["draws"].append(draws)
        pieces["truth"].append(truth)
        pieces["predicted_density"].append(mean)
        pieces["posterior_width"].append(width)
        pieces["predicted_shear"].append(shear)
        pieces["predicted_web_class"].append(web)
        pieces["low_variance"].append(low_variance)
        pieces["high_variance"].append(high_variance)
        pieces["core"].append(np.full(count, core_id, dtype=np.int64))
        pieces["shell"].append(np.full(count, int(metadata[core_id]["shell"]), dtype=np.int8))
        pieces["response"].append(np.asarray(record["angular_response"])[core].ravel()[valid])
        pieces["boundary"].append(np.asarray(record["boundary_distance_mpc"])[core].ravel()[valid])
        pieces["tracer"].append(np.asarray(record["tracer_density"])[core].ravel()[valid])
        print(json.dumps({"stage": "assemble", "core": ordinal + 1, "total": len(entries)}), flush=True)
    result: dict[str, np.ndarray] = {}
    for name, values in pieces.items():
        result[name] = np.concatenate(values, axis=1 if name == "draws" else 0)
    return result


def strata(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    output = {
        "shell": arrays["shell"].astype(np.int8),
        "predicted_web_class": arrays["predicted_web_class"].astype(np.int8),
    }
    for name in (
        "response", "boundary", "tracer", "predicted_density",
        "posterior_width", "predicted_shear",
    ):
        output[name], _ = quantile_labels(arrays[name], 4)
    output["true_density_diagnostic"], _ = quantile_labels(arrays["truth"], 4)
    return output


def intervals(draws: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    tail = (1.0 - level) / 2.0
    return tuple(np.quantile(draws, (tail, 1.0 - tail), axis=0))


def metrics_for_labels(draws: np.ndarray, truth: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    mean = draws.mean(axis=0)
    std = np.maximum(draws.std(axis=0, ddof=1), 1e-8)
    rank = (np.sum(draws < truth[None], axis=0) + 0.5) / (len(draws) + 1.0)
    output: dict[str, Any] = {}
    for label in np.unique(labels):
        chosen = labels == label
        if np.sum(chosen) < 64:
            continue
        residual = truth[chosen] - mean[chosen]
        row: dict[str, Any] = {
            "rows": int(np.sum(chosen)),
            "mean_truth_minus_mean_over_std": float(np.mean(residual / std[chosen])),
            "rms_truth_minus_mean_over_std": float(np.sqrt(np.mean(np.square(residual / std[chosen])))),
            "mean_bias": float(np.mean(mean[chosen] - truth[chosen])),
            "rmse": float(np.sqrt(np.mean(np.square(mean[chosen] - truth[chosen])))),
            "median_width": float(np.median(std[chosen])),
            "rank_mean": float(np.mean(rank[chosen])),
            "rank_cdf_maximum_deviation": cdf_deviation(rank[chosen]),
        }
        for level in LEVELS:
            low, high = intervals(draws[:, chosen], level)
            coverage = float(np.mean((truth[chosen] >= low) & (truth[chosen] <= high)))
            row[f"coverage{int(100*level)}"] = coverage
            row[f"coverage_error{int(100*level)}"] = coverage - level
        output[str(int(label))] = row
    return output


def metric_by_label(rows: dict[str, Any], field: str, labels: range = range(4)) -> list[float]:
    """Return fixed-axis metric values while preserving genuinely empty strata."""
    return [float(rows[str(label)][field]) if str(label) in rows else float("nan") for label in labels]


def finite_range(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.ptp(array)) if len(array) >= 2 else 0.0


def safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def self_consistency(
    draws: np.ndarray,
    actual_truth: np.ndarray,
    proxy_labels: dict[str, np.ndarray],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    collectors: dict[str, dict[str, list[np.ndarray]]] = {
        name: {"actual68": [], "actual90": [], "pseudo68": [], "pseudo90": []}
        for name in proxy_labels
    }
    for _ in range(repetitions):
        order = rng.permutation(len(draws))
        interval_draws = draws[order[: len(draws)//2]]
        pseudo_truths = draws[order[len(draws)//2 :]]
        for level in LEVELS:
            low, high = intervals(interval_draws, level)
            actual_inside = (actual_truth >= low) & (actual_truth <= high)
            pseudo_inside = (pseudo_truths >= low[None]) & (pseudo_truths <= high[None])
            suffix = str(int(100 * level))
            for name, labels in proxy_labels.items():
                actual_values, pseudo_values = [], []
                for label in range(4):
                    chosen = labels == label
                    actual_values.append(safe_mean(actual_inside[chosen]))
                    if name == "true_density_diagnostic":
                        per_draw = []
                        for pseudo, inside in zip(pseudo_truths, pseudo_inside, strict=True):
                            pseudo_labels, _ = quantile_labels(pseudo, 4)
                            per_draw.append(safe_mean(inside[pseudo_labels == label]))
                        pseudo_values.append(float(np.nanmean(per_draw)) if np.any(np.isfinite(per_draw)) else float("nan"))
                    else:
                        pseudo_values.append(safe_mean(pseudo_inside[:, chosen]))
                collectors[name][f"actual{suffix}"].append(np.asarray(actual_values))
                collectors[name][f"pseudo{suffix}"].append(np.asarray(pseudo_values))
    output: dict[str, Any] = {}
    for name, values in collectors.items():
        row: dict[str, Any] = {}
        for key, entries in values.items():
            matrix = np.stack(entries)
            row[key] = {
                "mean": matrix.mean(axis=0).tolist(),
                "q05_q95": np.quantile(matrix, (0.05, 0.95), axis=0).T.tolist(),
            }
        deviations = np.concatenate([
            np.abs(np.asarray(row[f"actual{level}"]["mean"]) - np.asarray(row[f"pseudo{level}"]["mean"]))
            for level in (68, 90)
        ])
        row["actual_minus_pseudo_max_abs"] = float(np.nanmax(deviations))
        output[name] = row
    return output


def core_bootstrap(arrays: dict[str, np.ndarray], labels: dict[str, np.ndarray], repeats: int, seed: int) -> dict[str, Any]:
    cores = np.unique(arrays["core"])
    rng = np.random.default_rng(seed)
    output: dict[str, Any] = {}
    for name, group in labels.items():
        totals = np.zeros((len(cores), 4), dtype=np.int64)
        success = {68: np.zeros_like(totals), 90: np.zeros_like(totals)}
        for index, core in enumerate(cores):
            core_mask = arrays["core"] == core
            for label in range(4):
                chosen = core_mask & (group == label)
                totals[index, label] = int(np.sum(chosen))
                for level in (68, 90):
                    low, high = intervals(arrays["draws"][:, chosen], level / 100)
                    success[level][index, label] = int(np.sum((arrays["truth"][chosen] >= low) & (arrays["truth"][chosen] <= high)))
        sampled = rng.integers(0, len(cores), size=(repeats, len(cores)))
        max_error = np.zeros(repeats)
        for level in (68, 90):
            denominator = totals[sampled].sum(axis=1)
            numerator = success[level][sampled].sum(axis=1)
            coverage = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan, dtype=float), where=denominator > 0)
            max_error = np.maximum(max_error, np.nanmax(np.abs(coverage - level / 100), axis=1))
        output[name] = {"maximum_error_q05_q50_q95": np.quantile(max_error, (0.05, 0.5, 0.95)).tolist()}
    return output


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    ar = np.empty(len(a), dtype=np.int64); ar[np.argsort(a, kind="stable")] = np.arange(len(a))
    br = np.empty(len(b), dtype=np.int64); br[np.argsort(b, kind="stable")] = np.arange(len(b))
    return float(np.corrcoef(ar, br)[0, 1])


def make_plots(report: dict[str, Any], output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    figure, axes = plt.subplots(3, 3, figsize=(17, 14), constrained_layout=True)
    for axis, name in zip(axes.ravel(), PROXY_ORDER, strict=True):
        row = report["self_consistency"][name]
        x = np.arange(4)
        for level, color in ((68, "#d1495b"), (90, "#2878b5")):
            actual = np.asarray(row[f"actual{level}"]["mean"])
            pseudo = np.asarray(row[f"pseudo{level}"]["mean"])
            axis.plot(x, actual, color=color, marker="o", label=f"truth {level}%")
            axis.plot(x, pseudo, color=color, marker="s", linestyle="--", alpha=.75, label=f"pseudo-truth {level}%")
            axis.axhline(level / 100, color=color, linewidth=.7, alpha=.3)
        axis.set(title=PROXY_LABELS[name], xlabel="stratum", ylabel="coverage", xticks=x, ylim=(.45, 1.0))
        axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7, ncol=2)
    figure.suptitle("F3-L2b observable-proxy coverage versus split-draw self-consistency", fontsize=16)
    path = output / "p12f3_observable_proxy_self_consistency.png"
    figure.savefig(path, dpi=180); figure.savefig(path.with_suffix(".pdf")); plt.close(figure); paths.append(path)

    figure, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    selected = ("response", "tracer", "predicted_density", "predicted_shear")
    metrics = report["metrics"]
    for column, name in enumerate(selected):
        rows = metrics[name]
        x = np.arange(4)
        axes[0, column].plot(x, metric_by_label(rows, "mean_truth_minus_mean_over_std"), marker="o", color="#d1495b")
        axes[0, column].axhline(0, color="black", linestyle="--")
        axes[0, column].set(title=PROXY_LABELS[name], ylabel="mean normalized location error")
        axes[1, column].plot(x, metric_by_label(rows, "rms_truth_minus_mean_over_std"), marker="o", color="#2878b5")
        axes[1, column].axhline(1, color="black", linestyle="--")
        axes[1, column].set(ylabel="RMS normalized error", xlabel="stratum")
        for axis in axes[:, column]: axis.set(xticks=x); axis.grid(alpha=.2)
    figure.suptitle("Location and scale diagnostics using deployable proxies", fontsize=16)
    path = output / "p12f3_observable_proxy_location_scale.png"
    figure.savefig(path, dpi=180); figure.savefig(path.with_suffix(".pdf")); plt.close(figure); paths.append(path)

    variance = report["variance_attribution"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    x = np.arange(4)
    for axis, name in zip(axes, ("predicted_density", "predicted_shear"), strict=True):
        row = variance[name]
        axis.plot(x, row["median_low_fraction"], marker="o", label="low-mode variance fraction")
        axis.plot(x, row["median_total_variance"], marker="s", label="total posterior variance")
        axis.set(title=PROXY_LABELS[name], xlabel="stratum", xticks=x); axis.grid(alpha=.2); axis.legend(fontsize=8)
    path = output / "p12f3_observable_proxy_variance_attribution.png"
    figure.savefig(path, dpi=180); figure.savefig(path.with_suffix(".pdf")); plt.close(figure); paths.append(path)
    return paths


def build_report(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    archive_path = Path(config["sources"]["f3l2b_archive"])
    manifest, entries = validate_manifest(archive_path)
    if manifest["method"] != "fourier_flow_h24" or manifest["panel_sha256"] != sha256(Path(config["sources"]["source_panel"])):
        raise RuntimeError("observable audit did not receive the frozen 30k flow panel")
    maximum_k = float(config["target"]["band_edges_h_mpc"][-1])
    arrays = assemble(manifest, entries, maximum_k)
    labels = strata(arrays)
    metrics = {name: metrics_for_labels(arrays["draws"], arrays["truth"], value) for name, value in labels.items()}
    split = config["proxy_contract"]["posterior_self_consistency"]
    self_report = self_consistency(
        arrays["draws"], arrays["truth"], labels,
        repetitions=int(split["fixed_split_repetitions"]), seed=int(config["training"]["seed"]) + 991,
    )
    variance: dict[str, Any] = {}
    total = arrays["low_variance"] + arrays["high_variance"]
    fraction = arrays["low_variance"] / np.maximum(total, 1e-12)
    for name in ("predicted_density", "predicted_shear"):
        variance[name] = {
            "median_low_fraction": [float(np.median(fraction[labels[name] == i])) for i in range(4)],
            "median_total_variance": [float(np.median(total[labels[name] == i])) for i in range(4)],
        }
    associations = {
        name: correlation(arrays[name], arrays["truth"])
        for name in ("response", "boundary", "tracer", "predicted_density", "posterior_width", "predicted_shear")
    }
    deployable = ("shell", "response", "boundary", "tracer", "predicted_density", "posterior_width", "predicted_shear", "predicted_web_class")
    maximum_proxy_error = max(
        abs(row[f"coverage_error{level}"])
        for name in deployable for row in metrics[name].values() for level in (68, 90)
    )
    maximum_excess = max(self_report[name]["actual_minus_pseudo_max_abs"] for name in deployable)
    physical_signal = max(
        finite_range(metric_by_label(metrics[name], "coverage68"))
        for name in ("predicted_density", "predicted_shear", "predicted_web_class")
    )
    return {
        "schema_version": "p12f3-observable-proxy-autopsy-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pass": True,
        "phase": "ph006",
        "cores": int(len(entries)),
        "draws": int(manifest["draws"]),
        "sampled_supported_voxels": int(len(arrays["truth"])),
        "metrics": metrics,
        "self_consistency": self_report,
        "core_bootstrap": core_bootstrap(
            arrays, labels, int(config["proxy_contract"]["core_bootstrap_repeats"]), int(config["training"]["seed"]) + 177,
        ),
        "proxy_truth_spearman": associations,
        "variance_attribution": variance,
        "diagnosis": {
            "maximum_deployable_proxy_coverage_error": float(maximum_proxy_error),
            "maximum_actual_minus_self_consistent_coverage": float(maximum_excess),
            "maximum_physical_proxy_coverage_range68": float(physical_signal),
            "observable_proxy_signal": bool(physical_signal >= 0.05),
            "interpretation": "signal licenses a train-only conditional location/scale control; it does not license ph006 recalibration",
        },
        "frozen": {
            "config": str(config_path.resolve()), "config_sha256": sha256(config_path),
            "archive": str(archive_path.resolve()), "archive_sha256": sha256(archive_path),
            "source_panel_sha256": sha256(Path(config["sources"]["source_panel"])),
            "source_sha256": sha256(Path(__file__)),
        },
        "truth_files_read": ["ph006 density field in frozen archive"],
        "ph006_used_for_fit": False,
        "ph001_opened": False,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(config, args.config)
    report_path = args.output_dir / "P12F3_OBSERVABLE_PROXY_AUTOPSY.json"
    paths = make_plots(report, args.output_dir)
    report["plots"] = [{"path": str(path.resolve()), "sha256": sha256(path)} for path in paths]
    atomic_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
