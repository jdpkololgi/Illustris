#!/usr/bin/env python3
"""Test whether frozen P12-A posterior width tracks available information.

The diagnostic uses only the registered ph006 folds-2--4 evaluation rows and
their cached posterior draws.  It distinguishes response information already
given to P12-A (redshift, expected tracer density and random-support boundary
distance) from an external realised-configuration diagnostic derived from the
observed BRIGHT catalogue (neighbour counts within 7, 10 and 20 Mpc/h).

ph001 is never opened.  The local-count relation is residualised within
redshift-shell x predicted-trace deciles so that ordinary environment-density
correlation is not mistaken for uncertainty adaptation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.plot_style import ACCENT_COLORS, TEXT_COLOR, apply_style
from workflows.abacus_tweb.p8_deterministic_common import sha256
from workflows.abacus_tweb.p10_training_contract import atomic_json
from workflows.sbi.p12_calibration_diagnostics import randomized_pit
from workflows.sbi.p12_train_base_response_fmpe import theta_to_eigenvalues


ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12a_base_response_v1")
AUDIT = ROOT / "fmpe_seed42" / "calibration_audit_v1"
POINTS = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph006/p1_canonical/points.npy")
ADAPTER = Path(
    "/pscratch/sd/d/dkololgi/abacus/p10_multiphase/"
    "training_contract_r1_random/adapters/ph006/field"
)
OUTPUT = ROOT / "fmpe_seed42" / "width_information_diagnostic_v1"
EIGEN_NAMES = ("lambda1", "lambda2", "lambda3")
EIGEN_LABELS = (r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$")
COLORS = (ACCENT_COLORS["blue"], ACCENT_COLORS["magenta"], "#F5C144")
PLANCK18_H = 0.6766


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(np.asarray(values, dtype=np.float64), weights=weights))


def weighted_quantile(
    values: np.ndarray, quantiles: float | np.ndarray, weights: np.ndarray
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    quantiles = np.atleast_1d(quantiles).astype(np.float64)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= weights.sum()
    return np.interp(quantiles, cumulative, values)


def weighted_corr(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    mx = weighted_mean(x, weight)
    my = weighted_mean(y, weight)
    xx = np.sum(weight * np.square(x - mx))
    yy = np.sum(weight * np.square(y - my))
    if xx <= 0.0 or yy <= 0.0:
        return float("nan")
    return float(np.sum(weight * (x - mx) * (y - my)) / np.sqrt(xx * yy))


def weighted_quantile_bin(
    values: np.ndarray, weights: np.ndarray, bins: int
) -> np.ndarray:
    edges = np.unique(
        weighted_quantile(values, np.linspace(0.0, 1.0, bins + 1), weights)
    )
    if len(edges) < 2:
        return np.zeros(len(values), dtype=np.int16)
    return np.minimum(np.digitize(values, edges[1:-1], right=False), len(edges) - 2)

def residualize_by_group(
    values: np.ndarray, groups: np.ndarray, weight: np.ndarray
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    result = np.empty_like(values)
    for group in np.unique(groups):
        chosen = groups == group
        result[chosen] = values[chosen] - weighted_mean(values[chosen], weight[chosen])
    return result


def block_bootstrap_corr(
    x: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    groups: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """Cluster-bootstrap a weighted correlation using per-block sufficient stats."""
    unique, inverse = np.unique(groups, return_inverse=True)
    blocks = len(unique)
    stats = np.zeros((blocks, 6), dtype=np.float64)
    for column, values in enumerate(
        (weight, weight * x, weight * y, weight * x * x, weight * y * y, weight * x * y)
    ):
        stats[:, column] = np.bincount(inverse, weights=values, minlength=blocks)

    def correlation(total: np.ndarray) -> float:
        w, wx, wy, wxx, wyy, wxy = total
        cov = wxy - wx * wy / w
        vx = wxx - wx * wx / w
        vy = wyy - wy * wy / w
        return float(cov / np.sqrt(max(vx * vy, 1.0e-30)))

    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        selected = rng.integers(0, blocks, size=blocks)
        draws[index] = correlation(stats[selected].sum(axis=0))
    return {
        "correlation": correlation(stats.sum(axis=0)),
        "spatial_blocks": int(blocks),
        "bootstrap_repeats": int(repeats),
        "bootstrap_q025_q50_q975": np.quantile(draws, [0.025, 0.5, 0.975]).tolist(),
    }


def local_bright_information(
    points_path: Path,
    active_parent_path: Path,
    active_offsets_path: Path,
    core_cap_path: Path,
    target_parent: np.ndarray,
    target_cap: np.ndarray,
    workers: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    points = np.load(points_path, mmap_mode="r")
    active_parent = np.load(active_parent_path, mmap_mode="r")
    active_offsets = np.load(active_offsets_path, mmap_mode="r")
    core_cap = np.load(core_cap_path, mmap_mode="r")
    active_cap = np.repeat(np.asarray(core_cap), np.diff(np.asarray(active_offsets)))
    if len(active_cap) != len(active_parent):
        raise RuntimeError("core offsets do not span active-parent array")
    if np.max(active_parent) >= len(points):
        raise RuntimeError("active parent exceeds points array")
    # P1/P3 observer coordinates are comoving Mpc. Convert to h^-1 Mpc
    # numerical coordinates before applying the registered physical radii.
    target_xyz = np.asarray(points[target_parent], dtype=np.float64) * PLANCK18_H
    result = {
        "neighbors_r7": np.empty(len(target_parent), dtype=np.int32),
        "neighbors_r10": np.empty(len(target_parent), dtype=np.int32),
        "neighbors_r20": np.empty(len(target_parent), dtype=np.int32),
        "fifth_neighbor_distance": np.empty(len(target_parent), dtype=np.float32),
    }
    for cap_value in (0, 1):
        catalogue_parent = np.asarray(active_parent[active_cap == cap_value], dtype=np.int64)
        tree = cKDTree(
            np.asarray(points[catalogue_parent], dtype=np.float64) * PLANCK18_H
        )
        chosen = target_cap == cap_value
        query = target_xyz[chosen]
        for radius, name in (
            (7.0, "neighbors_r7"),
            (10.0, "neighbors_r10"),
            (20.0, "neighbors_r20"),
        ):
            # The target galaxy itself is present in the catalogue and is removed.
            count = tree.query_ball_point(query, radius, return_length=True, workers=workers)
            result[name][chosen] = np.maximum(np.asarray(count, dtype=np.int32) - 1, 0)
        distances, _ = tree.query(query, k=6, workers=workers)
        result["fifth_neighbor_distance"][chosen] = distances[:, 5]
    return result, target_xyz


def interval_widths(eigen_samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q05, q16, q50, q84, q95 = np.quantile(
        eigen_samples, [0.05, 0.16, 0.5, 0.84, 0.95], axis=1
    )
    return 0.5 * (q84 - q16), q50, np.stack((q05, q16, q84, q95), axis=1)


def shell_width_report(
    width: np.ndarray, shell: np.ndarray, weight: np.ndarray
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in range(4):
        chosen = shell == value
        result[str(value)] = {
            "rows": int(chosen.sum()),
            "weighted_median_half_width": [
                float(weighted_quantile(width[chosen, component], 0.5, weight[chosen])[0])
                for component in range(3)
            ],
            "weighted_mean_half_width": [
                weighted_mean(width[chosen, component], weight[chosen])
                for component in range(3)
            ],
        }
    dense = np.asarray(result["0"]["weighted_median_half_width"])
    sparse = np.asarray(result["3"]["weighted_median_half_width"])
    result["sparse_to_dense_median_ratio"] = (sparse / dense).tolist()
    result["response_adaptation_heuristic"] = {
        "criterion": "shell3/shell0 median half-width > 1.10 for at least two eigenvalues",
        "pass": bool(np.count_nonzero(sparse / dense > 1.10) >= 2),
    }
    return result


def coverage_by_shell(
    intervals: np.ndarray, truth: np.ndarray, shell: np.ndarray, weight: np.ndarray
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in range(4):
        chosen = shell == value
        inside68 = (truth[chosen] >= intervals[chosen, 1]) & (
            truth[chosen] <= intervals[chosen, 2]
        )
        inside90 = (truth[chosen] >= intervals[chosen, 0]) & (
            truth[chosen] <= intervals[chosen, 3]
        )
        below68 = truth[chosen] < intervals[chosen, 1]
        above68 = truth[chosen] > intervals[chosen, 2]
        below90 = truth[chosen] < intervals[chosen, 0]
        above90 = truth[chosen] > intervals[chosen, 3]
        result[str(value)] = {
            "coverage68": [
                weighted_mean(inside68[:, component], weight[chosen])
                for component in range(3)
            ],
            "coverage90": [
                weighted_mean(inside90[:, component], weight[chosen])
                for component in range(3)
            ],
            "below68": [
                weighted_mean(below68[:, component], weight[chosen])
                for component in range(3)
            ],
            "above68": [
                weighted_mean(above68[:, component], weight[chosen])
                for component in range(3)
            ],
            "below90": [
                weighted_mean(below90[:, component], weight[chosen])
                for component in range(3)
            ],
            "above90": [
                weighted_mean(above90[:, component], weight[chosen])
                for component in range(3)
            ],
        }
    return result


def error_by_width_quartile(
    width: np.ndarray,
    median: np.ndarray,
    truth: np.ndarray,
    weight: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for component, name in enumerate(EIGEN_NAMES):
        group = weighted_quantile_bin(width[:, component], weight, 4)
        rows = []
        for quartile in range(4):
            chosen = group == quartile
            error = median[chosen, component] - truth[chosen, component]
            rows.append(
                {
                    "quartile": quartile,
                    "rows": int(chosen.sum()),
                    "weighted_rmse": float(
                        np.sqrt(weighted_mean(np.square(error), weight[chosen]))
                    ),
                    "weighted_median_absolute_error": float(
                        weighted_quantile(np.abs(error), 0.5, weight[chosen])[0]
                    ),
                    "weighted_median_half_width": float(
                        weighted_quantile(width[chosen, component], 0.5, weight[chosen])[0]
                    ),
                }
            )
        ratio = rows[-1]["weighted_rmse"] / rows[0]["weighted_rmse"]
        result[name] = {
            "quartiles": rows,
            "widest_to_narrowest_rmse_ratio": ratio,
        }
    result["error_discrimination_heuristic"] = {
        "criterion": "widest/narrowest posterior-width quartile RMSE > 1.15 for at least two eigenvalues",
        "pass": bool(
            sum(result[name]["widest_to_narrowest_rmse_ratio"] > 1.15 for name in EIGEN_NAMES)
            >= 2
        ),
    }
    return result


def controlled_information_report(
    width: np.ndarray,
    base: np.ndarray,
    shell: np.ndarray,
    weight: np.ndarray,
    spatial_group: np.ndarray,
    local: dict[str, np.ndarray],
    log_ntilde: np.ndarray,
    log_boundary: np.ndarray,
    bootstrap_repeats: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    trace = base.sum(axis=1)
    trace_bin = np.empty(len(trace), dtype=np.int16)
    for value in range(4):
        chosen = shell == value
        trace_bin[chosen] = weighted_quantile_bin(trace[chosen], weight[chosen], 10)
    cell = shell.astype(np.int64) * 10 + trace_bin
    local_residual = residualize_by_group(np.log1p(local["neighbors_r20"]), cell, weight)
    ntilde_residual = residualize_by_group(log_ntilde, cell, weight)
    boundary_residual = residualize_by_group(log_boundary, cell, weight)
    result: dict[str, Any] = {
        "control": "residuals within redshift-shell x predicted-trace decile",
        "local_proxy": "log1p observed BRIGHT neighbours within 20 Mpc/h, target excluded",
        "components": {},
    }
    width_residual = np.empty_like(width, dtype=np.float64)
    for component, name in enumerate(EIGEN_NAMES):
        width_residual[:, component] = residualize_by_group(
            np.log(np.maximum(width[:, component], 1.0e-8)), cell, weight
        )
        result["components"][name] = {
            "local_count": block_bootstrap_corr(
                local_residual,
                width_residual[:, component],
                weight,
                spatial_group,
                bootstrap_repeats,
                seed + component,
            ),
            "ntilde": block_bootstrap_corr(
                ntilde_residual,
                width_residual[:, component],
                weight,
                spatial_group,
                bootstrap_repeats,
                seed + 10 + component,
            ),
            "boundary_distance": block_bootstrap_corr(
                boundary_residual,
                width_residual[:, component],
                weight,
                spatial_group,
                bootstrap_repeats,
                seed + 20 + component,
            ),
        }
    return result, local_residual, width_residual


def choose_examples(
    truth: np.ndarray,
    width: np.ndarray,
    median: np.ndarray,
    intervals: np.ndarray,
    pit: np.ndarray,
    shell: np.ndarray,
    local: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    scalar_width = np.mean(width / np.median(width, axis=0), axis=1)
    inside68 = (truth >= intervals[:, 1]) & (truth <= intervals[:, 2])
    normalized_error = np.mean(np.abs(median - truth) / np.maximum(width, 1.0e-8), axis=1)

    def closest(mask: np.ndarray, score: np.ndarray) -> int:
        index = np.flatnonzero(mask)
        if not len(index):
            raise RuntimeError("no row satisfies deterministic example rule")
        return int(index[np.argmin(score[index])])

    dense_target_width = np.quantile(scalar_width[shell == 0], 0.25)
    dense = closest(
        (shell == 0) & np.all(inside68, axis=1),
        np.abs(scalar_width - dense_target_width) + 0.1 * normalized_error,
    )
    sparse_target_width = np.quantile(scalar_width[shell == 3], 0.5)
    sparse_success = closest(
        (shell == 3) & np.all(inside68, axis=1),
        np.abs(scalar_width - sparse_target_width) + 0.1 * normalized_error,
    )
    lambda2_failure = closest(
        (shell == 3) & (pit[:, 1] < 0.16),
        np.abs(pit[:, 1] - 0.08) + 0.1 * np.abs(scalar_width - sparse_target_width),
    )
    lambda3_failure = closest(
        (shell == 3) & (pit[:, 2] < 0.16) & (np.arange(len(shell)) != lambda2_failure),
        np.abs(pit[:, 2] - 0.08) + 0.1 * np.abs(scalar_width - sparse_target_width),
    )
    rows = [
        ("dense_narrow_covered", dense),
        ("sparse_wider_covered", sparse_success),
        ("sparse_lambda2_lower_tail_miss", lambda2_failure),
        ("sparse_lambda3_lower_tail_miss", lambda3_failure),
    ]
    return [
        {
            "label": label,
            "evaluation_row": index,
            "shell": int(shell[index]),
            "neighbors_r20": int(local["neighbors_r20"][index]),
            "half_width": width[index].tolist(),
            "truth": truth[index].tolist(),
            "posterior_median": median[index].tolist(),
            "pit": pit[index].tolist(),
        }
        for label, index in rows
    ]


def plot_summary(
    output: Path,
    shell_width: dict[str, Any],
    local_residual: np.ndarray,
    width_residual: np.ndarray,
    width: np.ndarray,
    median: np.ndarray,
    truth: np.ndarray,
    coverage: dict[str, Any],
    weight: np.ndarray,
) -> None:
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    shell_centres = np.asarray([0.20, 0.30, 0.40, 0.50])
    for component, (label, color) in enumerate(zip(EIGEN_LABELS, COLORS, strict=True)):
        values = [shell_width[str(shell)]["weighted_median_half_width"][component] for shell in range(4)]
        axes[0, 0].plot(shell_centres, values, marker="o", color=color, label=label)
    axes[0, 0].set(
        title="Posterior width grows as the tracer sample becomes sparse",
        xlabel="Redshift-shell centre",
        ylabel="Weighted median 68% half-width",
    )
    axes[0, 0].legend()

    local_q = weighted_quantile_bin(local_residual, weight, 4)
    for component, (label, color) in enumerate(zip(EIGEN_LABELS, COLORS, strict=True)):
        values = [
            100.0 * weighted_mean(width_residual[local_q == q, component], weight[local_q == q])
            for q in range(4)
        ]
        axes[0, 1].plot(np.arange(1, 5), values, marker="o", color=color, label=label)
    axes[0, 1].axhline(0.0, color=TEXT_COLOR, alpha=0.35, linewidth=1)
    axes[0, 1].set(
        title="Local-count effect after shell and environment control",
        xlabel="Within-cell BRIGHT neighbour-count quartile",
        ylabel="100 x weighted mean residual log-width",
        xticks=np.arange(1, 5),
    )
    axes[0, 1].legend()

    for component, (label, color) in enumerate(zip(EIGEN_LABELS, COLORS, strict=True)):
        group = weighted_quantile_bin(width[:, component], weight, 4)
        rmse = [
            np.sqrt(
                weighted_mean(
                    np.square(median[group == q, component] - truth[group == q, component]),
                    weight[group == q],
                )
            )
            for q in range(4)
        ]
        axes[1, 0].plot(np.arange(1, 5), rmse, marker="o", color=color, label=label)
    axes[1, 0].set(
        title="Wider posteriors identify harder galaxies",
        xlabel="Posterior-width quartile",
        ylabel="Weighted posterior-median RMSE",
        xticks=np.arange(1, 5),
    )
    axes[1, 0].legend()

    for component, (label, color) in enumerate(zip(EIGEN_LABELS, COLORS, strict=True)):
        values = [coverage[str(shell)]["coverage68"][component] for shell in range(4)]
        axes[1, 1].plot(shell_centres, values, marker="o", color=color, label=label)
    axes[1, 1].axhline(0.68, color=TEXT_COLOR, linestyle="--", alpha=0.65, label="Nominal 68%")
    axes[1, 1].set(
        title="Mild conditional failure is confined to sparse-shell gaps",
        xlabel="Redshift-shell centre",
        ylabel="Weighted empirical 68% coverage",
        ylim=(0.62, 0.72),
    )
    axes[1, 1].legend(ncol=2)
    fig.suptitle("Does P12-A uncertainty respond to available information?", fontsize=20)
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_examples(
    output: Path,
    eigen_samples: np.ndarray,
    truth: np.ndarray,
    examples: list[dict[str, Any]],
    redshift: np.ndarray,
    ntilde: np.ndarray,
    boundary: np.ndarray,
) -> None:
    apply_style()
    fig, axes = plt.subplots(4, 3, figsize=(15, 14), constrained_layout=True)
    for row, example in enumerate(examples):
        index = int(example["evaluation_row"])
        for component in range(3):
            ax = axes[row, component]
            values = eigen_samples[index, :, component]
            grid = np.linspace(np.quantile(values, 0.005), np.quantile(values, 0.995), 250)
            density = gaussian_kde(values)(grid)
            q05, q16, q50, q84, q95 = np.quantile(values, [0.05, 0.16, 0.5, 0.84, 0.95])
            ax.fill_between(grid, density, where=(grid >= q05) & (grid <= q95), color=COLORS[component], alpha=0.15)
            ax.fill_between(grid, density, where=(grid >= q16) & (grid <= q84), color=COLORS[component], alpha=0.35)
            ax.plot(grid, density, color=COLORS[component], linewidth=2, label="Posterior")
            ax.axvline(q50, color=TEXT_COLOR, linestyle="--", linewidth=1.2, label="Median")
            ax.axvline(truth[index, component], color="#D62828", linewidth=2, label="Truth")
            ax.set_yticks([])
            if row == 0:
                ax.set_title(EIGEN_LABELS[component])
            if row == 3:
                ax.set_xlabel("Physical eigenvalue")
            if component == 0:
                readable = example["label"].replace("_", " ")
                ax.set_ylabel(
                    f"{readable}\n"
                    f"z={redshift[index]:.3f}, N20={example['neighbors_r20']}\n"
                    f"ntilde={ntilde[index]:.2e}, edge={boundary[index]:.1f} Mpc"
                )
            if row == 0 and component == 2:
                ax.legend(loc="upper right", fontsize=9)
    fig.suptitle(
        "Representative P12-A posteriors: adaptive widening and sparse-shell tail misses",
        fontsize=20,
    )
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_sparse_sky(
    output: Path,
    xyz: np.ndarray,
    shell: np.ndarray,
    intervals: np.ndarray,
    truth: np.ndarray,
) -> None:
    apply_style()
    radius = np.linalg.norm(xyz, axis=1)
    ra = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])) % 360.0
    dec = np.degrees(np.arcsin(np.clip(xyz[:, 2] / radius, -1.0, 1.0)))
    sparse = shell == 3
    lambda2_low = truth[:, 1] < intervals[:, 1, 1]
    lambda3_low = truth[:, 2] < intervals[:, 1, 2]
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    ax.scatter(ra[sparse], dec[sparse], s=5, color="#7A7A7A", alpha=0.25, label="Sparse-shell evaluation rows")
    only2 = sparse & lambda2_low & ~lambda3_low
    only3 = sparse & lambda3_low & ~lambda2_low
    both = sparse & lambda2_low & lambda3_low
    ax.scatter(ra[only2], dec[only2], s=12, color=COLORS[1], alpha=0.8, label=r"Truth below $\lambda_2$ 68% interval")
    ax.scatter(ra[only3], dec[only3], s=12, color=COLORS[2], alpha=0.8, label=r"Truth below $\lambda_3$ 68% interval")
    ax.scatter(ra[both], dec[both], s=18, color="#D62828", alpha=0.9, label="Both lower-tail misses")
    ax.set(
        title="Where the mild sparse-shell high-location residual appears",
        xlabel="Right ascension [deg]",
        ylabel="Declination [deg]",
        xlim=(0, 360),
    )
    ax.legend(ncol=2, loc="upper center")
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT)
    parser.add_argument("--audit-root", type=Path, default=AUDIT)
    parser.add_argument("--points", type=Path, default=POINTS)
    parser.add_argument("--adapter-root", type=Path, default=ADAPTER)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    ready_path = args.dataset_root / "P12A_DATASET_READY.json"
    audit_path = args.audit_root / "P12A_CALIBRATION_AUDIT.json"
    checkpoint_path = args.dataset_root / "fmpe_seed42" / "fmpe_estimator.pt"
    ready = json.loads(ready_path.read_text())
    audit = json.loads(audit_path.read_text())
    if ready.get("sealed_phase_opened") or audit.get("sealed_phase_opened"):
        raise RuntimeError("sealed ph001 guard failed")
    if ready["validation_phase"] != "ph006" or audit["selection_phase"] != "ph006":
        raise RuntimeError("diagnostic is frozen to ph006")

    validation = np.load(ready["validation"]["path"])
    evaluation_index = np.load(args.audit_root / "evaluation_index.npy")
    samples_scaled = np.load(args.audit_root / "evaluation_samples_scaled.npy", mmap_mode="r")
    checkpoint = __import__("torch").load(checkpoint_path, map_location="cpu", weights_only=False)
    theta_mean = np.asarray(checkpoint["theta_mean"], dtype=np.float64)
    theta_std = np.asarray(checkpoint["theta_std"], dtype=np.float64)
    theta_samples = np.asarray(samples_scaled, dtype=np.float64) * theta_std + theta_mean
    eigen_samples = theta_to_eigenvalues(theta_samples)
    del theta_samples

    truth = np.asarray(validation["truth_eigenvalues"])[evaluation_index]
    base = np.asarray(validation["base_prediction_eigenvalues"])[evaluation_index]
    context = np.asarray(validation["context"])[evaluation_index]
    parent = np.asarray(validation["parent_node_id"])[evaluation_index].astype(np.int64)
    shell = np.asarray(validation["shell"])[evaluation_index].astype(np.int8)
    cap = np.asarray(validation["cap"])[evaluation_index].astype(np.int8)
    superblock = np.asarray(validation["superblock_id"])[evaluation_index].astype(np.int64)
    weight = np.asarray(validation["natural_weight"])[evaluation_index].astype(np.float64)
    spatial_group = (cap.astype(np.int64) << 32) + superblock
    local, target_xyz = local_bright_information(
        args.points,
        args.adapter_root / "core_active_parent.npy",
        args.adapter_root / "core_active_offsets.npy",
        args.adapter_root / "core_cap.npy",
        parent,
        cap,
        args.workers,
    )

    width, median, intervals = interval_widths(eigen_samples)
    pit = randomized_pit(eigen_samples, truth, args.seed + 1)
    shell_width = shell_width_report(width, shell, weight)
    coverage = coverage_by_shell(intervals, truth, shell, weight)
    error_width = error_by_width_quartile(width, median, truth, weight)
    controlled, local_residual, width_residual = controlled_information_report(
        width,
        base,
        shell,
        weight,
        spatial_group,
        local,
        context[:, 4],
        context[:, 6],
        args.bootstrap_repeats,
        args.seed + 100,
    )
    examples = choose_examples(truth, width, median, intervals, pit, shell, local)
    for example in examples:
        index = int(example["evaluation_row"])
        example.update(
            {
                "validation_row": int(evaluation_index[index]),
                "parent_node_id": int(parent[index]),
                "redshift": float(context[index, 3]),
                "ntilde_mpc3": float(np.exp(context[index, 4])),
                "boundary_distance_mpc": float(np.expm1(context[index, 6])),
                "neighbors_r7": int(local["neighbors_r7"][index]),
                "neighbors_r10": int(local["neighbors_r10"][index]),
                "fifth_neighbor_distance_mpc": float(local["fifth_neighbor_distance"][index]),
            }
        )

    report = {
        "schema_version": "p12a-width-information-diagnostic-v1",
        "created_utc": utc_now(),
        "purpose": "detect adaptive posterior width; illustrative examples are not calibration fitting",
        "selection_phase": "ph006",
        "evaluation_folds": [2, 3, 4],
        "evaluation_rows": int(len(evaluation_index)),
        "posterior_draws_per_row": int(eigen_samples.shape[1]),
        "sealed_phase": "ph001",
        "sealed_phase_opened": False,
        "conditioning_limit": (
            "P12-A sees base eigenvalue predictions, redshift, ntilde, cap and random-support boundary; "
            "realised local neighbour counts are external diagnostics, not posterior inputs"
        ),
        "information_proxies": {
            "response": ["redshift", "ntilde_mpc3", "random_support_boundary_distance_mpc"],
            "realised_local_configuration": [
                "BRIGHT neighbour counts within 7/10/20 Mpc/h",
                "fifth-neighbour distance",
            ],
        },
        "unit_contract": {
            "stored_points": "comoving Mpc",
            "neighbour_query_coordinates": "comoving Mpc/h",
            "planck18_h": PLANCK18_H,
            "posterior_width": "physical eigenvalue central-68-percent half-width",
        },
        "shell_width": shell_width,
        "coverage_by_shell": coverage,
        "error_by_width_quartile": error_width,
        "controlled_information_correlations": controlled,
        "examples": examples,
        "provenance": {
            "dataset_marker": str(ready_path),
            "dataset_marker_sha256": sha256(ready_path),
            "audit_report": str(audit_path),
            "audit_report_sha256": sha256(audit_path),
            "evaluation_index": str(args.audit_root / "evaluation_index.npy"),
            "evaluation_index_sha256": sha256(args.audit_root / "evaluation_index.npy"),
            "posterior_samples": str(args.audit_root / "evaluation_samples_scaled.npy"),
            "posterior_samples_sha256": sha256(args.audit_root / "evaluation_samples_scaled.npy"),
            "points": str(args.points),
            "points_sha256": sha256(args.points),
            "active_parent_sha256": sha256(args.adapter_root / "core_active_parent.npy"),
            "core_cap_sha256": sha256(args.adapter_root / "core_cap.npy"),
            "core_active_offsets_sha256": sha256(
                args.adapter_root / "core_active_offsets.npy"
            ),
        },
    }
    atomic_json(args.output_root / "P12A_WIDTH_INFORMATION_DIAGNOSTIC.json", report)
    np.savez_compressed(
        args.output_root / "p12a_width_information_examples.npz",
        evaluation_index=evaluation_index,
        parent_node_id=parent,
        shell=shell,
        cap=cap,
        truth=truth,
        posterior_median=median,
        posterior_half_width=width,
        posterior_pit=pit,
        neighbors_r7=local["neighbors_r7"],
        neighbors_r10=local["neighbors_r10"],
        neighbors_r20=local["neighbors_r20"],
        fifth_neighbor_distance=local["fifth_neighbor_distance"],
    )
    plot_summary(
        args.output_root / "p12a_width_information_summary",
        shell_width,
        local_residual,
        width_residual,
        width,
        median,
        truth,
        coverage,
        weight,
    )
    plot_examples(
        args.output_root / "p12a_posterior_examples",
        eigen_samples,
        truth,
        examples,
        context[:, 3],
        np.exp(context[:, 4]),
        np.expm1(context[:, 6]),
    )
    plot_sparse_sky(
        args.output_root / "p12a_sparse_shell_failure_sky",
        target_xyz,
        shell,
        intervals,
        truth,
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
