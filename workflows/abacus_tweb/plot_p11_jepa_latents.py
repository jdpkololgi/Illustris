#!/usr/bin/env python3
"""Plot a P11 JEPA checkpoint series in one frozen teacher-space PCA basis.

The PCA basis and displayed sample IDs are fitted once from the reference checkpoint
and saved to ``--projection-state``.  Later invocations reuse that exact state, making
apparent epoch-to-epoch drift meaningful.  Student points are shown after the learned
predictor when available, otherwise after a probe-fit-only Procrustes map; arrows point
from the mapped degraded view to the paired dense view.  Native latent axes are not
assumed to have a physical coordinate-wise ordering.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from workflows.abacus_tweb.p11_jepa_latent_diagnostics import (
    LatentSnapshot,
    _validate_series,
    load_latent_snapshot,
    procrustes_crossfit,
)


@dataclass(frozen=True)
class ProjectionState:
    mean: np.ndarray
    components: np.ndarray
    sample_id: np.ndarray
    reference_global_step: int
    reference_run_id: str

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values) - self.mean) @ self.components.T


def _stable_sample_ids(sample_id: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(sample_id) <= max_points:
        return np.asarray(sample_id)
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(sample_id), size=max_points, replace=False))
    return np.asarray(sample_id)[chosen]


def fit_reference_projection(
    snapshot: LatentSnapshot, *, max_points: int = 800, seed: int = 1729
) -> ProjectionState:
    sample_id = _stable_sample_ids(snapshot.sample_id, max_points, seed)
    index = {identifier: row for row, identifier in enumerate(snapshot.sample_id.tolist())}
    rows = np.asarray([index[identifier] for identifier in sample_id.tolist()])
    # Teacher coordinates are the reference space.  Do not refit on later students.
    dense = snapshot.dense_latent[rows]
    mean = dense.mean(axis=0)
    _, _, vt = np.linalg.svd(dense - mean, full_matrices=False)
    components = vt[:2]
    return ProjectionState(
        mean=mean,
        components=components,
        sample_id=sample_id,
        reference_global_step=int(snapshot.metadata["global_step"]),
        reference_run_id=str(snapshot.metadata["run_id"]),
    )


def save_projection_state(path: Path, state: ProjectionState) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            mean=state.mean,
            components=state.components,
            sample_id=state.sample_id,
            reference_global_step=np.asarray(state.reference_global_step),
            reference_run_id=np.asarray(state.reference_run_id),
        )
    temporary.replace(path)


def load_projection_state(path: Path) -> ProjectionState:
    with np.load(path, allow_pickle=False) as data:
        return ProjectionState(
            mean=np.asarray(data["mean"]),
            components=np.asarray(data["components"]),
            sample_id=np.asarray(data["sample_id"]),
            reference_global_step=int(np.asarray(data["reference_global_step"]).item()),
            reference_run_id=str(np.asarray(data["reference_run_id"]).item()),
        )


def mapped_student(snapshot: LatentSnapshot) -> np.ndarray:
    predictor_trained = bool(
        snapshot.metadata["arm"] == "jepa"
        and snapshot.metadata.get("predictor_trained", False)
    )
    if snapshot.predicted_dense_latent is not None and predictor_trained:
        return snapshot.predicted_dense_latent
    _, mapped = procrustes_crossfit(snapshot)
    return mapped


def _selected_rows(snapshot: LatentSnapshot, state: ProjectionState) -> np.ndarray:
    if str(snapshot.metadata["run_id"]) != state.reference_run_id:
        raise ValueError("projection state belongs to a different run_id")
    index = {identifier: row for row, identifier in enumerate(snapshot.sample_id.tolist())}
    try:
        return np.asarray([index[identifier] for identifier in state.sample_id.tolist()])
    except KeyError as error:
        raise ValueError("projection sample IDs are absent from this checkpoint") from error


def render(
    snapshots: list[LatentSnapshot],
    summary: dict,
    state: ProjectionState,
    output: Path,
    *,
    arrow_points: int = 80,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    checkpoints = summary["checkpoints"]
    if len(checkpoints) != len(snapshots):
        raise ValueError("summary checkpoint count does not match snapshots")
    display = np.unique(
        np.linspace(0, len(snapshots) - 1, min(4, len(snapshots)), dtype=int)
    )
    figure = plt.figure(figsize=(4.6 * len(display), 8.2), constrained_layout=True)
    grid = figure.add_gridspec(2, len(display), height_ratios=(0.75, 1.25))
    metric_axis = figure.add_subplot(grid[0, :])
    steps = np.asarray([row["global_step"] for row in checkpoints])
    cka = np.asarray([
        row["predictor"]["cka_to_dense"]
        if row["predictor"] is not None
        else row["linear_cka"]["native_student_to_dense"]
        for row in checkpoints
    ])
    cka_shuffle = np.asarray([
        row["linear_cka"]["response_matched_shuffled_control"]
        for row in checkpoints
    ])
    rank = np.asarray([
        row["spread"]["degraded_student"]["effective_rank_fraction"]
        for row in checkpoints
    ])
    retrieval = np.asarray([
        row["cross_view_retrieval"]["mean_reciprocal_rank"] for row in checkpoints
    ])
    metric_axis.plot(steps, cka, "o-", label="paired linear CKA")
    metric_axis.plot(
        steps, cka_shuffle, "o--", label="response-matched shuffled CKA"
    )
    metric_axis.plot(steps, rank, "s-", label="student effective-rank fraction")
    metric_axis.plot(steps, retrieval, "^-", label="retrieval MRR")
    if checkpoints[0]["downstream_linear_probe"] is not None:
        probe = np.asarray([
            row["downstream_linear_probe"]["degraded_student"]["macro_r2"]
            for row in checkpoints
        ])
        metric_axis.plot(steps, probe, "d-", label="held-out student probe $R^2$")
    metric_axis.set(xlabel="global optimizer step", ylabel="diagnostic value", ylim=(-0.05, 1.05))
    metric_axis.grid(alpha=0.2)
    metric_axis.legend(ncol=3, fontsize=9)

    color_limits = (np.inf, -np.inf)
    projected = []
    for position in display:
        snapshot = snapshots[position]
        rows = _selected_rows(snapshot, state)
        dense = state.transform(snapshot.dense_latent[rows])
        student = state.transform(mapped_student(snapshot)[rows])
        response = snapshot.response_strength[rows]
        projected.append((position, snapshot, dense, student, response))
        color_limits = (min(color_limits[0], response.min()), max(color_limits[1], response.max()))
    scatter = None
    for column, (position, snapshot, dense, student, response) in enumerate(projected):
        axis = figure.add_subplot(grid[1, column])
        scatter = axis.scatter(
            student[:, 0], student[:, 1], c=response, s=8, alpha=0.55,
            cmap="viridis", vmin=color_limits[0], vmax=color_limits[1], label="degraded mapped"
        )
        axis.scatter(dense[:, 0], dense[:, 1], c="black", s=6, alpha=0.28, label="dense teacher")
        arrow_rows = np.linspace(0, len(dense) - 1, min(arrow_points, len(dense)), dtype=int)
        for row in arrow_rows:
            axis.annotate(
                "", xy=dense[row], xytext=student[row],
                arrowprops={"arrowstyle": "->", "lw": 0.45, "alpha": 0.28, "color": "tab:red"},
            )
        metric = checkpoints[position]
        axis.set_title(
            f"epoch {snapshot.metadata['epoch']} / step {snapshot.metadata['global_step']}\n"
            f"CKA={cka[position]:.3f}, rank={rank[position]:.2f}, MRR={retrieval[position]:.3f}"
        )
        axis.set(xlabel="fixed teacher PCA 1", ylabel="fixed teacher PCA 2")
        axis.grid(alpha=0.12)
        if column == 0:
            axis.legend(fontsize=8, loc="best")
    if scatter is not None:
        figure.colorbar(
            scatter,
            ax=[figure.axes[index + 1] for index in range(len(display))],
            label="response strength",
            shrink=0.72,
        )
    figure.suptitle(
        "P11 paired-view latent evolution — fixed reference projection\n"
        "arrows: mapped degraded view → paired dense view (not posterior uncertainty)",
        fontsize=13,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, nargs="+", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--projection-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-index", type=int, default=0)
    parser.add_argument("--max-points", type=int, default=800)
    parser.add_argument("--arrow-points", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    snapshots = _validate_series(load_latent_snapshot(path) for path in args.snapshots)
    if args.projection_state.exists():
        state = load_projection_state(args.projection_state)
    else:
        state = fit_reference_projection(
            snapshots[args.reference_index], max_points=args.max_points, seed=args.seed
        )
        save_projection_state(args.projection_state, state)
    import json

    summary = json.loads(args.summary.read_text())
    render(snapshots, summary, state, args.output, arrow_points=args.arrow_points)
    print(args.output)


if __name__ == "__main__":
    main()
