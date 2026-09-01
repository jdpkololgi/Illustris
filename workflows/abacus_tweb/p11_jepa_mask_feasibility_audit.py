#!/usr/bin/env python3
"""Audit the production P11 JEPA-v2 exact-M mask without opening ph001.

This is a diagnostic, not a training-contract mutator.  Mask construction,
minimum-count handling, and pool-aligned target propagation are imported from
``p11_jepa_canary`` so the audit cannot silently diverge from the trainer.  The
historical nearest-neighbour resize is retained only as a labelled rejected
counterfactual.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np

from workflows.abacus_tweb.p11_jepa_canary import (
    auxiliary_cluster_mask,
    deterministic_cluster_mask,
    load_contract,
)


VISIBLE_PHASES = ("ph002", "ph003", "ph004", "ph005", "ph006")
SEALED_PHASE = "ph001"
HALO_VOXELS = 24
ALIGNMENT_VOXELS = 8
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = REPO_ROOT / "configs/p11_paired_degrade_jepa_v2.json"


def quantiles(values: list[float] | np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {str(q): None for q in (0, 0.01, 0.05, 0.5, 0.95, 0.99, 1)}
    return {
        str(q): float(np.quantile(array, q))
        for q in (0, 0.01, 0.05, 0.5, 0.95, 0.99, 1)
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def production_mask_record(
    exact_core_support: np.ndarray,
    *,
    core_start: np.ndarray,
    context_start: np.ndarray,
    context_stop: np.ndarray,
    seed: int,
    epoch: int,
    phase_index: int,
    core_id: int,
    mask_spec: dict,
) -> dict:
    """Run exactly the trainer's mask policy on one embedded core.

    ``exact_core_support`` is embedded at its true offset in the aligned context
    lattice before calling the production implementation.  The returned
    ``target_mask`` is the actual intervention seen by training: it is all-zero
    for a registered auxiliary-invalid example.
    """
    exact_core_support = np.asarray(exact_core_support, dtype=bool)
    context_shape = tuple(
        int(value) for value in np.asarray(context_stop) - np.asarray(context_start)
    )
    eligible = np.zeros(context_shape, dtype=bool)
    offset = np.asarray(core_start, dtype=np.int64) - np.asarray(
        context_start, dtype=np.int64
    )
    slices = tuple(
        slice(int(left), int(left + width))
        for left, width in zip(offset, exact_core_support.shape)
    )
    eligible[slices] = exact_core_support

    # The rejected nearest-resize diagnostic is evaluated on the same stable,
    # production-selected proposal whenever the registered full-resolution
    # minima permit one.  It never determines production validity.
    try:
        proposal = deterministic_cluster_mask(
            eligible,
            seed=seed,
            epoch=epoch,
            phase_index=phase_index,
            core_id=core_id,
            mask_spec=mask_spec,
        )
    except RuntimeError:
        proposal = np.zeros_like(eligible, dtype=bool)
    proposal_core = proposal[slices]
    nearest_count = nearest_bottleneck_count(
        proposal_core, core_start, context_start, context_stop
    )

    target_mask, valid, reason, aligned = auxiliary_cluster_mask(
        eligible,
        seed=seed,
        epoch=epoch,
        phase_index=phase_index,
        core_id=core_id,
        mask_spec=mask_spec,
    )
    if np.any(target_mask & ~eligible):
        raise RuntimeError("production audit selected an unsupported target")
    if valid != bool(np.any(target_mask)):
        raise RuntimeError("production auxiliary-valid flag and intervention disagree")
    return {
        "eligible_voxels": int(eligible.sum()),
        "proposal_target_voxels": int(proposal.sum()),
        "nearest_resize_bottleneck_targets_rejected": int(nearest_count),
        "production_target_voxels": int(target_mask.sum()),
        "production_full_resolution_targets": int(
            aligned["full_resolution_targets"]
        ),
        "production_bottleneck_targets": int(aligned["bottleneck_targets"]),
        "auxiliary_valid": bool(valid),
        "auxiliary_invalid_reason": reason,
    }


def context_bounds(
    core_start: np.ndarray, core_stop: np.ndarray, field_shape: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    requested_start = np.asarray(core_start, dtype=np.int64) - HALO_VOXELS
    requested_stop = np.asarray(core_stop, dtype=np.int64) + HALO_VOXELS
    requested_start = (requested_start // ALIGNMENT_VOXELS) * ALIGNMENT_VOXELS
    requested_stop = (
        (requested_stop + ALIGNMENT_VOXELS - 1) // ALIGNMENT_VOXELS
    ) * ALIGNMENT_VOXELS
    return (
        np.maximum(requested_start, 0),
        np.minimum(requested_stop, np.asarray(field_shape, dtype=np.int64)),
    )


def nearest_bottleneck_count(
    core_mask: np.ndarray,
    core_start: np.ndarray,
    context_start: np.ndarray,
    context_stop: np.ndarray,
) -> int:
    """Match ``F.interpolate(..., mode='nearest')`` after three ceil-mode pools."""
    context_shape = np.asarray(context_stop - context_start, dtype=np.int64)
    output_shape = np.asarray(
        [int(math.ceil(int(value) / 8.0)) for value in context_shape],
        dtype=np.int64,
    )
    source = [
        np.floor(np.arange(out) * int(size) / out).astype(np.int64)
        for size, out in zip(context_shape, output_shape)
    ]
    selected = np.zeros(tuple(int(v) for v in context_shape), dtype=bool)
    offset = np.asarray(core_start, dtype=np.int64) - context_start
    slices = tuple(
        slice(int(left), int(left + width))
        for left, width in zip(offset, core_mask.shape)
    )
    selected[slices] = core_mask
    return int(selected[np.ix_(*source)].sum())


def selected_ids(path: Path, phase: str, limit: int) -> np.ndarray:
    name = "validation_core_id.npy" if phase == "ph006" else "training_core_id.npy"
    ids = np.sort(np.asarray(np.load(path / "phases" / phase / name), dtype=np.int64))
    if limit and limit < len(ids):
        positions = np.linspace(0, len(ids) - 1, limit, dtype=np.int64)
        ids = ids[np.unique(positions)]
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/global/homes/d/dkololgi/p11_contracts/"
            "training_contract_r1_random_repair_v2_20260901"
        ),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--epochs", default="1")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    epochs = tuple(int(value) for value in args.epochs.split(",") if value)
    if not epochs or any(value <= 0 for value in epochs):
        parser.error("--epochs must contain positive comma-separated integers")
    contract = load_contract(args.contract)
    mask_spec = contract["masking"]
    if mask_spec.get("strategy") != "exact_support_compact_clusters_v2":
        parser.error("--contract must use exact_support_compact_clusters_v2")
    report = {
        "schema_version": "p11-exact-m-cluster-mask-feasibility-v2",
        "root": str(args.root),
        "contract": str(args.contract),
        "contract_sha256": sha256(args.contract),
        "visible_phases": list(VISIBLE_PHASES),
        "sealed_phase": SEALED_PHASE,
        "sealed_phase_opened": False,
        "support_field": "support_random == 1",
        "mask": {
            "strategy": mask_spec["strategy"],
            "fraction_of_exact_support": mask_spec["registered_target_fraction"],
            "clusters": mask_spec["clusters"],
            "minimum_eligible_voxels": mask_spec["minimum_eligible_voxels"],
            "minimum_target_voxels": mask_spec["minimum_target_voxels"],
            "minimum_context_voxels": mask_spec["minimum_context_voxels"],
            "downsample_policy": mask_spec["downsample_policy"],
            "implementation": (
                "p11_jepa_canary.auxiliary_cluster_mask and "
                "p11_jepa_canary.deterministic_cluster_mask"
            ),
        },
        "epochs": list(epochs),
        "limit_per_phase": args.limit,
        "phases": {},
    }
    for phase_index, phase in enumerate(VISIBLE_PHASES):
        ids = selected_ids(args.root, phase, args.limit)
        adapter = args.root / "adapters" / phase / "field"
        manifest = json.loads((adapter / "adapter_manifest.json").read_text())
        caps = np.load(adapter / "core_cap.npy", mmap_mode="r")
        starts = np.load(adapter / "core_voxel_start.npy", mmap_mode="r")
        stops = np.load(adapter / "core_voxel_stop.npy", mmap_mode="r")
        support_counts: list[int] = []
        proposal_target_counts = {epoch: [] for epoch in epochs}
        production_target_counts = {epoch: [] for epoch in epochs}
        nearest_bottleneck_counts = {epoch: [] for epoch in epochs}
        production_bottleneck_counts = {epoch: [] for epoch in epochs}
        proposal_coverages = {epoch: [] for epoch in epochs}
        auxiliary_valid = {epoch: [] for epoch in epochs}
        invalid_reason_counts = {epoch: {} for epoch in epochs}
        invalid = {epoch: [] for epoch in epochs}
        for cap_id, cap_name in ((0, "SGC"), (1, "NGC")):
            cap_ids = ids[np.asarray(caps[ids]) == cap_id]
            if not len(cap_ids):
                continue
            path = Path(manifest["caps"][cap_name]["field_path"])
            with h5py.File(path, "r") as handle:
                dataset = handle["support_random"]
                field_shape = tuple(int(value) for value in dataset.shape)
                for core_id in cap_ids:
                    core_id = int(core_id)
                    core_slices = tuple(
                        slice(int(left), int(right))
                        for left, right in zip(starts[core_id], stops[core_id])
                    )
                    exact_m = np.asarray(dataset[core_slices], dtype=bool)
                    count = int(exact_m.sum())
                    support_counts.append(count)
                    cstart, cstop = context_bounds(
                        starts[core_id], stops[core_id], field_shape
                    )
                    for epoch in epochs:
                        record = production_mask_record(
                            exact_m,
                            core_start=starts[core_id],
                            context_start=cstart,
                            context_stop=cstop,
                            seed=args.seed,
                            epoch=epoch,
                            phase_index=phase_index,
                            core_id=core_id,
                            mask_spec=mask_spec,
                        )
                        proposal_target = int(record["proposal_target_voxels"])
                        production_target = int(record["production_target_voxels"])
                        nearest_count = int(
                            record["nearest_resize_bottleneck_targets_rejected"]
                        )
                        production_bottleneck = int(
                            record["production_bottleneck_targets"]
                        )
                        is_valid = bool(record["auxiliary_valid"])
                        reason = record["auxiliary_invalid_reason"]
                        proposal_target_counts[epoch].append(proposal_target)
                        production_target_counts[epoch].append(production_target)
                        nearest_bottleneck_counts[epoch].append(nearest_count)
                        production_bottleneck_counts[epoch].append(
                            production_bottleneck
                        )
                        proposal_coverages[epoch].append(
                            proposal_target / count if count else 0.0
                        )
                        auxiliary_valid[epoch].append(is_valid)
                        if not is_valid:
                            reason_name = str(reason)
                            invalid_reason_counts[epoch][reason_name] = (
                                invalid_reason_counts[epoch].get(reason_name, 0) + 1
                            )
                            if len(invalid[epoch]) < 30:
                                invalid[epoch].append(
                                    {
                                        "core_id": core_id,
                                        "cap": cap_name,
                                        "support": count,
                                        **record,
                                    }
                                )
        support_array = np.asarray(support_counts, dtype=np.int64)
        phase_report = {
            "cores": int(len(ids)),
            "exact_support_count": quantiles(support_array),
            "support_threshold_fraction": {
                str(threshold): float(np.mean(support_array >= threshold))
                for threshold in (1, 8, 16, 32)
            },
            "epochs": {},
        }
        for epoch in epochs:
            proposal = np.asarray(proposal_target_counts[epoch], dtype=np.int64)
            intervention = np.asarray(production_target_counts[epoch], dtype=np.int64)
            nearest = np.asarray(nearest_bottleneck_counts[epoch], dtype=np.int64)
            pooled = np.asarray(production_bottleneck_counts[epoch], dtype=np.int64)
            valid = np.asarray(auxiliary_valid[epoch], dtype=bool)
            nearest_counterfactual_valid = (proposal >= 2) & (nearest >= 2)
            phase_report["epochs"][str(epoch)] = {
                "proposal_target_count": quantiles(proposal),
                "production_intervention_target_count": quantiles(intervention),
                "proposal_target_coverage": quantiles(proposal_coverages[epoch]),
                "nearest_resize_bottleneck_target_count_rejected": quantiles(nearest),
                "production_pool_aligned_bottleneck_target_count": quantiles(pooled),
                "nearest_resize_aux_valid_fraction_rejected": float(
                    np.mean(nearest_counterfactual_valid)
                ),
                "nearest_resize_aux_invalid_fraction_rejected": float(
                    np.mean(~nearest_counterfactual_valid)
                ),
                "production_aux_valid_fraction": float(np.mean(valid)),
                "production_aux_invalid_fraction": float(np.mean(~valid)),
                "production_aux_invalid_cores": int(np.count_nonzero(~valid)),
                "production_aux_invalid_reason_counts": invalid_reason_counts[epoch],
                "first_production_aux_invalid": invalid[epoch],
            }
        report["phases"][phase] = phase_report
        print(phase, json.dumps(phase_report), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
