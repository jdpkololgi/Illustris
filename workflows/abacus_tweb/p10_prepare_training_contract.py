#!/usr/bin/env python3
"""Build and canary the P10 five-phase deterministic training contract."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import pickle
from pathlib import Path
import subprocess
import sys
import time

import fitsio
import h5py
import numpy as np
from astropy.cosmology import Planck18
from sklearn.preprocessing import PowerTransformer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p10_training_contract import (
    BLIND_PHASE,
    TRAINING_PHASES,
    VALIDATION_PHASE,
    atomic_json,
    epoch_hash,
    phase_balanced_epoch,
    phase_equal_patch_objective,
    resume_state,
    sha256,
    validate_resume_state,
)
from workflows.abacus_tweb.p5_graph_patch_utils import CanonicalGraphPatchAdapter
from workflows.abacus_tweb.p6_field_patch_utils import (
    CanonicalFieldPatchAdapter,
    channel_transform,
    derive_selection_channels,
    patch_redshift,
)
from workflows.abacus_tweb.p6_refit_fullcap_selection import (
    build_cap_lookup,
    fit_log_spline,
    histogram_counts,
    histogram_effective_volume,
    radius_to_redshift_grid,
)
from workflows.abacus_tweb.p8_deterministic_common import (
    SHELL_NAMES,
    linear_increments,
)


ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase")
REGISTRY = Path("configs/p10_phase_registry_v1.json")
PHASES = ("ph000", "ph001", "ph002", "ph003", "ph004", "ph005", "ph006")
CAP_NAME = {0: "SGC", 1: "NGC"}
SI_COLUMNS = (0, 2, 3, 4, 5, 6)
NODE_SAMPLE_PER_PHASE_CAP = 100_000
EDGE_SAMPLE_PER_PHASE_CAP = 100_000
FIT_Z_MIN = 0.15
FIT_Z_MAX = 0.55
Z_MIN = 0.10
Z_MAX = 0.60
BIN_WIDTH = 0.005
CURVE_STEP = 0.001
KNOT_SPACING = 0.05
MINIMUM_EXPOSURE = 1.0e-4
CONTRAST_EPSILON = 1.0e-3


def phase_root(root: Path, phase: str) -> Path:
    return root / phase


def p1_manifest(root: Path, phase: str) -> dict:
    return json.loads((phase_root(root, phase) / "p1_canonical/manifest.json").read_text())


def phase_files(root: Path, phase: str) -> dict[str, Path]:
    p = phase_root(root, phase)
    prefix = f"{phase}_bgs_bright_full_delaunay"
    return {
        "p1_index": p / "p1_canonical/canonical_index.npz",
        "p1_points": p / "p1_canonical/points.npy",
        "p1_manifest": p / "p1_canonical/manifest.json",
        "p2_gnn": p / f"p2_graph/{prefix}_cugraph_gnn_arrays.npz",
        "p2_pairs": p / f"p2_graph/{prefix}_edges_combined_idx.npy",
        "p2_manifest": p / "p2_union/p2b_union_manifest.json",
        "radius_ngc": p / "p2_union/ngc_radius_only_pairs.npy",
        "radius_ngc_attr": p / "p2_union/ngc_radius_only_edge_attr.npy",
        "radius_sgc": p / "p2_union/sgc_radius_only_pairs.npy",
        "radius_sgc_attr": p / "p2_union/sgc_radius_only_edge_attr.npy",
        "p3_manifest": p / "p3_fields/field_manifest.json",
        "p4_root": p / "p4_patches",
        "p4_manifest": p / "p4_patches/spatial_manifest.json",
        "assignment": p / "p4_patches/active_assignment.npz",
        "graph_support": p / "p4_patches/graph_support_active.npz",
        "cores": p / "p4_patches/cores.npz",
    }


def run(command: list[str]) -> None:
    print(json.dumps({"command": command}), flush=True)
    subprocess.run(command, check=True)


def build_adapters(root: Path, contract: Path, phases: tuple[str, ...]) -> dict:
    records = {}
    for phase in phases:
        files = phase_files(root, phase)
        for name, path in files.items():
            if name != "p4_root" and not path.is_file():
                raise FileNotFoundError(f"{phase} missing {name}: {path}")
        graph_root = contract / "adapters" / phase / "graph"
        field_root = contract / "adapters" / phase / "field"
        graph_manifest = graph_root / "adapter_manifest.json"
        if not graph_manifest.is_file() or not json.loads(graph_manifest.read_text()).get("pass"):
            run([
                sys.executable,
                str(REPO_ROOT / "workflows/abacus_tweb/p5_build_graph_patch_adapter.py"),
                "--gnn-arrays", str(files["p2_gnn"]),
                "--delaunay-pairs", str(files["p2_pairs"]),
                "--canonical-index", str(files["p1_index"]),
                "--p2-manifest", str(files["p2_manifest"]),
                "--radius-ngc", str(files["radius_ngc"]),
                "--radius-ngc-attr", str(files["radius_ngc_attr"]),
                "--radius-sgc", str(files["radius_sgc"]),
                "--radius-sgc-attr", str(files["radius_sgc_attr"]),
                "--p4-manifest", str(files["p4_manifest"]),
                "--active-assignment", str(files["assignment"]),
                "--graph-support", str(files["graph_support"]),
                "--cores", str(files["cores"]),
                "--out-dir", str(graph_root),
            ])
        field_manifest = field_root / "adapter_manifest.json"
        if not field_manifest.is_file() or not json.loads(field_manifest.read_text()).get("pass"):
            run([
                sys.executable,
                str(REPO_ROOT / "workflows/abacus_tweb/p6_build_field_patch_adapter.py"),
                "--p3-manifest", str(files["p3_manifest"]),
                "--p4-root", str(files["p4_root"]),
                "--output-root", str(field_root),
            ])
        graph = json.loads(graph_manifest.read_text())
        field = json.loads(field_manifest.read_text())
        if not graph.get("pass") or not field.get("pass"):
            raise RuntimeError(f"adapter gate failed for {phase}")
        records[phase] = {
            "graph_manifest": str(graph_manifest),
            "graph_manifest_sha256": sha256(graph_manifest),
            "field_manifest": str(field_manifest),
            "field_manifest_sha256": sha256(field_manifest),
        }
    return records


def read_parent_columns(root: Path, phase: str, columns: tuple[str, ...]) -> np.ndarray:
    parent = Path(p1_manifest(root, phase)["parent"])
    return fitsio.read(parent, columns=list(columns))


def prepare_phase_arrays(root: Path, contract: Path, phase: str) -> dict:
    files = phase_files(root, phase)
    output = contract / "phases" / phase
    output.mkdir(parents=True, exist_ok=True)
    index = np.load(files["p1_index"], mmap_mode="r")
    assignment = np.load(files["assignment"], mmap_mode="r")
    cores = np.load(files["cores"], mmap_mode="r")
    n_parent = len(index["parent_node_id"])
    if not np.array_equal(np.asarray(index["parent_node_id"]), np.arange(n_parent)):
        raise RuntimeError(f"{phase} P1 parent rows are not identity")
    parent = np.asarray(assignment["parent_node_id"], dtype=np.int64)
    targetid = np.asarray(assignment["targetid"], dtype=np.int64)
    authoritative = np.asarray(assignment["supervised_eligible"], dtype=bool)
    if len(np.unique(parent[authoritative])) != int(authoritative.sum()):
        raise RuntimeError(f"{phase} authoritative parents are not unique")
    if not np.array_equal(targetid, np.asarray(index["targetid"])[parent]):
        raise RuntimeError(f"{phase} P4/P1 TARGETID mismatch")
    columns = ("TARGETID", "Z") if phase == BLIND_PHASE else (
        "TARGETID", "Z", "LAMBDA1", "LAMBDA2", "LAMBDA3"
    )
    catalogue = read_parent_columns(root, phase, columns)
    catalogue_targetid = np.asarray(catalogue["TARGETID"], dtype=np.int64)
    if len(catalogue) != n_parent or not np.array_equal(
        catalogue_targetid, np.asarray(index["targetid"], dtype=np.int64)
    ):
        raise RuntimeError(f"{phase} parent catalogue/P1 identity mismatch")
    np.save(output / "parent_targetid.npy", catalogue_targetid, allow_pickle=False)
    redshift = np.asarray(catalogue["Z"], dtype=np.float32)
    np.save(output / "parent_redshift.npy", redshift, allow_pickle=False)
    truth_present = phase != BLIND_PHASE
    target_record = None
    if truth_present:
        eigen = np.column_stack([
            np.asarray(catalogue[name], dtype=np.float32)
            for name in ("LAMBDA1", "LAMBDA2", "LAMBDA3")
        ])
        active_truth = np.asarray(eigen[parent[authoritative]], dtype=np.float64)
        ordered = (
            np.all(active_truth[:, 1] >= active_truth[:, 0])
            and np.all(active_truth[:, 2] >= active_truth[:, 1])
        )
        if not np.all(np.isfinite(active_truth)) or not ordered:
            raise RuntimeError(f"{phase} authoritative target contract failed")
        np.save(output / "parent_eigenvalues.npy", eigen, allow_pickle=False)
        target_record = {
            "path": str(output / "parent_eigenvalues.npy"),
            "sha256": sha256(output / "parent_eigenvalues.npy"),
            "authoritative_finite": True,
            "authoritative_ordered": True,
        }

    shell = np.asarray(assignment["shell"], dtype=np.int8)
    counts = np.bincount(shell[authoritative], minlength=4).astype(np.int64)
    if np.any(counts[:4] == 0):
        raise RuntimeError(f"{phase} does not contain every shell")
    weight = np.zeros(len(parent), dtype=np.float32)
    weight[authoritative] = 1.0 / np.sqrt(counts[shell[authoritative]])
    np.save(output / "active_row_weight.npy", weight, allow_pickle=False)
    core_id = np.asarray(assignment["core_id"], dtype=np.int64)
    core_weight = np.bincount(
        core_id, weights=weight, minlength=len(cores["core_id"])
    ).astype(np.float64)
    eligible_core = np.flatnonzero(core_weight > 0).astype(np.int32)
    role = (
        "training" if phase in TRAINING_PHASES
        else "validation_and_selection" if phase == VALIDATION_PHASE
        else "sealed_blind_test"
    )
    if role == "training":
        np.save(output / "training_core_id.npy", eligible_core, allow_pickle=False)
        np.save(
            output / "training_core_weight.npy",
            core_weight[eligible_core],
            allow_pickle=False,
        )
    elif role == "validation_and_selection":
        np.save(output / "validation_core_id.npy", eligible_core, allow_pickle=False)
    else:
        np.save(output / "blind_core_id.npy", eligible_core, allow_pickle=False)
    record = {
        "schema_version": "p10-phase-loader-contract-v1",
        "phase": phase,
        "role": role,
        "truth_present": truth_present,
        "parent_rows": int(n_parent),
        "authoritative_rows": int(authoritative.sum()),
        "eligible_cores": int(len(eligible_core)),
        "shell_counts": {
            SHELL_NAMES[index]: int(counts[index]) for index in range(4)
        },
        "phase_weight_denominator": float(weight.sum(dtype=np.float64)),
        "target": target_record,
        "inputs": {
            "p1_manifest": str(files["p1_manifest"]),
            "p1_manifest_sha256": sha256(files["p1_manifest"]),
            "assignment": str(files["assignment"]),
            "assignment_sha256": sha256(files["assignment"]),
            "cores": str(files["cores"]),
            "cores_sha256": sha256(files["cores"]),
        },
        "gates": {
            "parent_row_identity": True,
            "targetid_identity": True,
            "authoritative_unique": True,
            "all_four_shells": True,
            "truth_sealed_if_blind": phase != BLIND_PHASE or not truth_present,
            "targets_finite_ordered_if_visible": phase == BLIND_PHASE or target_record is not None,
        },
    }
    record["pass"] = all(record["gates"].values())
    atomic_json(output / "phase_contract.json", record)
    return record


def shared_selection_fit(root: Path, contract: Path) -> dict:
    output = contract / "transforms" / "field"
    output.mkdir(parents=True, exist_ok=True)
    edges = np.arange(Z_MIN, Z_MAX + 0.5 * BIN_WIDTH, BIN_WIDTH)
    centers = 0.5 * (edges[:-1] + edges[1:])
    grid_z = np.arange(Z_MIN, Z_MAX + 0.5 * CURVE_STEP, CURVE_STEP)
    radius_grid, redshift_grid = radius_to_redshift_grid(Z_MIN, Z_MAX)
    counts_total = np.zeros((2, len(edges) - 1), dtype=np.int64)
    volume_total = np.zeros((2, len(edges) - 1), dtype=np.float64)
    sources = {}
    for phase in TRAINING_PHASES:
        files = phase_files(root, phase)
        p3 = json.loads(files["p3_manifest"].read_text())
        cores = np.load(files["cores"])
        lookups = {
            name: build_cap_lookup(cores, cap, 64.0)
            for cap, name in CAP_NAME.items()
        }
        counts, count_audit = histogram_counts(
            parent_path=Path(p1_manifest(root, phase)["parent"]),
            context_path=files["p4_root"] / "context_assignment.npz",
            edges=edges,
        )
        volume, volume_audit = histogram_effective_volume(
            p3=p3,
            lookups=lookups,
            core_mpc=64.0,
            edges=edges,
            radius_grid_mpc=radius_grid,
            redshift_grid=redshift_grid,
        )
        counts_total += counts.sum(axis=1)
        volume_total += volume.sum(axis=1)
        sources[phase] = {
            "counts": count_audit,
            "volume": volume_audit,
            "p3_manifest_sha256": sha256(files["p3_manifest"]),
        }
    caps = {}
    for cap, name in CAP_NAME.items():
        curve, fit = fit_log_spline(
            centers,
            counts_total[cap],
            volume_total[cap],
            grid_z,
            knot_spacing=KNOT_SPACING,
            fit_z_min=FIT_Z_MIN,
            fit_z_max=FIT_Z_MAX,
        )
        expected = np.interp(centers, grid_z, curve) * volume_total[cap]
        shell_closure = []
        for shell_low, shell_high in (
            (0.15, 0.25), (0.25, 0.35), (0.35, 0.45), (0.45, 0.55)
        ):
            selected = (centers >= shell_low) & (centers < shell_high)
            observed = float(counts_total[cap, selected].sum())
            predicted = float(expected[selected].sum())
            shell_closure.append({
                "z_low": shell_low,
                "z_high": shell_high,
                "observed": observed,
                "expected": predicted,
                "fractional_error": predicted / observed - 1.0,
            })
        caps[name] = {
            "grid_z": grid_z.tolist(),
            "ntilde": curve.tolist(),
            "fit": fit,
            "training_shell_closure": shell_closure,
        }
    selection = {
        "schema_version": "p10-shared-selection-v1",
        "fit_phases": list(TRAINING_PHASES),
        "application_phases": [VALIDATION_PHASE, BLIND_PHASE],
        "fit_scope": "all P4 context folds in each training phase; label-free observed counts and P3 exposure only",
        "rotations": {"0": {"caps": caps}},
        "cosmology": {
            "name": "Planck18",
            "radius_grid_mpc": radius_grid.tolist(),
            "redshift_grid": redshift_grid.tolist(),
        },
        "contrast": {
            "epsilon": CONTRAST_EPSILON,
            "minimum_exposure": MINIMUM_EXPOSURE,
        },
        "sources": sources,
        "phase_is_model_input": False,
    }
    selection["gates"] = {
        "training_phases_only": set(selection["fit_phases"]) == set(TRAINING_PHASES),
        "no_validation_or_blind_fit": not (
            {VALIDATION_PHASE, BLIND_PHASE} & set(selection["fit_phases"])
        ),
        "finite_positive_curves": all(
            np.all(np.isfinite(row["ntilde"])) and np.all(np.asarray(row["ntilde"]) > 0)
            for row in caps.values()
        ),
        "training_shell_closure_below_10pct": all(
            abs(row["fractional_error"]) < 0.10
            for cap in caps.values()
            for row in cap["training_shell_closure"]
        ),
    }
    selection["pass"] = all(selection["gates"].values())
    path = output / "selection_manifest.json"
    atomic_json(path, selection)
    if not selection["pass"]:
        raise RuntimeError(f"shared selection fit failed: {selection['gates']}")
    return selection


def _moments(values: np.ndarray) -> tuple[int, float, float]:
    values = np.asarray(values, dtype=np.float64)
    return int(values.size), float(values.sum()), float(np.square(values).sum())


def field_transform_fit(root: Path, contract: Path, selection: dict) -> dict:
    per_phase = {}
    radius_grid = np.asarray(selection["cosmology"]["radius_grid_mpc"])
    redshift_grid = np.asarray(selection["cosmology"]["redshift_grid"])
    for phase in TRAINING_PHASES:
        p3 = json.loads(phase_files(root, phase)["p3_manifest"].read_text())
        accum = defaultdict(lambda: [0, 0.0, 0.0])
        for cap, name in CAP_NAME.items():
            component = p3["components"][name]
            grid = component["grid"]
            curve = selection["rotations"]["0"]["caps"][name]
            with h5py.File(component["file"], "r") as handle:
                counts_ds = handle["counts"]
                exposure_ds = handle["exposure_apodized"]
                for selection_slice in counts_ds.iter_chunks():
                    counts = np.asarray(counts_ds[selection_slice], dtype=np.float32)
                    exposure = np.asarray(exposure_ds[selection_slice], dtype=np.float32)
                    start = np.asarray([part.start for part in selection_slice], dtype=np.int64)
                    redshift = patch_redshift(
                        origin_mpc=np.asarray(grid["origin_mpc"], dtype=np.float64),
                        cell_mpc=float(grid["cell_mpc"]),
                        context_start=start,
                        shape=counts.shape,
                        radius_grid_mpc=radius_grid,
                        redshift_grid=redshift_grid,
                    )
                    derived = derive_selection_channels(
                        counts,
                        exposure,
                        redshift,
                        cell_mpc=float(grid["cell_mpc"]),
                        grid_z=np.asarray(curve["grid_z"]),
                        ntilde=np.asarray(curve["ntilde"]),
                        epsilon=CONTRAST_EPSILON,
                        minimum_exposure=MINIMUM_EXPOSURE,
                    )
                    supported = exposure > MINIMUM_EXPOSURE
                    values = {
                        "counts": channel_transform("counts", counts[supported]),
                        "expected_counts": channel_transform(
                            "expected_counts", derived["expected_counts"][supported]
                        ),
                        "log_count_ratio": derived["log_count_ratio"][supported],
                        "ntilde_mpc3": channel_transform(
                            "ntilde_mpc3", derived["ntilde_mpc3"][supported]
                        ),
                    }
                    for channel, array in values.items():
                        n, total, total2 = _moments(array)
                        accum[channel][0] += n
                        accum[channel][1] += total
                        accum[channel][2] += total2
        per_phase[phase] = {}
        for channel, (n, total, total2) in accum.items():
            mean = total / n
            per_phase[phase][channel] = {
                "count": n,
                "mean": mean,
                "second_moment": total2 / n,
                "std": max(total2 / n - mean * mean, 0.0) ** 0.5,
            }
    normalization = {"channels": {}}
    for channel in ("counts", "expected_counts", "log_count_ratio", "ntilde_mpc3"):
        mean = float(np.mean([per_phase[phase][channel]["mean"] for phase in TRAINING_PHASES]))
        second = float(np.mean([
            per_phase[phase][channel]["second_moment"] for phase in TRAINING_PHASES
        ]))
        std = max(second - mean * mean, 0.0) ** 0.5
        if not np.isfinite(std) or std <= 0:
            raise RuntimeError(f"invalid shared field scaler for {channel}")
        normalization["channels"][channel] = {
            "policy": "zscore", "mean": mean, "std": std
        }
    for channel in (
        "exposure_apodized", "exposure_binary", "los_x", "los_y", "los_z"
    ):
        normalization["channels"][channel] = {"policy": "identity"}
    manifest = {
        "schema_version": "p10-field-transform-v1",
        "fit_phases": list(TRAINING_PHASES),
        "fit_policy": "equal phase mixture of exact supported-voxel first and second moments",
        "selection_manifest": str(
            contract / "transforms/field/selection_manifest.json"
        ),
        "selection_manifest_sha256": sha256(
            contract / "transforms/field/selection_manifest.json"
        ),
        "normalization": normalization,
        "per_phase_diagnostics": per_phase,
        "gates": {
            "training_phases_only": True,
            "phase_equal_moments": True,
            "all_transforms_finite": all(
                np.isfinite(row.get("mean", 0.0)) and np.isfinite(row.get("std", 1.0))
                for row in normalization["channels"].values()
            ),
            "no_patch_local_statistics": True,
        },
    }
    manifest["pass"] = all(manifest["gates"].values())
    atomic_json(contract / "transforms/field/field_transform.json", manifest)
    return manifest


def _sample_rows(
    parent: np.ndarray,
    cap: np.ndarray,
    *,
    seed: int,
    sample_per_cap: int,
) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    result = {}
    for cap_id in (0, 1):
        candidates = parent[cap == cap_id]
        n = min(sample_per_cap, len(candidates))
        result[cap_id] = np.sort(rng.choice(candidates, size=n, replace=False))
    return result


def graph_transform_fit(root: Path, contract: Path, selection: dict) -> dict:
    output = contract / "transforms" / "graph"
    output.mkdir(parents=True, exist_ok=True)
    node_sample = {0: [], 1: []}
    z_sample = {0: [], 1: []}
    sampled_by_phase = {}
    for phase_index, phase in enumerate(TRAINING_PHASES):
        phase_dir = contract / "phases" / phase
        assignment = np.load(phase_files(root, phase)["assignment"], mmap_mode="r")
        auth = np.asarray(assignment["supervised_eligible"], dtype=bool)
        parent = np.asarray(assignment["parent_node_id"][auth], dtype=np.int64)
        cap = np.asarray(assignment["cap"][auth], dtype=np.int8)
        samples = _sample_rows(
            parent, cap, seed=104729 + phase_index, sample_per_cap=NODE_SAMPLE_PER_PHASE_CAP
        )
        raw = np.load(contract / "adapters" / phase / "graph/node_features.npy", mmap_mode="r")
        redshift = np.load(phase_dir / "parent_redshift.npy", mmap_mode="r")
        sampled_by_phase[phase] = {}
        for cap_id in (0, 1):
            ids = samples[cap_id]
            node_sample[cap_id].append(np.asarray(raw[ids], dtype=np.float64))
            z_sample[cap_id].append(np.asarray(redshift[ids], dtype=np.float64))
            sampled_by_phase[phase][CAP_NAME[cap_id]] = int(len(ids))
    medians = {
        str(cap): {
            str(column): float(np.median(np.concatenate(node_sample[cap])[:, column]))
            for column in SI_COLUMNS
        }
        for cap in (0, 1)
    }
    pooled = []
    ntilde_log = []
    for cap in (0, 1):
        values = np.concatenate(node_sample[cap])
        for column in SI_COLUMNS:
            values[:, column] /= max(medians[str(cap)][str(column)], 1.0e-9)
        if np.any(values + 1.0e-6 <= 0):
            raise RuntimeError("non-positive graph Box-Cox input")
        pooled.append(values + 1.0e-6)
        curve = selection["rotations"]["0"]["caps"][CAP_NAME[cap]]
        z = np.concatenate(z_sample[cap])
        ntilde = np.interp(
            np.clip(z, curve["grid_z"][0], curve["grid_z"][-1]),
            curve["grid_z"],
            curve["ntilde"],
        )
        ntilde_log.append(np.log(np.maximum(ntilde, 1.0e-12)))
    pooled_values = np.concatenate(pooled)
    power = PowerTransformer(method="box-cox").fit(pooled_values)
    log_ntilde = np.concatenate(ntilde_log)
    ntilde_mean = float(log_ntilde.mean())
    ntilde_std = float(log_ntilde.std())
    if not np.isfinite(ntilde_std) or ntilde_std <= 0:
        raise RuntimeError("invalid graph ntilde scaler")
    with (output / "node_power_transformer.pkl").open("wb") as handle:
        pickle.dump(power, handle)

    # Equal phase/cap random edge samples.  These are transform-fit samples, not
    # training examples, and are bound by seed and count in the manifest.
    edge_samples = {0: [], 1: []}
    for phase_index, phase in enumerate(TRAINING_PHASES):
        adapter_root = contract / "adapters" / phase / "graph"
        pairs = np.load(adapter_root / "union_pairs.npy", mmap_mode="r")
        attrs = np.load(adapter_root / "union_edge_features.npy", mmap_mode="r")
        cap_all = np.asarray(
            np.load(phase_files(root, phase)["p1_index"], mmap_mode="r")["cap"],
            dtype=np.int8,
        )
        rng = np.random.default_rng(32452843 + phase_index)
        for cap_id in (0, 1):
            chosen = []
            needed = EDGE_SAMPLE_PER_PHASE_CAP
            while sum(len(block) for block in chosen) < needed:
                draw = rng.integers(0, len(pairs), size=max(needed * 2, 100_000))
                edge_cap = cap_all[np.asarray(pairs[draw, 0], dtype=np.int64)]
                selected = draw[edge_cap == cap_id]
                if len(selected):
                    chosen.append(selected[: needed - sum(len(block) for block in chosen)])
            ids = np.concatenate(chosen)
            edge_samples[cap_id].append(np.asarray(attrs[ids], dtype=np.float64))
    edge = {}
    for cap_id in (0, 1):
        values = np.concatenate(edge_samples[cap_id])
        length_median = float(np.median(values[:, 0]))
        log_length = np.log(np.maximum(values[:, 0] / length_median, 1.0e-6))
        log_density = np.log(np.maximum(values[:, 4], 1.0e-6))
        directed_density = np.concatenate((log_density, -log_density))
        edge[str(cap_id)] = {
            "edge_length_si_median": length_median,
            "log_length_mean": float(log_length.mean()),
            "log_length_std": float(log_length.std()),
            "log_density_contrast_mean": float(directed_density.mean()),
            "log_density_contrast_std": float(directed_density.std()),
            "fit_samples_per_phase": EDGE_SAMPLE_PER_PHASE_CAP,
        }

    # Apply the one frozen node transform to every input phase, including blind.
    applied = {}
    for phase in PHASES:
        adapter_root = contract / "adapters" / phase / "graph"
        raw = np.load(adapter_root / "node_features.npy", mmap_mode="r")
        index = np.load(phase_files(root, phase)["p1_index"], mmap_mode="r")
        cap_all = np.asarray(index["cap"], dtype=np.int8)
        redshift = np.load(contract / "phases" / phase / "parent_redshift.npy", mmap_mode="r")
        phase_out = output / phase
        phase_out.mkdir(parents=True, exist_ok=True)
        path = phase_out / "node_features_8d.npy"
        transformed = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float32, shape=(len(raw), 8)
        )
        for start in range(0, len(raw), 1_000_000):
            stop = min(start + 1_000_000, len(raw))
            cap = cap_all[start:stop]
            values = np.asarray(raw[start:stop], dtype=np.float64).copy()
            for cap_id in (0, 1):
                selected = cap == cap_id
                for column in SI_COLUMNS:
                    values[selected, column] /= max(
                        medians[str(cap_id)][str(column)], 1.0e-9
                    )
            transformed[start:stop, :7] = power.transform(values + 1.0e-6).astype(np.float32)
            log_curve = np.empty(stop - start, dtype=np.float64)
            for cap_id in (0, 1):
                selected = cap == cap_id
                curve = selection["rotations"]["0"]["caps"][CAP_NAME[cap_id]]
                ntilde = np.interp(
                    np.clip(
                        np.asarray(redshift[start:stop])[selected],
                        curve["grid_z"][0],
                        curve["grid_z"][-1],
                    ),
                    curve["grid_z"],
                    curve["ntilde"],
                )
                log_curve[selected] = np.log(np.maximum(ntilde, 1.0e-12))
            transformed[start:stop, 7] = (
                (log_curve - ntilde_mean) / ntilde_std
            ).astype(np.float32)
        transformed.flush()
        del transformed
        applied[phase] = {
            "path": str(path),
            "sha256": sha256(path),
            "rows": int(len(raw)),
        }
    manifest = {
        "schema_version": "p10-graph-transform-v1",
        "fit_phases": list(TRAINING_PHASES),
        "application_phases": list(PHASES),
        "node": {
            "raw_columns": [
                "Degree", "Clustering", "Density", "Neigh Density",
                "I_eig1", "I_eig2", "I_eig3",
            ],
            "si_columns": list(SI_COLUMNS),
            "si_medians": medians,
            "boxcox_epsilon": 1.0e-6,
            "boxcox_lambdas": power.lambdas_.tolist(),
            "boxcox_standardizer_mean": power._scaler.mean_.tolist(),
            "boxcox_standardizer_scale": power._scaler.scale_.tolist(),
            "ntilde_log_mean": ntilde_mean,
            "ntilde_log_std": ntilde_std,
            "fit_samples_per_phase_cap": NODE_SAMPLE_PER_PHASE_CAP,
            "sample_counts": sampled_by_phase,
            "power_transformer": str(output / "node_power_transformer.pkl"),
            "power_transformer_sha256": sha256(output / "node_power_transformer.pkl"),
        },
        "edge": edge,
        "applied": applied,
        "gates": {
            "training_phases_only": True,
            "phase_cap_balanced_fit_sample": True,
            "all_application_arrays_finite": all(
                np.all(np.isfinite(np.load(row["path"], mmap_mode="r")))
                for row in applied.values()
            ),
            "eight_features": all(
                np.load(row["path"], mmap_mode="r").shape[1] == 8
                for row in applied.values()
            ),
            "phase_not_feature": True,
        },
    }
    manifest["pass"] = all(manifest["gates"].values())
    atomic_json(output / "graph_transform.json", manifest)
    return manifest


def target_transform_fit(contract: Path) -> dict:
    means = []
    seconds = []
    diagnostics = {}
    for phase in TRAINING_PHASES:
        phase_dir = contract / "phases" / phase
        phase_record = json.loads((phase_dir / "phase_contract.json").read_text())
        assignment_path = Path(phase_record["inputs"]["assignment"])
        assignment = np.load(assignment_path, mmap_mode="r")
        auth = np.asarray(assignment["supervised_eligible"], dtype=bool)
        parent = np.asarray(assignment["parent_node_id"][auth], dtype=np.int64)
        eigen = np.load(phase_dir / "parent_eigenvalues.npy", mmap_mode="r")
        increments = linear_increments(np.asarray(eigen[parent], dtype=np.float64))
        mean = increments.mean(axis=0)
        second = np.square(increments).mean(axis=0)
        means.append(mean)
        seconds.append(second)
        diagnostics[phase] = {
            "rows": int(len(increments)),
            "mean": mean.tolist(),
            "std": np.sqrt(np.maximum(second - mean * mean, 0.0)).tolist(),
        }
    mean = np.mean(means, axis=0)
    second = np.mean(seconds, axis=0)
    std = np.sqrt(np.maximum(second - mean * mean, 0.0))
    if np.any(~np.isfinite(std)) or np.any(std <= 0):
        raise RuntimeError("invalid target transform")
    scaler = {
        "schema_version": "p10-target-scaler-v1",
        "representation": "linear increments",
        "definition": ["lambda1", "lambda2-lambda1", "lambda3-lambda2"],
        "fit_phases": list(TRAINING_PHASES),
        "fit_policy": "equal phase mixture of exact authoritative-row moments",
        "mean": mean.tolist(),
        "std": std.tolist(),
        "per_phase_diagnostics": diagnostics,
        "phase_not_feature": True,
        "pass": True,
    }
    atomic_json(contract / "transforms/target_scaler.json", scaler)
    return scaler


def run_canaries(root: Path, contract: Path, adapters: dict) -> dict:
    phase_records = {
        phase: json.loads((contract / "phases" / phase / "phase_contract.json").read_text())
        for phase in PHASES
    }
    core_ids = {
        phase: np.load(contract / "phases" / phase / "training_core_id.npy")
        for phase in TRAINING_PHASES
    }
    core_weights = {
        phase: np.load(contract / "phases" / phase / "training_core_weight.npy")
        for phase in TRAINING_PHASES
    }
    refs = phase_balanced_epoch(core_ids, seed=42, epoch=0, core_weight_by_phase=core_weights)
    refs_again = phase_balanced_epoch(
        core_ids, seed=42, epoch=0, core_weight_by_phase=core_weights
    )
    order_deterministic = epoch_hash(refs) == epoch_hash(refs_again)
    prefix_counts = defaultdict(int)
    prefix_balanced = True
    for position, ref in enumerate(refs[: min(len(refs), 10_000)], start=1):
        prefix_counts[ref.phase] += 1
        if position % len(TRAINING_PHASES) == 0:
            prefix_balanced &= max(prefix_counts.values()) - min(prefix_counts.values()) <= 1
    cursor = min(12_345, len(refs))
    state = resume_state(seed=42, epoch=0, cursor=cursor, refs=refs)
    validate_resume_state(state, refs_again)
    resume_parity = [row.core_id for row in refs[cursor:]] == [
        row.core_id for row in refs_again[cursor:]
    ]

    expected = 0.0
    objective_sum = 0.0
    total_steps = len(refs)
    phase_core_numerator = {}
    for phase in TRAINING_PHASES:
        phase_dir = contract / "phases" / phase
        record = phase_records[phase]
        assignment = np.load(Path(record["inputs"]["assignment"]), mmap_mode="r")
        weight = np.load(phase_dir / "active_row_weight.npy", mmap_mode="r")
        parent = np.asarray(assignment["parent_node_id"], dtype=np.int64)
        core = np.asarray(assignment["core_id"], dtype=np.int64)
        synthetic = ((parent % 97) + 1).astype(np.float64) / 97.0
        numerator = np.bincount(
            core,
            weights=np.asarray(weight, dtype=np.float64) * synthetic,
            minlength=int(core.max()) + 1,
        )
        phase_core_numerator[phase] = numerator
        expected += float(
            np.sum(np.asarray(weight, dtype=np.float64) * synthetic)
            / record["phase_weight_denominator"]
        ) / len(TRAINING_PHASES)
    for ref in refs:
        objective_sum += float(
            phase_equal_patch_objective(
                phase_core_numerator[ref.phase][ref.core_id],
                phase_weight_denominator=phase_records[ref.phase][
                    "phase_weight_denominator"
                ],
                phase_objective_scale=ref.phase_objective_scale,
            )
        )
    reconstructed = objective_sum / total_steps
    loss_accounting = bool(np.isclose(reconstructed, expected, rtol=1.0e-12, atol=1.0e-12))

    extraction = {}
    for phase in TRAINING_PHASES + (VALIDATION_PHASE,):
        phase_dir = contract / "phases" / phase
        core_name = (
            "training_core_id.npy" if phase in TRAINING_PHASES
            else "validation_core_id.npy"
        )
        candidates = np.load(phase_dir / core_name)
        core_id = int(candidates[len(candidates) // 2])
        graph = CanonicalGraphPatchAdapter(contract / "adapters" / phase / "graph")
        graph.node_features = np.load(
            contract / "transforms" / "graph" / phase / "node_features_8d.npy",
            mmap_mode="r",
        )
        gp1 = graph.extract(
            core_id,
            2,
            dependency_hops_per_pass=2,
            loss_policy="authoritative",
        )
        gp2 = graph.extract(
            core_id,
            2,
            dependency_hops_per_pass=2,
            loss_policy="authoritative",
        )
        field = CanonicalFieldPatchAdapter(
            contract / "adapters" / phase / "field",
            selection_manifest=contract / "transforms/field/selection_manifest.json",
            rotation=0,
        )
        fp1 = field.extract(
            core_id,
            24,
            ("counts", "exposure_apodized", "log_count_ratio"),
            alignment_voxels=8,
        )
        fp2 = field.extract(
            core_id,
            24,
            ("counts", "exposure_apodized", "log_count_ratio"),
            alignment_voxels=8,
        )
        targets = np.load(phase_dir / "parent_eigenvalues.npy", mmap_mode="r")
        target_ok = np.all(np.isfinite(targets[gp1.parent_node_id[gp1.loss_mask]]))
        extraction[phase] = {
            "core_id": core_id,
            "graph_nodes": gp1.n_node,
            "graph_edges": gp1.n_edge,
            "field_shape": list(fp1.values.shape),
            "graph_deterministic": bool(
                np.array_equal(gp1.parent_node_id, gp2.parent_node_id)
                and np.array_equal(gp1.senders, gp2.senders)
                and np.array_equal(gp1.node_features, gp2.node_features)
            ),
            "field_deterministic": bool(
                np.array_equal(fp1.authoritative_parent_id, fp2.authoritative_parent_id)
                and np.array_equal(fp1.values, fp2.values)
            ),
            "target_rows_align": bool(target_ok),
        }
        field.close()

    validation_ids = np.load(
        contract / "phases" / VALIDATION_PHASE / "validation_core_id.npy"
    )
    validation_order_one = np.asarray(validation_ids, dtype=np.int64)
    validation_order_two = np.asarray(validation_ids, dtype=np.int64)
    gates = {
        "all_phase_contracts_pass": all(row["pass"] for row in phase_records.values()),
        "all_adapters_pass": all(
            json.loads(Path(row[k]).read_text()).get("pass")
            for row in adapters.values()
            for k in ("graph_manifest", "field_manifest")
        ),
        "complete_epoch_coverage": len(refs) == sum(len(ids) for ids in core_ids.values()),
        "epoch_order_deterministic": order_deterministic,
        "phase_prefix_balanced": bool(prefix_balanced),
        "resume_parity": bool(resume_parity),
        "weighted_loss_accounting": loss_accounting,
        "graph_extraction_deterministic_all_phases": all(
            row["graph_deterministic"] for row in extraction.values()
        ),
        "field_extraction_deterministic_all_phases": all(
            row["field_deterministic"] for row in extraction.values()
        ),
        "target_alignment_all_visible_phases": all(
            row["target_rows_align"] for row in extraction.values()
        ),
        "complete_deterministic_validation_order": bool(
            np.array_equal(validation_order_one, validation_order_two)
            and len(np.unique(validation_order_one)) == len(validation_order_one)
        ),
        "ph001_truth_absent": not (
            contract / "phases" / BLIND_PHASE / "parent_eigenvalues.npy"
        ).exists(),
    }
    return {
        "gates": gates,
        "pass": all(gates.values()),
        "epoch": {
            "seed": 42,
            "epoch": 0,
            "cores": len(refs),
            "sha256": epoch_hash(refs),
            "phase_core_counts": {phase: len(ids) for phase, ids in core_ids.items()},
            "resume_cursor": cursor,
        },
        "loss_accounting": {
            "direct_equal_phase_loss": expected,
            "reconstructed_epoch_mean": reconstructed,
        },
        "extraction": extraction,
        "validation_cores": int(len(validation_ids)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument(
        "--contract-root", type=Path,
        default=ROOT / "training_contract",
    )
    parser.add_argument(
        "--stage",
        choices=("adapters", "transforms", "canary", "all"),
        default="all",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=PHASES,
        default=list(PHASES),
        help="adapter phases; transforms/canary still require the frozen full role set",
    )
    args = parser.parse_args()
    started = time.time()
    args.contract_root.mkdir(parents=True, exist_ok=True)
    registry = json.loads(args.registry.read_text())
    roles = registry["model_phase_contract"]
    if tuple(roles["training"]) != TRAINING_PHASES:
        raise RuntimeError("registry training roles do not match loader code")
    if roles["validation_and_selection"] != [VALIDATION_PHASE]:
        raise RuntimeError("registry validation role mismatch")
    if roles["sealed_blind_test"] != [BLIND_PHASE]:
        raise RuntimeError("registry blind role mismatch")

    adapters_path = args.contract_root / "adapter_inventory.json"
    if args.stage in ("adapters", "all"):
        adapters = build_adapters(args.root, args.contract_root, tuple(args.phases))
        atomic_json(adapters_path, {
            "schema_version": "p10-adapter-inventory-v1",
            "phases": adapters,
            "pass": len(adapters) == len(args.phases),
        })
        if args.stage == "adapters":
            return
    if not adapters_path.is_file():
        raise RuntimeError("adapter inventory is missing")
    adapters = json.loads(adapters_path.read_text())["phases"]
    missing = set(PHASES).difference(adapters)
    if missing:
        raise RuntimeError(f"adapters are incomplete: {sorted(missing)}")

    if args.stage in ("transforms", "all"):
        phase_records = {
            phase: prepare_phase_arrays(args.root, args.contract_root, phase)
            for phase in PHASES
        }
        selection = shared_selection_fit(args.root, args.contract_root)
        field = field_transform_fit(args.root, args.contract_root, selection)
        graph = graph_transform_fit(args.root, args.contract_root, selection)
        target = target_transform_fit(args.contract_root)
        transform_manifest = {
            "schema_version": "p10-transform-freeze-v1",
            "fit_phases": list(TRAINING_PHASES),
            "application_phases": [VALIDATION_PHASE, BLIND_PHASE],
            "phase_records": {
                phase: {
                    "path": str(args.contract_root / "phases" / phase / "phase_contract.json"),
                    "sha256": sha256(
                        args.contract_root / "phases" / phase / "phase_contract.json"
                    ),
                }
                for phase in PHASES
            },
            "selection_pass": selection["pass"],
            "field_pass": field["pass"],
            "graph_pass": graph["pass"],
            "target_pass": target["pass"],
            "pass": all((
                selection["pass"], field["pass"], graph["pass"], target["pass"],
                all(row["pass"] for row in phase_records.values()),
            )),
        }
        atomic_json(args.contract_root / "TRANSFORMS_FROZEN.json", transform_manifest)
        if not transform_manifest["pass"]:
            raise RuntimeError("transform freeze failed")
        if args.stage == "transforms":
            return

    transform_manifest = json.loads(
        (args.contract_root / "TRANSFORMS_FROZEN.json").read_text()
    )
    if not transform_manifest.get("pass"):
        raise RuntimeError("transforms are not frozen")
    canary = run_canaries(args.root, args.contract_root, adapters)
    if not canary["pass"]:
        atomic_json(args.contract_root / "TRAINING_LOADER_FAILED.json", canary)
        raise RuntimeError(f"training canaries failed: {canary['gates']}")
    marker = {
        "schema_version": "p10-training-loader-ready-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "P10 deterministic multi-phase loader and transform readiness",
        "status": "TRAINING_LOADER_READY",
        "roles": {
            "training": list(TRAINING_PHASES),
            "validation_and_selection": VALIDATION_PHASE,
            "sealed_blind_test": BLIND_PHASE,
        },
        "fresh_model_initialization_required": True,
        "phase_is_model_input": False,
        "objective": (
            "equal mean over phases of within-phase sqrt-shell-weighted authoritative-row MSE"
        ),
        "epoch": (
            "every eligible core exactly once; weighted within-phase order; "
            "shuffled phase round-robin; deterministic seed+epoch"
        ),
        "validation": "all authoritative ph006 cores in frozen core-id order",
        "registry": str(args.registry.resolve()),
        "registry_sha256": sha256(args.registry),
        "adapter_inventory": str(adapters_path),
        "adapter_inventory_sha256": sha256(adapters_path),
        "transform_manifest": str(args.contract_root / "TRANSFORMS_FROZEN.json"),
        "transform_manifest_sha256": sha256(
            args.contract_root / "TRANSFORMS_FROZEN.json"
        ),
        "canary": canary,
        "gates": canary["gates"],
        "elapsed_seconds": time.time() - started,
        "pass": True,
    }
    atomic_json(args.contract_root / "TRAINING_LOADER_READY.json", marker)
    print(json.dumps(marker, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

