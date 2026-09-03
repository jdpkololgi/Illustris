#!/usr/bin/env python3
"""Post-selection causal diagnostics for the frozen P12-F G1 field posterior.

The commands in this module are explanatory ph006 diagnostics.  They never fit a
production correction, never open ph001, and cannot change the frozen P12-F
no-field-finalist decision.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Iterable

import numpy as np
import torch

from workflows.abacus_tweb.p3br_training_contract import P10RandomResponseLoader
from workflows.abacus_tweb.p8_deterministic_common import atomic_json, sha256
from workflows.abacus_tweb.p8_train_unet_patch import CHANNELS, model_inputs
from workflows.sbi.p12f_challenger_common import core_joint_scores
from workflows.sbi.p12f_common_evaluator import (
    load_core_record,
    validate_archive_manifest,
)
from workflows.sbi.p12f_dependency_rescue_evaluator import (
    _sbc,
    _subpanel_labels,
    tarp_curve,
    tidal_eigenvalues_at_galaxies,
)
from workflows.sbi.p12f_field_posterior_diagnostics import scalar_posterior_report
from workflows.sbi.p12f_freeze_selection_panel import summarize_observed_core
from workflows.sbi.p12f_gaussian_controls import ConditionalGaussianUNet
from workflows.sbi.p12f_train_conditional_field_flow import (
    FieldTargetStore,
    target_tensor,
    unscale,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/p12f_causal_autopsy_v1.json"
METHODS = (
    "g1",
    "lowk_mean_oracle",
    "lowk_power_oracle",
    "lowk_mean_power_oracle",
    "empirical_residual_patch",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    if config.get("schema_version") != "p12f-causal-autopsy-v1":
        raise RuntimeError("unsupported P12-F causal-autopsy contract")
    roles = config.get("roles", {})
    guards = config.get("guards", {})
    if (
        roles.get("diagnostic_validation") != "ph006"
        or roles.get("sealed_blind_test") != "ph001"
        or guards.get("ph001_opened")
        or guards.get("ph006_recalibration_allowed")
        or guards.get("may_change_no_field_finalist")
    ):
        raise PermissionError("P12-F causal-autopsy phase/production guard changed")
    source_strings = json.dumps(config.get("sources", {})).lower()
    if "ph001" in source_strings:
        raise PermissionError("a ph001 path entered the causal-autopsy sources")
    return config


def source_paths(config: dict) -> dict[str, Path]:
    output = {}
    for name, value in config["sources"].items():
        path = Path(value)
        output[name] = path if path.is_absolute() else REPO_ROOT / path
    return output


def validate_frozen_sources(config_path: Path, config: dict) -> tuple[dict, list[dict]]:
    source = source_paths(config)
    for name, path in source.items():
        if name in {"conditioning_contract", "phase_root"}:
            if not path.is_dir():
                raise FileNotFoundError(path)
        elif not path.is_file():
            raise FileNotFoundError(path)
    panel = json.loads(source["panel_marker"].read_text())
    archive = json.loads(source["g1_archive_manifest"].read_text())
    parent = json.loads(source["parent_config"].read_text())
    if (
        panel.get("schema_version") != "p12f-truth-free-selection-panel-v1"
        or not panel.get("pass")
        or panel.get("selection_uses_truth")
        or panel.get("truth_files_read")
        or panel.get("ph001_opened")
        or len(panel.get("selected_core_id", [])) != 1024
    ):
        raise RuntimeError("causal autopsy requires the frozen truth-free ph006 panel")
    entries = validate_archive_manifest(
        archive,
        archive_path=source["g1_archive_manifest"],
        panel=panel,
        panel_path=source["panel_marker"],
        config=parent,
    )
    if archive.get("method") != "gaussian_correlated_g1" or int(archive["draws"]) != 256:
        raise RuntimeError("causal autopsy requires the frozen 256-draw G1 archive")
    compact = json.loads(source["compact_archive_ready"].read_text())
    compact_entries = compact.get("entries", [])
    if (
        compact.get("schema_version") != "p12f-dependency-rescue-compact-ready-v2"
        or not compact.get("pass")
        or compact.get("ph001_opened")
        or len(compact_entries) != 1024
    ):
        raise RuntimeError("causal autopsy requires the complete compact archive")
    if [int(row["core_id"]) for row in compact_entries] != [
        int(row["core_id"]) for row in entries
    ]:
        raise RuntimeError("G1 sample and compact archives are not core-aligned")
    return panel, entries


def frozen_manifest(config_path: Path, config: dict, panel: dict, entries: list[dict]) -> dict:
    source = source_paths(config)
    files = {
        name: {"path": str(path.resolve()), "sha256": sha256(path)}
        for name, path in source.items()
        if path.is_file()
    }
    payload = {
        "config": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "sources": files,
        "conditioning_ready_sha256": sha256(
            source["conditioning_contract"] / "TRAINING_LOADER_READY.json"
        ),
        "archive_entries": [
            {"core_id": int(row["core_id"]), "sha256": row["sha256"]}
            for row in entries
        ],
        "panel_core_ids": [int(value) for value in panel["selected_core_id"]],
        "ph001_opened": False,
    }
    return {
        "schema_version": "p12f-causal-autopsy-run-v1",
        "created_utc": utc_now(),
        "git_revision_at_launch": git_revision(),
        "frozen_digest": canonical_digest(payload),
        "frozen": payload,
        "truth_files_read": ["ph006 frozen G1 diagnostic archive"],
        "ph001_opened": False,
    }


def physical_invariants(eigenvalues: np.ndarray) -> dict[str, np.ndarray]:
    value = np.asarray(eigenvalues, dtype=np.float64)
    if value.shape[-1] != 3 or not np.all(np.isfinite(value)):
        raise ValueError("physical invariants require finite [...,3] eigenvalues")
    trace = np.sum(value, axis=-1)
    shear = value - trace[..., None] / 3.0
    shear2 = np.sum(np.square(shear), axis=-1)
    q = np.sqrt(1.5 * np.maximum(shear2, 0.0))
    denominator = np.power(np.maximum(shear2, 1e-24), 1.5)
    eta = 3.0 * np.sqrt(6.0) * np.prod(shear, axis=-1) / denominator
    eta = np.where(shear2 > 1e-20, np.clip(eta, -1.0, 1.0), 0.0)
    return {
        "trace": trace.astype(np.float32),
        "shear_q": q.astype(np.float32),
        "lode_eta": eta.astype(np.float32),
        "gap12": (value[..., 1] - value[..., 0]).astype(np.float32),
        "gap23": (value[..., 2] - value[..., 1]).astype(np.float32),
    }


def _coverage_core_interval(
    samples: np.ndarray,
    truth: np.ndarray,
    groups: np.ndarray,
    probability: float,
    *,
    repeats: int,
    seed: int,
) -> dict:
    tail = (1.0 - float(probability)) / 2.0
    lower, upper = np.quantile(samples, [tail, 1.0 - tail], axis=0)
    covered = (truth >= lower) & (truth <= upper)
    unique, inverse = np.unique(groups, return_inverse=True)
    count = np.bincount(inverse)
    hits = np.bincount(inverse, weights=covered.astype(np.float64))
    rng = np.random.default_rng(seed)
    distribution = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        chosen = rng.integers(0, len(unique), size=len(unique))
        distribution[repeat] = hits[chosen].sum() / count[chosen].sum()
    return {
        "nominal": float(probability),
        "empirical": float(np.mean(covered)),
        "error": float(np.mean(covered) - probability),
        "core_bootstrap_interval95": np.quantile(distribution, [0.025, 0.975]).tolist(),
        "cores": int(len(unique)),
    }


def posterior_summary(
    samples: np.ndarray,
    truth: np.ndarray,
    core_id: np.ndarray,
    *,
    seed: int,
    bootstrap_replicates: int,
) -> dict:
    draws = np.asarray(samples, dtype=np.float32)
    target = np.asarray(truth, dtype=np.float32)
    groups = np.asarray(core_id, dtype=np.int64)
    report = scalar_posterior_report(draws, target, seed=seed)
    report["sbc"] = _sbc(draws, target, groups, seed=seed + 100)
    report["coverage_core_bootstrap"] = {
        str(probability): _coverage_core_interval(
            draws,
            target,
            groups,
            probability,
            repeats=bootstrap_replicates,
            seed=seed + int(1000 * probability),
        )
        for probability in (0.5, 0.68, 0.9)
    }
    return report


def trace_shear_autopsy(config: dict, output_root: Path) -> Path:
    source = source_paths(config)
    ready = json.loads(source["compact_archive_ready"].read_text())
    samples_parts: list[np.ndarray] = []
    truth_parts: list[np.ndarray] = []
    core_parts: list[np.ndarray] = []
    shell_parts: list[np.ndarray] = []
    for row in ready["entries"]:
        path = Path(row["compact_path"])
        if not path.is_file() or sha256(path) != row["compact_sha256"]:
            raise RuntimeError(f"compact artifact changed: {path}")
        with np.load(path, allow_pickle=False) as values:
            sample = np.asarray(values["lambda_samples"], dtype=np.float32)
            truth = np.asarray(values["lambda_truth"], dtype=np.float32)
        samples_parts.append(sample)
        truth_parts.append(truth)
        core_parts.append(np.full(len(truth), int(row["core_id"]), dtype=np.int64))
        shell_parts.append(np.full(len(truth), int(row["shell"]), dtype=np.int8))
    eigen_samples = np.concatenate(samples_parts, axis=1)
    eigen_truth = np.concatenate(truth_parts)
    cores = np.concatenate(core_parts)
    shells = np.concatenate(shell_parts)
    sample_invariant = physical_invariants(eigen_samples)
    truth_invariant = physical_invariants(eigen_truth)
    repeats = int(config["common"]["bootstrap_replicates"])
    scalar = {
        name: posterior_summary(
            sample_invariant[name],
            truth_invariant[name],
            cores,
            seed=220 + index,
            bootstrap_replicates=repeats,
        )
        for index, name in enumerate(config["trace_shear"]["invariants"])
    }
    joint = {
        "trace_shear_shape": tarp_curve(
            np.stack(
                [sample_invariant[name] for name in ("trace", "shear_q", "lode_eta")],
                axis=-1,
            ),
            np.stack(
                [truth_invariant[name] for name in ("trace", "shear_q", "lode_eta")],
                axis=-1,
            ),
            seed=42,
        ),
        "shear_shape": tarp_curve(
            np.stack([sample_invariant["shear_q"], sample_invariant["lode_eta"]], axis=-1),
            np.stack([truth_invariant["shear_q"], truth_invariant["lode_eta"]], axis=-1),
            seed=43,
        ),
        "eigengaps": tarp_curve(
            np.stack([sample_invariant["gap12"], sample_invariant["gap23"]], axis=-1),
            np.stack([truth_invariant["gap12"], truth_invariant["gap23"]], axis=-1),
            seed=44,
        ),
    }
    by_shell = {}
    for shell in range(4):
        selected = shells == shell
        by_shell[str(shell)] = {
            "rows": int(np.count_nonzero(selected)),
            "trace_shear_shape": tarp_curve(
                np.stack(
                    [sample_invariant[name][:, selected] for name in ("trace", "shear_q", "lode_eta")],
                    axis=-1,
                ),
                np.stack(
                    [truth_invariant[name][selected] for name in ("trace", "shear_q", "lode_eta")],
                    axis=-1,
                ),
                seed=52 + shell,
            ),
            "shear_shape": tarp_curve(
                np.stack(
                    [sample_invariant["shear_q"][:, selected], sample_invariant["lode_eta"][:, selected]],
                    axis=-1,
                ),
                np.stack(
                    [truth_invariant["shear_q"][selected], truth_invariant["lode_eta"][selected]],
                    axis=-1,
                ),
                seed=62 + shell,
            ),
        }
    report = {
        "schema_version": "p12f-trace-shear-autopsy-v1",
        "created_utc": utc_now(),
        "phase": "ph006",
        "method": "gaussian_correlated_g1",
        "draws": int(eigen_samples.shape[0]),
        "rows": int(len(eigen_truth)),
        "cores": int(len(np.unique(cores))),
        "scalar": scalar,
        "joint_tarp": joint,
        "shell_tarp": by_shell,
        "interpretation_guard": "trace calibration and algebraic trace closure are distinct",
        "production_effect": "none",
        "ph001_opened": False,
    }
    path = output_root / "P12F_TRACE_SHEAR_AUTOPSY.json"
    atomic_json(path, report)
    return path


def selected_autopsy_entries(panel: dict, entries: list[dict], expected: int = 256) -> list[dict]:
    metadata = {int(row["core_id"]): row for row in panel["selected_core_metadata"]}
    core_id = np.asarray([int(row["core_id"]) for row in entries], dtype=np.int64)
    shell = np.asarray([int(metadata[int(value)]["shell"]) for value in core_id], dtype=np.int8)
    labels = _subpanel_labels(core_id, shell)
    chosen = [row for row, label in zip(entries, labels, strict=True) if int(label) == 0]
    if len(chosen) != expected:
        raise RuntimeError(f"causal-autopsy subpanel has {len(chosen)}, expected {expected}")
    counts = np.bincount([int(metadata[int(row["core_id"])]["shell"]) for row in chosen], minlength=4)
    if not np.array_equal(counts, np.full(4, expected // 4)):
        raise RuntimeError("causal-autopsy subpanel is not shell balanced")
    return chosen


def radial_low_k_mask(
    shape: tuple[int, int, int],
    *,
    bins: int,
    corrected_bins: Iterable[int],
    exclude_dc: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kx = np.fft.fftfreq(shape[0])[:, None, None]
    ky = np.fft.fftfreq(shape[1])[None, :, None]
    kz = np.fft.rfftfreq(shape[2])[None, None, :]
    radius = np.sqrt(kx * kx + ky * ky + kz * kz)
    edges = np.linspace(0.0, np.sqrt(3.0) / 2.0 + np.finfo(float).eps, bins + 1)
    label = np.minimum(np.searchsorted(edges[1:], radius, side="right"), bins - 1)
    mask = np.isin(label, np.asarray(tuple(corrected_bins), dtype=np.int64))
    if exclude_dc:
        mask[0, 0, 0] = False
    return mask, label.astype(np.int16), edges


def low_k_intervention(
    samples: np.ndarray,
    truth: np.ndarray,
    *,
    low_mask: np.ndarray,
    bin_label: np.ndarray,
    power_scale_by_bin: np.ndarray,
    correct_mean: bool,
    correct_power: bool,
) -> np.ndarray:
    draws = np.asarray(samples, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    mean = draws.mean(axis=0)
    mean_k = np.fft.rfftn(mean, norm="ortho")
    truth_k = np.fft.rfftn(target, norm="ortho")
    residual_k = np.fft.rfftn(draws - mean[None], axes=(-3, -2, -1), norm="ortho")
    if correct_power:
        scale = np.asarray(power_scale_by_bin, dtype=np.float64)[bin_label]
        residual_k[:, low_mask] *= scale[low_mask][None]
    base = np.broadcast_to(mean_k, residual_k.shape).copy()
    if correct_mean:
        base[:, low_mask] = truth_k[low_mask][None]
    transformed = base + residual_k
    output = np.fft.irfftn(
        transformed,
        s=target.shape,
        axes=(-3, -2, -1),
        norm="ortho",
    ).real
    if not np.all(np.isfinite(output)):
        raise RuntimeError("low-k intervention produced non-finite fields")
    return output.astype(np.float32)


def _power_oracle(entries: list[dict], *, bins: int, corrected_bins: list[int]) -> dict:
    truth_sum = np.zeros(bins, dtype=np.float64)
    posterior_sum = np.zeros(bins, dtype=np.float64)
    count = np.zeros(bins, dtype=np.int64)
    edges_reference = None
    for entry in entries:
        record = load_core_record(entry, 256)
        samples = np.asarray(record["delta_samples"], dtype=np.float64)
        truth = np.asarray(record["delta_truth"], dtype=np.float64)
        mean = samples.mean(axis=0)
        innovation_k = np.fft.rfftn(truth - mean, norm="ortho")
        residual_k = np.fft.rfftn(samples - mean[None], axes=(-3, -2, -1), norm="ortho")
        _, label, edges = radial_low_k_mask(
            truth.shape, bins=bins, corrected_bins=corrected_bins, exclude_dc=True
        )
        if edges_reference is None:
            edges_reference = edges
        elif not np.array_equal(edges_reference, edges):
            raise RuntimeError("low-k spectral edges changed with patch shape")
        truth_power = np.square(np.abs(innovation_k))
        posterior_power = np.mean(np.square(np.abs(residual_k)), axis=0)
        for value in range(bins):
            selected = label == value
            truth_sum[value] += float(np.sum(truth_power[selected]))
            posterior_sum[value] += float(np.sum(posterior_power[selected]))
            count[value] += int(np.count_nonzero(selected))
    truth = truth_sum / np.maximum(count, 1)
    posterior = posterior_sum / np.maximum(count, 1)
    scale = np.ones(bins, dtype=np.float64)
    selected = np.asarray(corrected_bins, dtype=np.int64)
    scale[selected] = np.sqrt(
        np.divide(truth[selected], posterior[selected], out=np.ones_like(truth[selected]), where=posterior[selected] > 0)
    )
    return {
        "edges": edges_reference,
        "truth_power": truth,
        "posterior_power": posterior,
        "mode_count": count,
        "scale": scale,
    }


def _method_lambda_report(parts: list[dict], config: dict, method: str) -> dict:
    samples = np.concatenate([row["samples"] for row in parts], axis=1)
    truth = np.concatenate([row["truth"] for row in parts], axis=0)
    core = np.concatenate(
        [np.full(len(row["truth"]), row["core_id"], dtype=np.int64) for row in parts]
    )
    gap = samples[..., 1:] - samples[..., :-1]
    gap_truth = truth[..., 1:] - truth[..., :-1]
    report = {
        "method": method,
        "draws": int(samples.shape[0]),
        "rows": int(len(truth)),
        "cores": int(len(parts)),
        "ordered_eigenvalue_tarp": tarp_curve(samples, truth, seed=42),
        "eigengap_tarp": tarp_curve(gap, gap_truth, seed=43),
        "scalar": {},
    }
    repeats = int(config["common"]["bootstrap_replicates"])
    names = ("lambda1", "lambda2", "lambda3")
    for index, name in enumerate(names):
        report["scalar"][name] = posterior_summary(
            samples[..., index], truth[..., index], core,
            seed=300 + index, bootstrap_replicates=repeats,
        )
    for index, name in enumerate(("gap12", "gap23")):
        report["scalar"][name] = posterior_summary(
            gap[..., index], gap_truth[..., index], core,
            seed=310 + index, bootstrap_replicates=repeats,
        )
    return report


def low_k_autopsy(config: dict, panel: dict, entries: list[dict], output_root: Path, device: str) -> Path:
    contract = config["low_k_intervention"]
    selected = selected_autopsy_entries(panel, entries, int(contract["cores"]))
    power = _power_oracle(
        selected,
        bins=int(contract["radial_bins"]),
        corrected_bins=[int(value) for value in contract["corrected_radial_bins"]],
    )
    source = source_paths(config)
    compact_ready = json.loads(source["compact_archive_ready"].read_text())
    compact_map = {int(row["core_id"]): row for row in compact_ready["entries"]}
    output = output_root / "low_k_compact"
    output.mkdir(parents=True, exist_ok=True)
    metadata = {int(row["core_id"]): row for row in panel["selected_core_metadata"]}
    method_parts = {name: [] for name in contract["interventions"]}
    score = {name: [] for name in contract["interventions"]}
    for ordinal, entry in enumerate(selected):
        core_id = int(entry["core_id"])
        path = output / f"core_{core_id:08d}.npz"
        score_path = output / f"core_{core_id:08d}.json"
        if not path.exists() or not score_path.exists():
            record = load_core_record(entry, 256)
            samples = np.asarray(record["delta_samples"], dtype=np.float32)
            truth = np.asarray(record["delta_truth"], dtype=np.float32)
            low_mask, label, _ = radial_low_k_mask(
                truth.shape,
                bins=int(contract["radial_bins"]),
                corrected_bins=contract["corrected_radial_bins"],
                exclude_dc=bool(contract["exclude_dc"]),
            )
            variants = {
                "lowk_mean_oracle": low_k_intervention(
                    samples, truth, low_mask=low_mask, bin_label=label,
                    power_scale_by_bin=power["scale"], correct_mean=True, correct_power=False,
                ),
                "lowk_power_oracle": low_k_intervention(
                    samples, truth, low_mask=low_mask, bin_label=label,
                    power_scale_by_bin=power["scale"], correct_mean=False, correct_power=True,
                ),
                "lowk_mean_power_oracle": low_k_intervention(
                    samples, truth, low_mask=low_mask, bin_label=label,
                    power_scale_by_bin=power["scale"], correct_mean=True, correct_power=True,
                ),
            }
            coordinates = np.asarray(record["galaxy_frac_index_local"], dtype=np.float32)
            compact_row = compact_map[core_id]
            with np.load(compact_row["compact_path"], allow_pickle=False) as cached:
                g1_lambda = np.asarray(cached["lambda_samples"], dtype=np.float32)
                lambda_truth = np.asarray(cached["lambda_truth"], dtype=np.float32)
            lambda_variants = {"g1": g1_lambda}
            physics = {}
            for name, fields in variants.items():
                value, closure = tidal_eigenvalues_at_galaxies(
                    torch.from_numpy(fields).to(device), coordinates
                )
                lambda_variants[name] = value
                physics[name] = closure
            atomic_npz(
                path,
                lambda_truth=lambda_truth,
                **{f"lambda_{name}": value for name, value in lambda_variants.items()},
            )
            core_slice = tuple(
                slice(int(left), int(right))
                for left, right in zip(record["core_bounds"][0], record["core_bounds"][1], strict=True)
            )
            support_core = np.asarray(record["support"], dtype=bool)[core_slice]
            truth_core = truth[core_slice]
            fields_by_method = {"g1": samples}
            fields_by_method.update(variants)
            row_scores = {
                name: core_joint_scores(
                    fields[(slice(None),) + core_slice][:64], truth_core, support_core,
                    feature_count=512, pair_count=1024, seed=42000 + core_id,
                )
                for name, fields in fields_by_method.items()
            }
            atomic_json(
                score_path,
                {
                    "schema_version": "p12f-low-k-core-v1",
                    "core_id": core_id,
                    "shell": int(metadata[core_id]["shell"]),
                    "npz_sha256": sha256(path),
                    "physics": physics,
                    "proper_scores_64_draws": row_scores,
                    "ph001_opened": False,
                },
            )
        row = json.loads(score_path.read_text())
        if row.get("ph001_opened") or row.get("npz_sha256") != sha256(path):
            raise RuntimeError("unsafe or stale low-k compact core")
        with np.load(path, allow_pickle=False) as values:
            truth_lambda = np.asarray(values["lambda_truth"], dtype=np.float32)
            for name in method_parts:
                method_parts[name].append(
                    {
                        "core_id": core_id,
                        "samples": np.asarray(values[f"lambda_{name}"], dtype=np.float32),
                        "truth": truth_lambda,
                    }
                )
                score[name].append(row["proper_scores_64_draws"][name])
        print(json.dumps({"low_k_core": ordinal + 1, "total": len(selected), "core_id": core_id}), flush=True)
    reports = {
        name: _method_lambda_report(method_parts[name], config, name)
        for name in method_parts
    }
    for name in reports:
        reports[name]["proper_scores_64_draws"] = {
            key: float(np.mean([row[key] for row in score[name]]))
            for key in ("energy", "coarse_energy", "variogram_p0p5")
        }
    report = {
        "schema_version": "p12f-low-k-causal-autopsy-v1",
        "created_utc": utc_now(),
        "phase": "ph006",
        "selected_core_ids": [int(row["core_id"]) for row in selected],
        "shell_counts": np.bincount(
            [int(metadata[int(row["core_id"])]["shell"]) for row in selected], minlength=4
        ).tolist(),
        "truth_assisted": True,
        "production_sampler": False,
        "low_k_contract": contract,
        "power_oracle": {
            "edges_cycles_per_voxel": power["edges"].tolist(),
            "truth_innovation_power": power["truth_power"].tolist(),
            "posterior_residual_power": power["posterior_power"].tolist(),
            "scatter_scale": power["scale"].tolist(),
            "mode_count": power["mode_count"].tolist(),
        },
        "methods": reports,
        "ph001_opened": False,
    }
    path = output_root / "P12F_LOW_K_CAUSAL_AUTOPSY.json"
    atomic_json(path, report)
    return path


def cube_symmetry(field: np.ndarray, *, seed: int) -> np.ndarray:
    value = np.asarray(field)
    rng = np.random.default_rng(seed)
    eligible = [axis for axis in range(3)]
    permutations = [
        permutation
        for permutation in __import__("itertools").permutations(eligible)
        if tuple(value.shape[index] for index in permutation) == tuple(value.shape)
    ]
    permutation = permutations[int(rng.integers(0, len(permutations)))]
    output = np.transpose(value, permutation)
    for axis in range(3):
        if rng.integers(0, 2):
            output = np.flip(output, axis=axis)
    return np.ascontiguousarray(output)


def build_residual_library(
    config: dict,
    output_root: Path,
    device: str,
    max_wall_seconds: float,
) -> Path:
    source = source_paths(config)
    checkpoint = torch.load(source["gaussian_checkpoint"], map_location=device, weights_only=False)
    run = json.loads(source["gaussian_run_manifest"].read_text())
    if (
        checkpoint.get("schema_version") != "p12f-matched-challenger-checkpoint-v1"
        or checkpoint.get("method") != "gaussian"
        or checkpoint.get("ph001_opened")
        or checkpoint.get("frozen_digest") != run.get("frozen_digest")
    ):
        raise RuntimeError("empirical residual library received unsafe Gaussian provenance")
    phases = tuple(config["roles"]["training"])
    selected = {
        phase: [int(value) for value in run["frozen"]["selected_core_ids"][phase]]
        for phase in phases
    }
    expected = int(config["empirical_residual"]["training_cores_per_phase"])
    if any(len(values) != expected for values in selected.values()) or "ph001" in selected:
        raise RuntimeError("empirical residual library training-core contract changed")
    parent = json.loads(source["parent_config"].read_text())
    loader = P10RandomResponseLoader(source["conditioning_contract"], include_blind=False)
    store = FieldTargetStore(source["phase_root"], phases)
    model = ConditionalGaussianUNet(condition_channels=3, base=int(parent["matched_contract"]["unet_base"])).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    normalization = loader.field_normalization
    flow_parent = json.loads((REPO_ROOT / parent["parent_flow_config"]).read_text())
    halo = int(flow_parent["patch"]["context_halo_voxels"])
    alignment = int(flow_parent["patch"]["alignment_voxels"])
    library_root = output_root / "training_residual_library"
    library_root.mkdir(parents=True, exist_ok=True)
    progress_path = library_root / "PROGRESS.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
        "schema_version": "p12f-training-residual-library-progress-v1",
        "entries": [],
        "ph001_opened": False,
    }
    done = {(row["phase"], int(row["core_id"])): row for row in progress["entries"]}
    started = time.monotonic()
    ordinal = 0
    total = sum(map(len, selected.values()))
    with torch.inference_mode():
        for phase in phases:
            adapter = loader.field_adapter(phase)
            for core_id in selected[phase]:
                ordinal += 1
                key = (phase, core_id)
                if key in done:
                    path = Path(done[key]["path"])
                    if not path.is_file() or sha256(path) != done[key]["sha256"]:
                        raise RuntimeError("residual-library artifact changed before resume")
                    continue
                observed = summarize_observed_core(adapter, core_id)
                if observed is None:
                    raise RuntimeError("registered residual-library core has no support")
                patch = adapter.extract(core_id, halo, CHANNELS, alignment_voxels=alignment)
                condition, _ = model_inputs(patch, normalization, device)
                target_data = store.extract(phase, patch)
                target = target_tensor(target_data["delta"], checkpoint["target_scaler"], device)
                mean, log_std = model(condition)
                residual = ((target - mean) / torch.exp(log_std))[0, 0].cpu().numpy().astype(np.float32)
                support = np.asarray(target_data["support"], dtype=np.uint8)
                path = library_root / f"{phase}_core_{core_id:08d}.npz"
                atomic_npz(path, normalized_residual=residual, support=support)
                row = {
                    "phase": phase,
                    "core_id": core_id,
                    "path": str(path.resolve()),
                    "sha256": sha256(path),
                    "shape": list(residual.shape),
                    "shell": int(observed["shell"]),
                    "cap": int(observed["cap"]),
                    "support_fraction": float(np.mean(support)),
                }
                progress["entries"].append(row)
                done[key] = row
                atomic_json(progress_path, progress)
                print(json.dumps({"residual_library": ordinal, "total": total, "phase": phase, "core_id": core_id}), flush=True)
                if time.monotonic() - started >= max_wall_seconds:
                    store.close()
                    loader.close()
                    raise SystemExit(75)
    entries = [done[(phase, core_id)] for phase in phases for core_id in selected[phase]]
    manifest = {
        "schema_version": "p12f-training-residual-library-v1",
        "created_utc": utc_now(),
        "entries": entries,
        "fields": len(entries),
        "training_phases": list(phases),
        "checkpoint_sha256": sha256(source["gaussian_checkpoint"]),
        "run_manifest_sha256": sha256(source["gaussian_run_manifest"]),
        "target_scaler": checkpoint["target_scaler"],
        "raw_residual_outside_support_retained": True,
        "validation_phase_read": False,
        "ph001_opened": False,
        "pass": True,
    }
    path = library_root / "P12F_TRAINING_RESIDUAL_LIBRARY.json"
    atomic_json(path, manifest)
    store.close()
    loader.close()
    return path


def _donor_order(entries: list[dict], *, shape: tuple[int, ...], shell: int, cap: int, support_fraction: float, seed: int) -> list[dict]:
    exact = [row for row in entries if tuple(row["shape"]) == shape]
    if not exact:
        raise RuntimeError(f"no empirical residual donor has shape {shape}")
    rng = np.random.default_rng(seed)
    tie = rng.random(len(exact))
    order = sorted(
        range(len(exact)),
        key=lambda index: (
            int(int(exact[index]["shell"]) != shell),
            int(int(exact[index]["cap"]) != cap),
            abs(float(exact[index]["support_fraction"]) - support_fraction),
            float(tie[index]),
        ),
    )
    return [exact[index] for index in order]


def empirical_residual_autopsy(
    config: dict,
    panel: dict,
    entries: list[dict],
    output_root: Path,
    library_path: Path,
    device: str,
) -> Path:
    library = json.loads(library_path.read_text())
    if (
        library.get("schema_version") != "p12f-training-residual-library-v1"
        or not library.get("pass")
        or library.get("validation_phase_read")
        or library.get("ph001_opened")
        or int(library.get("fields", -1)) != int(config["empirical_residual"]["library_fields"])
    ):
        raise RuntimeError("empirical residual library is incomplete or unsafe")
    source = source_paths(config)
    checkpoint = torch.load(source["gaussian_checkpoint"], map_location=device, weights_only=False)
    parent = json.loads(source["parent_config"].read_text())
    flow_parent = json.loads((REPO_ROOT / parent["parent_flow_config"]).read_text())
    loader = P10RandomResponseLoader(source["conditioning_contract"], include_blind=False)
    adapter = loader.field_adapter("ph006")
    model = ConditionalGaussianUNet(condition_channels=3, base=int(parent["matched_contract"]["unet_base"])).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    selected = selected_autopsy_entries(panel, entries, int(config["empirical_residual"]["recipient_cores"]))
    metadata = {int(row["core_id"]): row for row in panel["selected_core_metadata"]}
    compact_ready = json.loads(source["compact_archive_ready"].read_text())
    compact_map = {int(row["core_id"]): row for row in compact_ready["entries"]}
    normalization = loader.field_normalization
    halo = int(flow_parent["patch"]["context_halo_voxels"])
    alignment = int(flow_parent["patch"]["alignment_voxels"])
    draws = int(config["empirical_residual"]["draws"])
    output = output_root / "empirical_compact"
    output.mkdir(parents=True, exist_ok=True)
    parts = {"g1": [], "empirical_residual_patch": []}
    scores = {"g1": [], "empirical_residual_patch": []}
    donor_rows = library["entries"]
    donor_cache: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for ordinal, entry in enumerate(selected):
            core_id = int(entry["core_id"])
            path = output / f"core_{core_id:08d}.npz"
            score_path = output / f"core_{core_id:08d}.json"
            if not path.exists() or not score_path.exists():
                record = load_core_record(entry, 256)
                patch = adapter.extract(core_id, halo, CHANNELS, alignment_voxels=alignment)
                condition, _ = model_inputs(patch, normalization, device)
                mean, log_std = model(condition)
                mean = mean[0, 0].cpu().numpy()
                standard = torch.exp(log_std[0, 0]).cpu().numpy()
                support = np.asarray(record["support"], dtype=bool)
                info = metadata[core_id]
                ordered = _donor_order(
                    donor_rows,
                    shape=tuple(mean.shape),
                    shell=int(info["shell"]),
                    cap=int(info["cap"]),
                    support_fraction=float(np.mean(support)),
                    seed=81000 + core_id,
                )
                rng = np.random.default_rng(91000 + core_id)
                pool = ordered[: min(128, len(ordered))]
                chosen = rng.integers(0, len(pool), size=draws)
                residual = np.empty((draws, *mean.shape), dtype=np.float32)
                donor_identity = []
                for draw, index in enumerate(chosen):
                    donor = pool[int(index)]
                    donor_path = str(donor["path"])
                    if donor_path not in donor_cache:
                        with np.load(donor_path, allow_pickle=False) as values:
                            donor_cache[donor_path] = np.asarray(
                                values["normalized_residual"], dtype=np.float32
                            )
                    value = donor_cache[donor_path]
                    residual[draw] = cube_symmetry(value, seed=92000 + core_id * draws + draw)
                    donor_identity.append(f"{donor['phase']}:{donor['core_id']}")
                scaled = mean[None] + standard[None] * residual
                fields = unscale(scaled, checkpoint["target_scaler"])
                lambda_empirical, closure = tidal_eigenvalues_at_galaxies(
                    torch.from_numpy(fields).to(device),
                    np.asarray(record["galaxy_frac_index_local"], dtype=np.float32),
                )
                compact_row = compact_map[core_id]
                with np.load(compact_row["compact_path"], allow_pickle=False) as cached:
                    lambda_g1 = np.asarray(cached["lambda_samples"], dtype=np.float32)
                    lambda_truth = np.asarray(cached["lambda_truth"], dtype=np.float32)
                atomic_npz(path, lambda_truth=lambda_truth, lambda_g1=lambda_g1, lambda_empirical_residual_patch=lambda_empirical)
                core_slice = tuple(
                    slice(int(left), int(right))
                    for left, right in zip(record["core_bounds"][0], record["core_bounds"][1], strict=True)
                )
                truth_core = np.asarray(record["delta_truth"], dtype=np.float32)[core_slice]
                support_core = support[core_slice]
                score_row = {
                    "g1": core_joint_scores(
                        np.asarray(record["delta_samples"], dtype=np.float32)[:64][(slice(None),) + core_slice],
                        truth_core, support_core, feature_count=512, pair_count=1024, seed=93000 + core_id,
                    ),
                    "empirical_residual_patch": core_joint_scores(
                        fields[:64][(slice(None),) + core_slice], truth_core, support_core,
                        feature_count=512, pair_count=1024, seed=93000 + core_id,
                    ),
                }
                atomic_json(
                    score_path,
                    {
                        "schema_version": "p12f-empirical-residual-core-v1",
                        "core_id": core_id,
                        "npz_sha256": sha256(path),
                        "donor_pool": len(pool),
                        "unique_donors": len(set(donor_identity)),
                        "physics": closure,
                        "proper_scores_64_draws": score_row,
                        "ph001_opened": False,
                    },
                )
            row = json.loads(score_path.read_text())
            if row.get("ph001_opened") or row.get("npz_sha256") != sha256(path):
                raise RuntimeError("unsafe or stale empirical-residual core")
            with np.load(path, allow_pickle=False) as values:
                truth_lambda = np.asarray(values["lambda_truth"], dtype=np.float32)
                for name in parts:
                    parts[name].append({
                        "core_id": core_id,
                        "samples": np.asarray(values[f"lambda_{name}"], dtype=np.float32),
                        "truth": truth_lambda,
                    })
                    scores[name].append(row["proper_scores_64_draws"][name])
            print(json.dumps({"empirical_core": ordinal + 1, "total": len(selected), "core_id": core_id}), flush=True)
    reports = {name: _method_lambda_report(parts[name], config, name) for name in parts}
    for name in reports:
        reports[name]["proper_scores_64_draws"] = {
            key: float(np.mean([row[key] for row in scores[name]]))
            for key in ("energy", "coarse_energy", "variogram_p0p5")
        }
    report = {
        "schema_version": "p12f-empirical-residual-causal-autopsy-v1",
        "created_utc": utc_now(),
        "phase": "ph006",
        "selected_core_ids": [int(row["core_id"]) for row in selected],
        "training_residual_library": str(library_path.resolve()),
        "training_residual_library_sha256": sha256(library_path),
        "methods": reports,
        "fit_on_ph006": False,
        "production_sampler": False,
        "ph001_opened": False,
    }
    path = output_root / "P12F_EMPIRICAL_RESIDUAL_CAUSAL_AUTOPSY.json"
    atomic_json(path, report)
    loader.close()
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("trace-shear", "low-k", "build-residual-library", "empirical-residual", "all"),
        default="all",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-wall-seconds", type=float, default=13_500.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "ph001" in str(args.output_root).lower():
        raise PermissionError("the sealed blind phase appeared in the output path")
    config = load_config(args.config)
    panel, entries = validate_frozen_sources(args.config, config)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "run_manifest.json"
    manifest = frozen_manifest(args.config, config, panel, entries)
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old.get("frozen_digest") != manifest["frozen_digest"]:
            raise RuntimeError("causal-autopsy frozen contract changed")
    else:
        atomic_json(manifest_path, manifest)
    stages = (
        ("trace-shear", "low-k", "build-residual-library", "empirical-residual")
        if args.stage == "all"
        else (args.stage,)
    )
    if any(stage in {"low-k", "build-residual-library", "empirical-residual"} for stage in stages):
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("field causal autopsy requires a compute GPU")
    outputs = {}
    for stage in stages:
        if stage == "trace-shear":
            outputs[stage] = str(trace_shear_autopsy(config, args.output_root))
        elif stage == "low-k":
            outputs[stage] = str(low_k_autopsy(config, panel, entries, args.output_root, args.device))
        elif stage == "build-residual-library":
            outputs[stage] = str(
                build_residual_library(config, args.output_root, args.device, args.max_wall_seconds)
            )
        elif stage == "empirical-residual":
            library = args.output_root / "training_residual_library/P12F_TRAINING_RESIDUAL_LIBRARY.json"
            outputs[stage] = str(
                empirical_residual_autopsy(config, panel, entries, args.output_root, library, args.device)
            )
    print(json.dumps({"outputs": outputs, "ph001_opened": False}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
