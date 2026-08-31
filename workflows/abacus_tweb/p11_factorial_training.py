#!/usr/bin/env python3
"""Freeze and consume the P11 dense-view U-PATCH response adapter.

The dense teacher uses the same three-channel U-PATCH interface as P10:
log-count, density contrast, and exposure.  Only the observed count field and
its view-specific selection curve change.  The fit uses ph002--ph005 only;
ph006 is application-only and ph001 is never opened.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

import h5py
import numpy as np

from workflows.abacus_tweb.p10_training_contract import (
    P10PhaseBalancedLoader,
    atomic_json,
    sha256,
)
from workflows.abacus_tweb.p6_field_patch_utils import (
    CAP_NAME,
    FieldPatch,
    channel_transform,
    derive_selection_channels,
    patch_redshift,
)
from workflows.abacus_tweb.p6_refit_fullcap_selection import fit_log_spline


P11_TRAINING_PHASES = ("ph002", "ph003", "ph004", "ph005")
P11_VALIDATION_PHASE = "ph006"
P11_SEALED_PHASE = "ph001"
DEFAULT_ROOT = Path(
    "/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p11_factorial_views_v1"
)
DEFAULT_CONTRACT = DEFAULT_ROOT / "dense_response_adapter_v1"
P10_CONTRACT = Path(
    "/pscratch/sd/d/dkololgi/abacus/p10_multiphase/training_contract"
)
EDGES = np.arange(0.10, 0.6001, 0.005)
GRID_Z = np.arange(0.10, 0.6001, 0.001)
FIT_Z_MIN = 0.15
FIT_Z_MAX = 0.55
KNOT_SPACING = 0.05
MINIMUM_EXPOSURE = 1.0e-4
EPSILON = 1.0e-3
CHANNELS = ("counts", "exposure_apodized", "log_count_ratio")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _stage_component(root: Path, phase: str, cap_name: str) -> tuple[dict, dict]:
    phase_manifest = _load(root / phase / "PHASE_FACTORIAL_VIEW_COUNTS_READY.json")
    if (
        not phase_manifest.get("pass")
        or phase_manifest.get("sealed_phase_opened")
        or phase_manifest.get("truth_or_targets_read")
    ):
        raise RuntimeError(f"{phase} factorial-view contract does not pass")
    stage = phase_manifest["V_dense"]["counts"]["components"][cap_name]
    response_manifest = _load(Path(phase_manifest["response_manifest"]))
    response = response_manifest["components"][cap_name]
    return stage, response


def _chunk_redshift(
    selection: tuple[slice, slice, slice],
    response_handle: h5py.File,
    radius_grid: np.ndarray,
    redshift_grid: np.ndarray,
) -> np.ndarray:
    start = np.asarray([part.start for part in selection], dtype=np.int64)
    shape = tuple(part.stop - part.start for part in selection)
    return patch_redshift(
        origin_mpc=np.asarray(response_handle.attrs["origin_mpc"], dtype=np.float64),
        cell_mpc=float(response_handle.attrs["cell_mpc"]),
        context_start=start,
        shape=shape,
        radius_grid_mpc=radius_grid,
        redshift_grid=redshift_grid,
    )


def _iter_dense_chunks(
    root: Path,
    phase: str,
    cap_name: str,
    radius_grid: np.ndarray,
    redshift_grid: np.ndarray,
):
    stage, response = _stage_component(root, phase, cap_name)
    with h5py.File(stage["file"], "r") as counts_handle, h5py.File(
        response["file"], "r"
    ) as response_handle:
        dataset = counts_handle["bright_counts"]
        for selection in dataset.iter_chunks():
            counts = np.asarray(dataset[selection], dtype=np.float32)
            support_exposure = np.asarray(
                response_handle["exposure_apodized_random"][selection], dtype=np.float32
            )
            angular_response = np.asarray(
                response_handle["angular_response"][selection], dtype=np.float32
            )
            response_exposure = support_exposure * np.maximum(angular_response, 0.0)
            redshift = _chunk_redshift(
                selection, response_handle, radius_grid, redshift_grid
            )
            yield (
                counts,
                support_exposure,
                response_exposure,
                redshift,
                float(response_handle.attrs["cell_mpc"]),
            )


def _accumulate_moments(values: np.ndarray, accumulator: list[float]) -> None:
    values = np.asarray(values, dtype=np.float64)
    accumulator[0] += int(values.size)
    accumulator[1] += float(values.sum(dtype=np.float64))
    accumulator[2] += float(np.square(values).sum(dtype=np.float64))


def build_dense_contract(root: Path, output: Path, p10_contract: Path) -> dict:
    products = _load(root / "FACTORIAL_VIEW_PRODUCTS_READY.json")
    if (
        not products.get("pass")
        or products.get("sealed_phase_opened")
        or products.get("truth_or_targets_read")
    ):
        raise RuntimeError("passing visible-phase factorial products are required")
    p10_selection_path = p10_contract / "transforms/field/selection_manifest.json"
    p10_selection = _load(p10_selection_path)
    radius_grid = np.asarray(
        p10_selection["cosmology"]["radius_grid_mpc"], dtype=np.float64
    )
    redshift_grid = np.asarray(
        p10_selection["cosmology"]["redshift_grid"], dtype=np.float64
    )
    centers = 0.5 * (EDGES[:-1] + EDGES[1:])
    counts_total = {name: np.zeros(len(centers), dtype=np.float64) for name in CAP_NAME.values()}
    volume_total = {name: np.zeros(len(centers), dtype=np.float64) for name in CAP_NAME.values()}
    sources: dict[str, dict] = {}
    for phase in P11_TRAINING_PHASES:
        sources[phase] = {}
        for cap_name in CAP_NAME.values():
            chunks = 0
            for counts, support, response, redshift, cell_mpc in _iter_dense_chunks(
                root, phase, cap_name, radius_grid, redshift_grid
            ):
                selected = support > MINIMUM_EXPOSURE
                counts_total[cap_name] += np.histogram(
                    redshift[selected], bins=EDGES, weights=counts[selected]
                )[0]
                volume_total[cap_name] += np.histogram(
                    redshift[selected],
                    bins=EDGES,
                    weights=(cell_mpc ** 3) * response[selected],
                )[0]
                chunks += 1
            stage, response_row = _stage_component(root, phase, cap_name)
            sources[phase][cap_name] = {
                "counts_file": stage["file"],
                "counts_file_sha256": stage["file_sha256"],
                "response_file": response_row["file"],
                "response_file_sha256": response_row["file_sha256"],
                "chunks": chunks,
            }

    caps = {}
    for cap_name in CAP_NAME.values():
        curve, fit = fit_log_spline(
            centers,
            counts_total[cap_name],
            volume_total[cap_name],
            GRID_Z,
            knot_spacing=KNOT_SPACING,
            fit_z_min=FIT_Z_MIN,
            fit_z_max=FIT_Z_MAX,
        )
        expected = np.interp(centers, GRID_Z, curve) * volume_total[cap_name]
        closure = []
        for low, high in ((0.15, 0.25), (0.25, 0.35), (0.35, 0.45), (0.45, 0.55)):
            chosen = (centers >= low) & (centers < high)
            observed = float(counts_total[cap_name][chosen].sum())
            predicted = float(expected[chosen].sum())
            closure.append(
                {
                    "z_low": low,
                    "z_high": high,
                    "observed": observed,
                    "expected": predicted,
                    "fractional_error": predicted / observed - 1.0,
                }
            )
        caps[cap_name] = {
            "grid_z": GRID_Z.tolist(),
            "ntilde": curve.tolist(),
            "fit": fit,
            "training_shell_closure": closure,
        }

    per_phase: dict[str, dict] = {}
    for phase in P11_TRAINING_PHASES:
        accum = {
            "counts": [0, 0.0, 0.0],
            "log_count_ratio": [0, 0.0, 0.0],
        }
        for cap_name in CAP_NAME.values():
            curve = caps[cap_name]
            for counts, support, response, redshift, cell_mpc in _iter_dense_chunks(
                root, phase, cap_name, radius_grid, redshift_grid
            ):
                derived = derive_selection_channels(
                    counts,
                    response,
                    redshift,
                    cell_mpc=cell_mpc,
                    grid_z=np.asarray(curve["grid_z"]),
                    ntilde=np.asarray(curve["ntilde"]),
                    epsilon=EPSILON,
                    minimum_exposure=MINIMUM_EXPOSURE,
                )
                selected = support > MINIMUM_EXPOSURE
                _accumulate_moments(
                    channel_transform("counts", counts[selected]), accum["counts"]
                )
                _accumulate_moments(
                    derived["log_count_ratio"][selected], accum["log_count_ratio"]
                )
        per_phase[phase] = {}
        for name, (n, total, total2) in accum.items():
            mean = total / n
            per_phase[phase][name] = {
                "count": int(n),
                "mean": mean,
                "second_moment": total2 / n,
                "std": max(total2 / n - mean * mean, 0.0) ** 0.5,
            }

    normalization = {"channels": {}}
    for name in ("counts", "log_count_ratio"):
        mean = float(np.mean([per_phase[p][name]["mean"] for p in P11_TRAINING_PHASES]))
        second = float(
            np.mean([per_phase[p][name]["second_moment"] for p in P11_TRAINING_PHASES])
        )
        normalization["channels"][name] = {
            "policy": "zscore",
            "mean": mean,
            "std": max(second - mean * mean, 0.0) ** 0.5,
        }
    normalization["channels"]["exposure_apodized"] = {"policy": "identity"}
    gates = {
        "training_phases_only": True,
        "ph006_application_only": True,
        "ph001_not_opened": True,
        "finite_positive_curves": all(
            np.all(np.isfinite(row["ntilde"])) and np.all(np.asarray(row["ntilde"]) > 0)
            for row in caps.values()
        ),
        "shell_closure_below_10pct": all(
            abs(shell["fractional_error"]) < 0.10
            for row in caps.values()
            for shell in row["training_shell_closure"]
        ),
        "finite_nonzero_normalization": all(
            np.isfinite(row.get("mean", 0.0)) and row.get("std", 1.0) > 0
            for row in normalization["channels"].values()
        ),
    }
    contract = {
        "schema_version": "p11-dense-response-adapter-v1",
        "created_utc": utc_now(),
        "view": "V_dense",
        "tracer": "BGS_BRIGHT",
        "training_phases": list(P11_TRAINING_PHASES),
        "validation_phase": P11_VALIDATION_PHASE,
        "sealed_phase": P11_SEALED_PHASE,
        "sealed_phase_opened": False,
        "truth_or_targets_read": False,
        "channel_order": list(CHANNELS),
        "model_mapping": [
            "zscored_log1p_counts",
            "clipped_expm1_log_count_ratio",
            "common_random_support_exposure",
        ],
        "response_contract": {
            "support": "P3b-R exposure_apodized_random",
            "angular_response": "P3b-R angular_response enters mu, not the third channel",
            "mu": "ntilde_dense(z) * voxel_volume * angular_response * support_exposure",
            "fibre_assignment": "not applicable to V_dense",
            "redshift_success": "C_z=1 in current mocks",
        },
        "caps": caps,
        "normalization": normalization,
        "per_phase_moments": per_phase,
        "sources": sources,
        "factorial_products_marker": str(root / "FACTORIAL_VIEW_PRODUCTS_READY.json"),
        "factorial_products_marker_sha256": sha256(
            root / "FACTORIAL_VIEW_PRODUCTS_READY.json"
        ),
        "p10_selection_manifest": str(p10_selection_path),
        "p10_selection_manifest_sha256": sha256(p10_selection_path),
        "gates": gates,
        "pass": bool(all(gates.values())),
    }
    if not contract["pass"]:
        raise RuntimeError(f"P11 dense response-adapter gates failed: {gates}")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "P11_DENSE_RESPONSE_ADAPTER_READY.json", contract)
    return contract


class P11DenseFieldAdapter:
    """Overlay dense targetable counts and their frozen response on P10 patch geometry."""

    def __init__(self, *, loader: P10PhaseBalancedLoader, phase: str, root: Path, contract: Path):
        if phase not in P11_TRAINING_PHASES + (P11_VALIDATION_PHASE,):
            raise ValueError(f"phase is outside visible P11 roles: {phase}")
        self.phase = phase
        self.root = Path(root)
        self.contract_path = Path(contract) / "P11_DENSE_RESPONSE_ADAPTER_READY.json"
        self.contract = _load(self.contract_path)
        if not self.contract.get("pass") or self.contract.get("sealed_phase_opened"):
            raise RuntimeError("P11 dense response adapter does not pass")
        self.base = P10PhaseBalancedLoader.field_adapter(loader, phase)
        self.core_cap = self.base.core_cap
        self.stage_manifest = _load(
            self.root / phase / "PHASE_FACTORIAL_VIEW_COUNTS_READY.json"
        )["V_dense"]["counts"]["components"]
        phase_manifest = _load(
            self.root / phase / "PHASE_FACTORIAL_VIEW_COUNTS_READY.json"
        )
        response_manifest = _load(Path(phase_manifest["response_manifest"]))
        self.response_components = response_manifest["components"]
        p10_selection = _load(
            Path(self.contract["p10_selection_manifest"])
        )
        self.radius_grid = np.asarray(
            p10_selection["cosmology"]["radius_grid_mpc"], dtype=np.float64
        )
        self.redshift_grid = np.asarray(
            p10_selection["cosmology"]["redshift_grid"], dtype=np.float64
        )
        self.count_handles: dict[int, h5py.File] = {}
        self.response_handles: dict[int, h5py.File] = {}

    def close(self) -> None:
        for handle in tuple(self.count_handles.values()) + tuple(self.response_handles.values()):
            handle.close()
        self.count_handles.clear()
        self.response_handles.clear()

    def _handles(self, cap: int) -> tuple[h5py.File, h5py.File]:
        cap = int(cap)
        name = CAP_NAME[cap]
        if cap not in self.count_handles:
            self.count_handles[cap] = h5py.File(
                self.stage_manifest[name]["file"], "r"
            )
            self.response_handles[cap] = h5py.File(
                self.response_components[name]["file"], "r"
            )
        return self.count_handles[cap], self.response_handles[cap]

    def extract(
        self,
        core_id: int,
        context_halo_voxels,
        channel_names=CHANNELS,
        *,
        alignment_voxels: int = 1,
    ) -> FieldPatch:
        names = tuple(channel_names)
        if names != CHANNELS:
            raise ValueError(f"P11 dense adapter requires channel order {CHANNELS}")
        geometry = self.base.extract(
            core_id,
            context_halo_voxels,
            ("exposure_binary",),
            alignment_voxels=alignment_voxels,
        )
        selection = tuple(
            slice(int(start), int(stop))
            for start, stop in zip(geometry.context_start, geometry.context_stop)
        )
        count_handle, response_handle = self._handles(geometry.cap)
        counts = np.asarray(count_handle["bright_counts"][selection], dtype=np.float32)
        support = np.asarray(
            response_handle["exposure_apodized_random"][selection], dtype=np.float32
        )
        angular = np.asarray(
            response_handle["angular_response"][selection], dtype=np.float32
        )
        response = support * np.maximum(angular, 0.0)
        redshift = _chunk_redshift(
            selection, response_handle, self.radius_grid, self.redshift_grid
        )
        curve = self.contract["caps"][CAP_NAME[geometry.cap]]
        derived = derive_selection_channels(
            counts,
            response,
            redshift,
            cell_mpc=float(response_handle.attrs["cell_mpc"]),
            grid_z=np.asarray(curve["grid_z"]),
            ntilde=np.asarray(curve["ntilde"]),
            epsilon=EPSILON,
            minimum_exposure=MINIMUM_EXPOSURE,
        )
        values = np.stack(
            (counts, support, derived["log_count_ratio"]), axis=0
        ).astype(np.float32)
        return replace(geometry, channel_names=CHANNELS, values=values)


class P11DensePhaseBalancedLoader(P10PhaseBalancedLoader):
    """P10 target/sampling contract with P11-only phase roles and dense fields."""

    def __init__(
        self,
        contract_root: Path | str = P10_CONTRACT,
        *,
        factorial_root: Path | str = DEFAULT_ROOT,
        adapter_contract: Path | str = DEFAULT_CONTRACT,
    ):
        super().__init__(contract_root, include_blind=False)
        self.training_phases = P11_TRAINING_PHASES
        self.validation_phase = P11_VALIDATION_PHASE
        self.factorial_root = Path(factorial_root)
        self.adapter_contract = Path(adapter_contract)
        self._p11_field: dict[str, P11DenseFieldAdapter] = {}

    def field_adapter(self, phase: str) -> P11DenseFieldAdapter:
        if phase not in self._p11_field:
            self._p11_field[phase] = P11DenseFieldAdapter(
                loader=self,
                phase=phase,
                root=self.factorial_root,
                contract=self.adapter_contract,
            )
        return self._p11_field[phase]

    @property
    def field_normalization(self) -> dict:
        contract = _load(
            self.adapter_contract / "P11_DENSE_RESPONSE_ADAPTER_READY.json"
        )
        return contract["normalization"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--p10-contract", type=Path, default=P10_CONTRACT)
    args = parser.parse_args()
    report = build_dense_contract(args.root, args.output, args.p10_contract)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
