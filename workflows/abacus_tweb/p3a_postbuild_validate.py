#!/usr/bin/env python3
"""Independent readback validation for completed P3a cap fields."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


DATASETS = (
    "counts",
    "exposure_binary",
    "exposure_apodized",
    "expected_counts",
    "log_count_ratio",
    "ntilde_mpc3",
    "los_x",
    "los_y",
    "los_z",
)


def sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def first_supported_block(handle: h5py.File) -> tuple[slice, slice, slice]:
    shape = handle["exposure_apodized"].shape
    chunk = handle["exposure_apodized"].chunks
    if chunk is None:
        raise RuntimeError("canonical fields must be chunked")
    for i in range(0, shape[0], chunk[0]):
        for j in range(0, shape[1], chunk[1]):
            for k in range(0, shape[2], chunk[2]):
                slices = (
                    slice(i, min(i + chunk[0], shape[0])),
                    slice(j, min(j + chunk[1], shape[1])),
                    slice(k, min(k + chunk[2], shape[2])),
                )
                if np.any(handle["exposure_apodized"][slices] > 0):
                    return slices
    raise RuntimeError("no supported HDF5 block found")


def expanded_slices(
    core: tuple[slice, slice, slice], shape: tuple[int, int, int], halo: int = 4
) -> tuple[tuple[slice, slice, slice], tuple[slice, slice, slice]]:
    outer = []
    inner = []
    for axis, part in enumerate(core):
        start = max(0, int(part.start) - halo)
        stop = min(shape[axis], int(part.stop) + halo)
        outer.append(slice(start, stop))
        inner.append(slice(int(part.start) - start, int(part.stop) - start))
    return tuple(outer), tuple(inner)


def validate_cap(name: str, metadata: dict, schema: dict) -> dict:
    path = Path(metadata["file"])
    expected_sha = metadata["file_sha256"]
    actual_sha = sha256(path)
    epsilon = float(schema["contrast"]["epsilon"])
    minimum_exposure = float(schema["contrast"]["minimum_exposure"])
    cell = float(schema["grid"]["cell_mpc"])

    with h5py.File(path, "r") as handle:
        core = first_supported_block(handle)
        outer, inner = expanded_slices(core, handle["counts"].shape)
        overlap_exact = True
        shapes_match = True
        chunks_match = True
        dtypes_match = True
        finite = True
        for key in DATASETS:
            direct = handle[key][core]
            nested = handle[key][outer][inner]
            overlap_exact &= bool(np.array_equal(direct, nested))
            shapes_match &= tuple(handle[key].shape) == tuple(metadata["grid"]["shape"])
            chunks_match &= tuple(handle[key].chunks) == tuple(schema["grid"]["chunk_shape"])
            dtypes_match &= str(handle[key].dtype) == schema["datasets"][key]["dtype"]
            if key != "exposure_binary":
                finite &= bool(np.isfinite(direct).all())

        counts = np.asarray(handle["counts"][core], dtype=np.float64)
        exposure_binary = np.asarray(handle["exposure_binary"][core], dtype=np.uint8)
        exposure = np.asarray(handle["exposure_apodized"][core], dtype=np.float64)
        expected = np.asarray(handle["expected_counts"][core], dtype=np.float64)
        contrast = np.asarray(handle["log_count_ratio"][core], dtype=np.float64)
        ntilde = np.asarray(handle["ntilde_mpc3"][core], dtype=np.float64)
        los = np.stack(
            [
                np.asarray(handle["los_x"][core], dtype=np.float64),
                np.asarray(handle["los_y"][core], dtype=np.float64),
                np.asarray(handle["los_z"][core], dtype=np.float64),
            ],
            axis=0,
        )
        expected_reconstructed = ntilde * cell ** 3 * exposure
        valid_contrast = exposure > minimum_exposure
        contrast_reconstructed = np.zeros_like(contrast)
        contrast_reconstructed[valid_contrast] = np.log(
            (counts[valid_contrast] + epsilon)
            / (expected[valid_contrast] + epsilon)
        )
        los_norm = np.sqrt(np.sum(los * los, axis=0))
        supported = exposure_binary > 0

        attrs = {
            "cap_name": str(handle.attrs["cap_name"]),
            "cell_mpc": float(handle.attrs["cell_mpc"]),
            "coordinate_units": str(handle.attrs.get("coordinate_units", "sidecar_only")),
        }

    metrics = {
        "file_bytes": path.stat().st_size,
        "file_sha256": actual_sha,
        "overlap_read_max_abs": 0.0 if overlap_exact else None,
        "expected_count_relation_max_abs": float(
            np.max(np.abs(expected - expected_reconstructed))
        ),
        "contrast_relation_max_abs": float(
            np.max(np.abs(contrast - contrast_reconstructed))
        ),
        "los_norm_max_abs_from_one": float(
            np.max(np.abs(los_norm[supported] - 1.0)) if np.any(supported) else np.inf
        ),
        "counts_min": float(np.min(counts)),
        "exposure_binary_values": sorted(np.unique(exposure_binary).astype(int).tolist()),
        "attrs": attrs,
    }
    gates = {
        "checksum_match": actual_sha == expected_sha,
        "overlap_reads_exact": overlap_exact,
        "dataset_shapes_match": shapes_match,
        "dataset_chunks_match": chunks_match,
        "dataset_dtypes_match_schema": dtypes_match,
        "sample_fields_finite": finite,
        "counts_nonnegative": metrics["counts_min"] >= 0.0,
        "binary_exposure_is_binary": set(metrics["exposure_binary_values"]) <= {0, 1},
        "expected_count_relation": metrics["expected_count_relation_max_abs"] < 1.0e-6,
        "contrast_relation": metrics["contrast_relation_max_abs"] < 1.0e-5,
        "los_unit_norm": metrics["los_norm_max_abs_from_one"] < 1.0e-6,
        "cap_attribute": attrs["cap_name"] == name,
        "coordinate_unit_bound_by_schema": schema["coordinate_frame"]["units"] == "Mpc",
        "cell_attribute": np.isclose(attrs["cell_mpc"], cell),
    }
    return {"file": str(path), "metrics": metrics, "gates": gates, "pass": all(gates.values())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest", type=Path,
        default=Path("/pscratch/sd/d/dkololgi/abacus/p3_full_footprint/field_manifest.json"),
    )
    ap.add_argument(
        "--schema", type=Path,
        default=Path("/pscratch/sd/d/dkololgi/abacus/p3_full_footprint/p3_field_schema_v1.json"),
    )
    ap.add_argument(
        "--unit-audit", type=Path,
        default=Path("/pscratch/sd/d/dkololgi/abacus/p3_full_footprint/unit_audit.json"),
    )
    ap.add_argument(
        "--complete", type=Path,
        default=Path("/pscratch/sd/d/dkololgi/abacus/p3_full_footprint/FIELD_COMPLETE"),
    )
    ap.add_argument(
        "--out", type=Path,
        default=Path("/pscratch/sd/d/dkololgi/abacus/p3_full_footprint/postbuild_validation.json"),
    )
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    schema = json.loads(args.schema.read_text())
    unit_audit = json.loads(args.unit_audit.read_text())
    components = {
        name: validate_cap(name, metadata, schema)
        for name, metadata in manifest["components"].items()
    }
    marker = args.complete.read_text().strip()
    global_gates = {
        "unit_audit_pass": bool(unit_audit["pass"]),
        "frozen_schema_checksum_match": manifest["frozen_schema_sha256"] == sha256(args.schema),
        "manifest_gates_pass": all(manifest["gates"].values()),
        "both_caps_validated": set(components) == {"NGC", "SGC"},
        "component_readback_pass": all(v["pass"] for v in components.values()),
        "marker_binds_manifest": f"manifest_sha256={sha256(args.manifest)}" in marker,
    }
    payload = {
        "schema_version": 1,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "unit_audit": str(args.unit_audit),
        "unit_audit_sha256": sha256(args.unit_audit),
        "field_complete": str(args.complete),
        "components": components,
        "gates": global_gates,
        "pass": all(global_gates.values()),
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=bool) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True, default=bool))
    if not payload["pass"]:
        raise RuntimeError("P3 post-build validation failed")


if __name__ == "__main__":
    main()
