#!/usr/bin/env python3
"""Validate the P6 full-cap selection overlay on production patch views."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from p6_field_patch_utils import (
    CAP_NAME,
    CanonicalFieldPatchAdapter,
    SELECTION_CHANNELS,
    apply_frozen_normalization,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter-root",
        type=Path,
        default=Path("/pscratch/sd/d/dkololgi/abacus/p6_unet_patch_adapter"),
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=Path(
            "/pscratch/sd/d/dkololgi/abacus/p6_unet_patch_adapter/"
            "fullcap_selection_v1/selection_manifest.json"
        ),
    )
    parser.add_argument("--small-halo", type=int, default=4)
    parser.add_argument("--large-halo", type=int, default=8)
    return parser.parse_args()


def overlap_slices(first, second):
    start = np.maximum(first.context_start, second.context_start)
    stop = np.minimum(first.context_stop, second.context_stop)
    if np.any(stop <= start):
        raise RuntimeError("patch views do not overlap")
    first_slice = tuple(
        slice(int(left), int(right))
        for left, right in zip(start - first.context_start, stop - first.context_start)
    )
    second_slice = tuple(
        slice(int(left), int(right))
        for left, right in zip(start - second.context_start, stop - second.context_start)
    )
    return first_slice, second_slice


def representative_cores(adapter: CanonicalFieldPatchAdapter) -> list[int]:
    authoritative = np.diff(np.asarray(adapter.core_offsets, dtype=np.int64))
    representatives = []
    for cap in (0, 1):
        for fold in range(5):
            candidate = np.flatnonzero(
                (np.asarray(adapter.core_cap) == cap)
                & (np.asarray(adapter.core_fold) == fold)
            )
            if len(candidate) == 0:
                raise RuntimeError(f"no core for cap={cap} fold={fold}")
            representatives.append(int(candidate[np.argmax(authoritative[candidate])]))
    return representatives


def main() -> None:
    args = parse_args()
    started = time.time()
    selection = json.loads(args.selection_manifest.read_text())
    if not selection.get("pass"):
        raise RuntimeError("selection refit manifest has not passed")
    rows = []
    maximum_overlap_difference = 0.0
    maximum_expected_identity_error = 0.0
    maximum_contrast_identity_error = 0.0
    maximum_legacy_difference = 0.0
    normalized_finite = True

    with CanonicalFieldPatchAdapter(args.adapter_root) as legacy:
        cores = representative_cores(legacy)
        for rotation in range(5):
            normalizer = selection["rotations"][str(rotation)]["normalization"]
            with CanonicalFieldPatchAdapter(
                args.adapter_root,
                selection_manifest=args.selection_manifest,
                rotation=rotation,
            ) as adapter:
                for core_id in cores:
                    small = adapter.extract(core_id, args.small_halo)
                    large = adapter.extract(core_id, args.large_halo)
                    first_slice, second_slice = overlap_slices(small, large)
                    channel_differences = {}
                    for name in SELECTION_CHANNELS:
                        index = small.channel_names.index(name)
                        difference = float(np.max(np.abs(
                            small.values[(index,) + first_slice]
                            - large.values[(index,) + second_slice]
                        )))
                        channel_differences[name] = difference
                        maximum_overlap_difference = max(
                            maximum_overlap_difference, difference
                        )

                    by_name = {
                        name: small.values[index]
                        for index, name in enumerate(small.channel_names)
                    }
                    cell_mpc = float(
                        adapter.manifest["caps"][CAP_NAME[small.cap]]["cell_mpc"]
                    )
                    minimum_exposure = float(
                        selection["contrast"]["minimum_exposure"]
                    )
                    epsilon = float(selection["contrast"]["epsilon"])
                    supported = by_name["exposure_apodized"] > minimum_exposure
                    reconstructed_expected = (
                        by_name["ntilde_mpc3"]
                        * cell_mpc**3
                        * by_name["exposure_apodized"]
                    )
                    expected_error = float(np.max(np.abs(
                        reconstructed_expected - by_name["expected_counts"]
                    )))
                    maximum_expected_identity_error = max(
                        maximum_expected_identity_error, expected_error
                    )
                    reconstructed_contrast = np.zeros_like(
                        by_name["log_count_ratio"]
                    )
                    reconstructed_contrast[supported] = np.log(
                        (by_name["counts"][supported] + epsilon)
                        / (by_name["expected_counts"][supported] + epsilon)
                    )
                    contrast_error = float(np.max(np.abs(
                        reconstructed_contrast - by_name["log_count_ratio"]
                    )))
                    maximum_contrast_identity_error = max(
                        maximum_contrast_identity_error, contrast_error
                    )

                    legacy_patch = legacy.extract(core_id, args.small_halo)
                    legacy_by_name = {
                        name: legacy_patch.values[index]
                        for index, name in enumerate(legacy_patch.channel_names)
                    }
                    legacy_difference = float(np.mean(np.abs(
                        by_name["expected_counts"] - legacy_by_name["expected_counts"]
                    )))
                    maximum_legacy_difference = max(
                        maximum_legacy_difference, legacy_difference
                    )
                    normalized = apply_frozen_normalization(small, normalizer)
                    finite = bool(np.all(np.isfinite(normalized)))
                    normalized_finite &= finite
                    rows.append({
                        "rotation": rotation,
                        "core_id": core_id,
                        "cap": CAP_NAME[small.cap],
                        "fold": small.fold,
                        "authoritative_galaxies": int(
                            len(small.authoritative_parent_id)
                        ),
                        "overlap_max_abs": channel_differences,
                        "expected_identity_max_abs": expected_error,
                        "contrast_identity_max_abs": contrast_error,
                        "legacy_expected_mean_abs_difference": legacy_difference,
                        "normalized_finite": finite,
                    })

    gates = {
        "all_rotations_caps_folds_represented": len(rows) == 50,
        "halo_invariant_to_float32_tolerance": maximum_overlap_difference <= 1.0e-6,
        "expected_count_identity": maximum_expected_identity_error <= 1.0e-6,
        "contrast_identity": maximum_contrast_identity_error <= 2.0e-6,
        "overlay_changes_legacy_selection_channels": maximum_legacy_difference > 0,
        "frozen_normalization_finite": normalized_finite,
        "selection_manifest_passed": bool(selection["pass"]),
    }
    report_path = args.selection_manifest.parent / "selection_overlay_validation.json"
    report = {
        "schema_version": "p6-selection-overlay-validation-v1",
        "stage": "P6_SELECTION_OVERLAY_VALIDATION",
        "status": "complete" if all(gates.values()) else "failed_gate",
        "pass": bool(all(gates.values())),
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "elapsed_seconds": time.time() - started,
        "inputs": {
            "adapter_root": str(args.adapter_root),
            "adapter_manifest_sha256": sha256(
                args.adapter_root / "adapter_manifest.json"
            ),
            "selection_manifest": str(args.selection_manifest),
            "selection_manifest_sha256": sha256(args.selection_manifest),
        },
        "halos_voxels": {
            "small": args.small_halo,
            "large": args.large_halo,
        },
        "maximum_overlap_difference": maximum_overlap_difference,
        "maximum_expected_identity_error": maximum_expected_identity_error,
        "maximum_contrast_identity_error": maximum_contrast_identity_error,
        "maximum_legacy_expected_mean_abs_difference": maximum_legacy_difference,
        "rows": rows,
        "gates": gates,
    }
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(report_path)
    marker = report_path.parent / "SELECTION_CHANNELS_READY"
    if report["pass"]:
        marker.write_text(json.dumps({
            "selection_manifest": str(args.selection_manifest),
            "selection_manifest_sha256": sha256(args.selection_manifest),
            "validation_report": str(report_path),
            "validation_report_sha256": sha256(report_path),
            "git_sha": report["git_sha"],
        }, sort_keys=True) + "\n")
    elif marker.exists():
        marker.unlink()
    print(json.dumps({
        "report": str(report_path),
        "pass": report["pass"],
        "gates": gates,
        "maximum_overlap_difference": maximum_overlap_difference,
        "maximum_expected_identity_error": maximum_expected_identity_error,
        "maximum_contrast_identity_error": maximum_contrast_identity_error,
    }, indent=2))


if __name__ == "__main__":
    main()
