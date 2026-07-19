#!/usr/bin/env python3
"""Run the first real-catalogue P7 graph/field/scatter/FFT composition gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import time

import numpy as np

from p5_graph_patch_utils import CanonicalGraphPatchAdapter
from p6_field_patch_utils import CAP_NAME, CanonicalFieldPatchAdapter, fractional_cell_index
from p7_ftier_patch_utils import (
    fft_tidal_components,
    scatter_nodes,
    tensor_and_eigensystem,
    trace_max_abs_error,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p5-root", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p5_graph_patch_adapter"))
    ap.add_argument("--p6-root", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p6_unet_patch_adapter"))
    ap.add_argument("--output-root", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p7_ftier_patch_adapter"))
    ap.add_argument("--selection-manifest", type=Path, default=Path(
        "/pscratch/sd/d/dkololgi/abacus/p6_unet_patch_adapter/"
        "fullcap_selection_v1/selection_manifest.json"))
    ap.add_argument("--rotation", type=int, default=0)
    ap.add_argument("--core-id", type=int, action="append")
    ap.add_argument("--halo-small", type=int, default=8)
    ap.add_argument("--halo-large", type=int, default=12)
    ap.add_argument("--rsmooth-mpc", type=float, default=7.0 / 0.6766,
                    help="Frozen P3 observer-frame equivalent of 7 Mpc/h "
                         "(Planck18 h=0.6766).")
    return ap.parse_args()


def complete_tsc(frac_local: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    nearest = np.rint(frac_local).astype(np.int64)
    shape = np.asarray(shape, dtype=np.int64)
    return np.all((nearest >= 1) & (nearest <= shape - 2), axis=1)


def overlap_slices(small, large):
    start = np.maximum(small.context_start, large.context_start)
    stop = np.minimum(small.context_stop, large.context_stop)
    small_slice = tuple(
        slice(int(a - b), int(c - b))
        for a, c, b in zip(start, stop, small.context_start)
    )
    large_slice = tuple(
        slice(int(a - b), int(c - b))
        for a, c, b in zip(start, stop, large.context_start)
    )
    return small_slice, large_slice


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    cores = args.core_id or [454, 17152]
    points = np.load(
        "/pscratch/sd/d/dkololgi/abacus/graph_constructions/"
        "path1_fiberassign_mock_bgs_maglim_rs7_points.npy",
        mmap_mode="r",
    )
    records = []
    selection_ready = args.selection_manifest.parent / "SELECTION_CHANNELS_READY"
    if not selection_ready.exists():
        raise FileNotFoundError(
            f"selection overlay has not passed P6 validation: {selection_ready}"
        )
    graph_adapter = CanonicalGraphPatchAdapter(args.p5_root)
    with CanonicalFieldPatchAdapter(
        args.p6_root,
        selection_manifest=args.selection_manifest,
        rotation=args.rotation,
    ) as field_adapter:
        for core_id in cores:
            graph = graph_adapter.extract(core_id, 2, loss_policy="authoritative")
            small = field_adapter.extract(core_id, args.halo_small)
            large = field_adapter.extract(core_id, args.halo_large)
            cap_name = CAP_NAME[small.cap]
            cap_spec = field_adapter.manifest["caps"][cap_name]
            xyz = np.asarray(points[graph.parent_node_id, :3], dtype=np.float64)
            frac_global = fractional_cell_index(
                xyz, cap_spec["origin_mpc"], cap_spec["cell_mpc"]
            )
            frac_small = frac_global - small.context_start[None, :]
            frac_large = frac_global - large.context_start[None, :]
            retained = (
                complete_tsc(frac_small, small.values.shape[1:])
                & complete_tsc(frac_large, large.values.shape[1:])
            )
            if not np.any(retained):
                raise RuntimeError(f"core {core_id} has no complete scatter nodes")
            latent = np.column_stack([
                np.ones(np.sum(retained), dtype=np.float32),
                np.asarray(graph.node_features[retained, 0], dtype=np.float32),
            ])
            grid_small, diag_small = scatter_nodes(
                latent, frac_small[retained], small.values.shape[1:], scheme="tsc"
            )
            grid_large, diag_large = scatter_nodes(
                latent, frac_large[retained], large.values.shape[1:], scheme="tsc"
            )
            slice_small, slice_large = overlap_slices(small, large)
            overlap_diff = float(np.max(np.abs(
                grid_small[(slice(None),) + slice_small]
                - grid_large[(slice(None),) + slice_large]
            )))
            input_sum = np.asarray(diag_small["input_sum_by_channel"])
            scatter_error = float(np.max(np.abs(
                np.asarray(diag_small["grid_sum_by_channel"]) - input_sum
            )))
            authoritative_match = np.array_equal(
                np.sort(graph.parent_node_id[graph.authoritative_core_mask]),
                np.sort(small.authoritative_parent_id),
            )
            delta_index = large.channel_names.index("log_count_ratio")
            delta = np.asarray(large.values[delta_index], dtype=np.float64)
            physics = []
            for taper in (0, 4, 8):
                components, smoothed = fft_tidal_components(
                    delta,
                    cell_mpc=float(cap_spec["cell_mpc"]),
                    rsmooth_mpc=args.rsmooth_mpc,
                    apodization_width_voxels=taper,
                )
                _, eigenvalues, eigenvectors = tensor_and_eigensystem(components)
                gaps = np.diff(eigenvalues, axis=-1)
                physics.append({
                    "apodization_width_voxels": taper,
                    "trace_max_abs_error": trace_max_abs_error(components, smoothed),
                    "ordered_eigenvalues": bool(np.all(gaps >= -1e-12)),
                    "finite_eigenvectors": bool(np.isfinite(eigenvectors).all()),
                    "lambda1_mean": float(eigenvalues[..., 0].mean()),
                    "lambda1_std": float(eigenvalues[..., 0].std()),
                })
            records.append({
                "core_id": core_id,
                "cap": cap_name,
                "graph_nodes": graph.n_node,
                "graph_directed_edges": graph.n_edge,
                "field_shape_small": list(small.values.shape[1:]),
                "field_shape_large": list(large.values.shape[1:]),
                "authoritative_identity": bool(authoritative_match),
                "scatter_nodes": int(np.sum(retained)),
                "scatter_conservation_max_abs": scatter_error,
                "scatter_overlap_max_abs": overlap_diff,
                "physics_candidates": physics,
            })

    gates = {
        "p5_p6_authoritative_identity": all(
            row["authoritative_identity"] for row in records),
        "tsc_scatter_conservation": all(
            row["scatter_conservation_max_abs"] < 2e-3 for row in records),
        "tsc_overlap_parity": all(
            row["scatter_overlap_max_abs"] < 2e-6 for row in records),
        "fixed_fft_trace_consistency": all(
            candidate["trace_max_abs_error"] < 1e-10
            for row in records for candidate in row["physics_candidates"]),
        "finite_ordered_eigensystems": all(
            candidate["ordered_eigenvalues"] and candidate["finite_eigenvectors"]
            for row in records for candidate in row["physics_candidates"]),
        "fullcap_selection_overlay_ready": selection_ready.exists(),
    }
    p5_manifest = args.p5_root / "adapter_manifest.json"
    p6_manifest = args.p6_root / "adapter_manifest.json"
    report = {
        "schema_version": 1,
        "stage": "P7 initial graph-field-FFT composition",
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "p5_manifest": str(p5_manifest),
        "p5_manifest_sha256": sha256(p5_manifest),
        "p6_manifest": str(p6_manifest),
        "p6_manifest_sha256": sha256(p6_manifest),
        "selection_manifest": str(args.selection_manifest),
        "selection_manifest_sha256": sha256(args.selection_manifest),
        "selection_rotation": args.rotation,
        "selection_channels_ready": str(selection_ready),
        "selection_channels_ready_sha256": sha256(selection_ready),
        "cell_units": "observer-frame comoving Mpc",
        "rsmooth_mpc": args.rsmooth_mpc,
        "records": records,
        "gates": gates,
        "pass": all(gates.values()),
        "ftier_patch_ready": False,
        "pending": [
            "trained graph encoder plus field-decoder context convergence",
            "FFT tile-size, padding, apodization, overlap, and central-trim convergence",
            "eigenvector convergence conditioned on eigengap and survey-boundary distance",
        ],
        "elapsed_seconds": time.time() - started,
    }
    report_path = args.output_root / "initial_composition_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["pass"]:
        raise RuntimeError(f"P7 initial gates failed: {gates}")
    (args.output_root / "FTIER_COMPOSITION_READY").write_text(
        f"initial_composition_report_sha256={sha256(report_path)}\n"
        "ftier_patch_ready=false\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
