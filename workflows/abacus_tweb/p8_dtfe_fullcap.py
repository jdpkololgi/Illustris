#!/usr/bin/env python3
"""Exact piecewise-linear full-cap DTFE construction and P8 evaluation.

This uses the immutable full NGC+SGC Delaunay tetrahedra.  Vertex densities are
the standard 3-D DTFE values ``4 / sum(incident tetrahedron volumes)`` and grid
values are barycentric interpolants inside the containing tetrahedron.  The
result is therefore not a vertex splat or a smoothed-DTFE approximation.

The workflow is deliberately staged: ``--mode preflight`` measures the exact
rasterisation workload; ``build`` produces the two cap fields; ``evaluate``
uses the frozen per-rotation selection curve, fixed tidal solve, training-only
affine calibration, and P4 authoritative validation rows.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path
import sys

import h5py
import numpy as np
import torch
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p8_classical_fullcap import (
    CAP_NAME,
    _sample_tidal_eigenvalues,
)
from workflows.abacus_tweb.p8_deterministic_common import (
    atomic_json,
    authoritative_mask,
    evaluate_complete_fold,
    fit_affine_on_training,
    fold_roles,
    sha256,
)


P8_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p8_deterministic_v1")
GRAPH_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/graph_constructions")
PREFIX = GRAPH_ROOT / "path1_fiberassign_mock_bgs_maglim_rs7"
POINTS = Path(str(PREFIX) + "_points.npy")
TETS = Path(str(PREFIX) + "_tetrahedra_idx.npy")
VOLUMES = Path(str(PREFIX) + "_tetrahedra_volumes.npy")
FIELD_ADAPTER = Path("/pscratch/sd/d/dkololgi/abacus/p6_unet_patch_adapter")
SELECTION = FIELD_ADAPTER / "fullcap_selection_v1/selection_manifest.json"
ASSIGNMENT = Path("/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/active_assignment.npz")
ROTATIONS = Path("/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/rotations.json")
RSMOOTH_MPC = 7.0 / 0.6766


def barycentric_interpolate(vertices: np.ndarray, values: np.ndarray, point: np.ndarray) -> float:
    """Reference CPU interpolation used by the synthetic parity test."""
    vertices = np.asarray(vertices, dtype=np.float64)
    weights123 = np.linalg.solve((vertices[1:] - vertices[0]).T, point - vertices[0])
    weights = np.r_[1.0 - weights123.sum(), weights123]
    return float(weights @ np.asarray(values, dtype=np.float64))


def compute_vertex_density(
    tets: np.ndarray,
    volumes: np.ndarray,
    n_points: int,
    *,
    device: str,
    chunk: int,
) -> np.ndarray:
    """GPU-accumulate contiguous Voronoi-star volumes and return 4/Vstar."""
    star = torch.zeros(n_points, dtype=torch.float64, device=device)
    for left in range(0, len(tets), chunk):
        right = min(left + chunk, len(tets))
        tet = torch.from_numpy(np.asarray(tets[left:right], dtype=np.int64)).to(device)
        vol = torch.from_numpy(np.asarray(volumes[left:right], dtype=np.float64)).to(device)
        for column in range(4):
            star.index_add_(0, tet[:, column], vol)
        del tet, vol
    density = torch.zeros_like(star, dtype=torch.float32)
    valid = star > 0
    density[valid] = (4.0 / star[valid]).float()
    result = density.cpu().numpy()
    del star, density, valid
    torch.cuda.empty_cache()
    return result


def build_incident_csr(
    tets: np.ndarray,
    n_points: int,
    *,
    output_root: Path,
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build vertex -> incident-tetrahedron CSR once using a GPU integer sort.

    The former tetrahedron-centric raster visited every voxel in every tetrahedron
    AABB.  Full-cap boundary tetrahedra make that workload pathological.  The
    accelerated locator instead starts from nearby Delaunay vertices and tests
    only tetrahedra incident on those vertices.  This CSR is the immutable
    acceleration structure for that exact search.
    """
    offsets_path = output_root / "incident_offsets.npy"
    tetra_path = output_root / "incident_tetrahedron_id.npy"
    manifest_path = output_root / "incident_csr_manifest.json"
    if offsets_path.exists() and tetra_path.exists() and manifest_path.exists():
        offsets = np.load(offsets_path, mmap_mode="r")
        incident = np.load(tetra_path, mmap_mode="r")
        manifest = json.loads(manifest_path.read_text())
        if (
            tuple(manifest["tetrahedra_shape"]) != tuple(tets.shape)
            or int(manifest["n_points"]) != int(n_points)
            or int(offsets[-1]) != int(tets.size)
            or len(incident) != int(tets.size)
        ):
            raise RuntimeError("stored incident CSR does not match the current tetrahedra")
        return offsets, incident, manifest

    started = time.time()
    flat = np.asarray(tets, dtype=np.int32).reshape(-1)
    counts = np.bincount(flat, minlength=n_points)
    offsets = np.empty(n_points + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, dtype=np.int64, out=offsets[1:])
    del counts

    flat_gpu = torch.from_numpy(flat).to(device)
    order = torch.argsort(flat_gpu, stable=False)
    incident = (order // 4).to(torch.int32).cpu().numpy()
    del flat_gpu, order
    torch.cuda.empty_cache()
    if len(incident) != int(tets.size) or int(offsets[-1]) != len(incident):
        raise RuntimeError("incident CSR construction failed its cardinality check")
    np.save(offsets_path, offsets)
    np.save(tetra_path, incident)
    manifest = {
        "schema_version": 1,
        "method": "GPU sort of flattened immutable tetrahedron vertices",
        "n_points": int(n_points),
        "tetrahedra_shape": list(tets.shape),
        "incidences": int(len(incident)),
        "mean_tetrahedra_per_vertex": float(len(incident) / n_points),
        "maximum_tetrahedra_per_vertex": int(np.diff(offsets).max(initial=0)),
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(manifest_path, manifest)
    return (
        np.load(offsets_path, mmap_mode="r"),
        np.load(tetra_path, mmap_mode="r"),
        manifest,
    )


def locate_points_incident_cpu(
    points: np.ndarray,
    tets: np.ndarray,
    vertex_density: np.ndarray,
    queries: np.ndarray,
    nearest_vertex: np.ndarray,
    offsets: np.ndarray,
    incident_tetrahedron: np.ndarray,
    *,
    epsilon: float = 1.0e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Small exact CPU reference for unit/parity tests.

    ``nearest_vertex`` may have shape ``(N,)`` or ``(N,K)``.  The production
    kernel uses K=1 and retries only unresolved points with a wider K-neighbour
    search.  The interpolation itself is exact piecewise-linear DTFE.
    """
    nearest = np.asarray(nearest_vertex, dtype=np.int64)
    if nearest.ndim == 1:
        nearest = nearest[:, None]
    values = np.full(len(queries), np.nan, dtype=np.float64)
    containing = np.full(len(queries), -1, dtype=np.int64)
    xyz = np.asarray(points, dtype=np.float64)
    for row, query in enumerate(np.asarray(queries, dtype=np.float64)):
        seen: set[int] = set()
        for vertex in nearest[row]:
            for at in range(int(offsets[vertex]), int(offsets[vertex + 1])):
                tet_id = int(incident_tetrahedron[at])
                if tet_id in seen:
                    continue
                seen.add(tet_id)
                ids = np.asarray(tets[tet_id], dtype=np.int64)
                vertices = xyz[ids]
                try:
                    weights123 = np.linalg.solve(
                        (vertices[1:] - vertices[0]).T, query - vertices[0]
                    )
                except np.linalg.LinAlgError:
                    continue
                weights = np.r_[1.0 - weights123.sum(), weights123]
                if np.min(weights) >= -epsilon and np.max(weights) <= 1.0 + epsilon:
                    values[row] = float(weights @ np.asarray(vertex_density[ids]))
                    containing[row] = tet_id
                    break
            if containing[row] >= 0:
                break
    return values, containing


def preflight_aabb_visits(
    points: np.ndarray,
    tets: np.ndarray,
    *,
    cap: int,
    origin: np.ndarray,
    shape: tuple[int, int, int],
    cell_mpc: float,
    chunk: int,
) -> dict:
    """Upper-bound raster workload before committing GPU time."""
    visits = []
    cap_tets = 0
    intersecting = 0
    total = 0
    maximum = 0
    for left in range(0, len(tets), chunk):
        right = min(left + chunk, len(tets))
        tet = np.asarray(tets[left:right], dtype=np.int64)
        keep = np.asarray(points[tet[:, 0], 3], dtype=np.int8) == cap
        if not keep.any():
            continue
        vertices = np.asarray(points[tet[keep], :3], dtype=np.float64)
        lower = np.ceil((vertices.min(axis=1) - origin[None, :]) / cell_mpc - 0.5).astype(np.int64)
        upper = np.floor((vertices.max(axis=1) - origin[None, :]) / cell_mpc - 0.5).astype(np.int64)
        lower = np.maximum(lower, 0)
        upper = np.minimum(upper, np.asarray(shape) - 1)
        spans = np.maximum(upper - lower + 1, 0)
        count = np.prod(spans, axis=1, dtype=np.int64)
        positive = count > 0
        cap_tets += int(keep.sum())
        intersecting += int(positive.sum())
        total += int(count.sum(dtype=np.int64))
        maximum = max(maximum, int(count.max(initial=0)))
        if positive.any():
            # A bounded deterministic reservoir is sufficient for workload quantiles.
            stride = max(1, int(math.ceil(positive.sum() / 200_000)))
            visits.append(count[positive][::stride])
    sample = np.concatenate(visits) if visits else np.zeros(0, dtype=np.int64)
    if len(sample) > 1_000_000:
        sample = sample[::int(math.ceil(len(sample) / 1_000_000))]
    return {
        "cap_tetrahedra": cap_tets,
        "grid_intersecting_tetrahedra": intersecting,
        "aabb_voxel_visits_upper_bound": total,
        "maximum_aabb_voxels": maximum,
        "sampled_quantiles": {
            str(q): float(np.quantile(sample, q)) if len(sample) else None
            for q in (0.5, 0.9, 0.99, 0.999, 0.9999)
        },
    }


def _raster_kernel():
    """Compile lazily so preflight and --help do not require a CUDA Python stack."""
    from numba import cuda

    @cuda.jit
    def kernel(points, cap_flag, tets, vertex_density, exposure, origin, cell, cap, output):
        tid = cuda.grid(1)
        if tid >= tets.shape[0]:
            return
        i0, i1, i2, i3 = tets[tid, 0], tets[tid, 1], tets[tid, 2], tets[tid, 3]
        if cap_flag[i0] != cap:
            return
        x0, y0, z0 = points[i0, 0], points[i0, 1], points[i0, 2]
        x1, y1, z1 = points[i1, 0], points[i1, 1], points[i1, 2]
        x2, y2, z2 = points[i2, 0], points[i2, 1], points[i2, 2]
        x3, y3, z3 = points[i3, 0], points[i3, 1], points[i3, 2]
        xmin, xmax = min(x0, x1, x2, x3), max(x0, x1, x2, x3)
        ymin, ymax = min(y0, y1, y2, y3), max(y0, y1, y2, y3)
        zmin, zmax = min(z0, z1, z2, z3), max(z0, z1, z2, z3)
        ix0 = max(0, int(math.ceil((xmin - origin[0]) / cell - 0.5)))
        iy0 = max(0, int(math.ceil((ymin - origin[1]) / cell - 0.5)))
        iz0 = max(0, int(math.ceil((zmin - origin[2]) / cell - 0.5)))
        ix1 = min(output.shape[0] - 1, int(math.floor((xmax - origin[0]) / cell - 0.5)))
        iy1 = min(output.shape[1] - 1, int(math.floor((ymax - origin[1]) / cell - 0.5)))
        iz1 = min(output.shape[2] - 1, int(math.floor((zmax - origin[2]) / cell - 0.5)))
        if ix1 < ix0 or iy1 < iy0 or iz1 < iz0:
            return
        a00, a01, a02 = x1 - x0, x2 - x0, x3 - x0
        a10, a11, a12 = y1 - y0, y2 - y0, y3 - y0
        a20, a21, a22 = z1 - z0, z2 - z0, z3 - z0
        c00, c01, c02 = a11 * a22 - a12 * a21, a02 * a21 - a01 * a22, a01 * a12 - a02 * a11
        c10, c11, c12 = a12 * a20 - a10 * a22, a00 * a22 - a02 * a20, a02 * a10 - a00 * a12
        c20, c21, c22 = a10 * a21 - a11 * a20, a01 * a20 - a00 * a21, a00 * a11 - a01 * a10
        det = a00 * c00 + a01 * c10 + a02 * c20
        if abs(det) < 1.0e-20:
            return
        invdet = 1.0 / det
        r0, r1, r2, r3 = vertex_density[i0], vertex_density[i1], vertex_density[i2], vertex_density[i3]
        for ix in range(ix0, ix1 + 1):
            px = origin[0] + (ix + 0.5) * cell - x0
            for iy in range(iy0, iy1 + 1):
                py = origin[1] + (iy + 0.5) * cell - y0
                for iz in range(iz0, iz1 + 1):
                    if exposure[ix, iy, iz] == 0:
                        continue
                    pz = origin[2] + (iz + 0.5) * cell - z0
                    w1 = (c00 * px + c01 * py + c02 * pz) * invdet
                    w2 = (c10 * px + c11 * py + c12 * pz) * invdet
                    w3 = (c20 * px + c21 * py + c22 * pz) * invdet
                    w0 = 1.0 - w1 - w2 - w3
                    eps = 2.0e-5
                    if w0 >= -eps and w1 >= -eps and w2 >= -eps and w3 >= -eps:
                        output[ix, iy, iz] = w0 * r0 + w1 * r1 + w2 * r2 + w3 * r3
    return kernel


def rasterize_cap(
    *,
    points: np.ndarray,
    tets: np.ndarray,
    vertex_density: np.ndarray,
    cap: int,
    field_path: Path,
    origin: np.ndarray,
    shape: tuple[int, int, int],
    cell_mpc: float,
    output: Path,
    threads: int,
) -> dict:
    from numba import cuda

    with h5py.File(field_path, "r") as handle:
        exposure = (np.asarray(handle["exposure_apodized"], dtype=np.float32) > 0.0).astype(np.uint8)
    d_points = cuda.to_device(np.ascontiguousarray(points[:, :3], dtype=np.float32))
    d_cap = cuda.to_device(np.ascontiguousarray(points[:, 3], dtype=np.uint8))
    d_tets = cuda.to_device(np.ascontiguousarray(tets, dtype=np.int32))
    d_density = cuda.to_device(np.ascontiguousarray(vertex_density, dtype=np.float32))
    d_exposure = cuda.to_device(exposure)
    d_origin = cuda.to_device(np.asarray(origin, dtype=np.float32))
    d_output = cuda.device_array(shape, dtype=np.float32)
    d_output[:] = np.nan
    blocks = (len(tets) + threads - 1) // threads
    kernel = _raster_kernel()
    started = time.time()
    kernel[blocks, threads](
        d_points, d_cap, d_tets, d_density, d_exposure, d_origin,
        np.float32(cell_mpc), np.uint8(cap), d_output,
    )
    cuda.synchronize()
    host = d_output.copy_to_host()
    supported = exposure > 0
    finite = np.isfinite(host)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, host)
    report = {
        "cap": int(cap), "cap_name": CAP_NAME[cap], "shape": list(shape),
        "field_path": str(field_path), "output": str(output),
        "supported_voxels": int(supported.sum()),
        "finite_supported_voxels": int((finite & supported).sum()),
        "supported_coverage": float((finite & supported).sum() / max(supported.sum(), 1)),
        "finite_density_min": float(np.nanmin(host)),
        "finite_density_max": float(np.nanmax(host)),
        "elapsed_seconds": time.time() - started,
    }
    del d_points, d_cap, d_tets, d_density, d_exposure, d_origin, d_output, host, exposure
    return report


def _incident_locator_kernel():
    """Compile the voxel-centric exact point locator lazily."""
    from numba import cuda

    @cuda.jit
    def kernel(
        points, tets, vertex_density, offsets, incident, queries,
        nearest_vertex, epsilon, output, containing,
    ):
        row = cuda.grid(1)
        if row >= queries.shape[0]:
            return
        qx, qy, qz = queries[row, 0], queries[row, 1], queries[row, 2]
        for seed_column in range(nearest_vertex.shape[1]):
            seed = nearest_vertex[row, seed_column]
            for at in range(offsets[seed], offsets[seed + 1]):
                tet_id = incident[at]
                i0 = tets[tet_id, 0]
                i1 = tets[tet_id, 1]
                i2 = tets[tet_id, 2]
                i3 = tets[tet_id, 3]
                x0, y0, z0 = points[i0, 0], points[i0, 1], points[i0, 2]
                ax, ay, az = points[i1, 0] - x0, points[i1, 1] - y0, points[i1, 2] - z0
                bx, by, bz = points[i2, 0] - x0, points[i2, 1] - y0, points[i2, 2] - z0
                cx, cy, cz = points[i3, 0] - x0, points[i3, 1] - y0, points[i3, 2] - z0
                px, py, pz = qx - x0, qy - y0, qz - z0

                bcx, bcy, bcz = by * cz - bz * cy, bz * cx - bx * cz, bx * cy - by * cx
                determinant = ax * bcx + ay * bcy + az * bcz
                if abs(determinant) < 1.0e-24:
                    continue
                inverse = 1.0 / determinant
                w1 = (px * bcx + py * bcy + pz * bcz) * inverse
                pcx, pcy, pcz = py * cz - pz * cy, pz * cx - px * cz, px * cy - py * cx
                w2 = (ax * pcx + ay * pcy + az * pcz) * inverse
                bpx, bpy, bpz = by * pz - bz * py, bz * px - bx * pz, bx * py - by * px
                w3 = (ax * bpx + ay * bpy + az * bpz) * inverse
                w0 = 1.0 - w1 - w2 - w3
                if (
                    w0 >= -epsilon and w1 >= -epsilon
                    and w2 >= -epsilon and w3 >= -epsilon
                    and w0 <= 1.0 + epsilon and w1 <= 1.0 + epsilon
                    and w2 <= 1.0 + epsilon and w3 <= 1.0 + epsilon
                ):
                    output[row] = (
                        w0 * vertex_density[i0] + w1 * vertex_density[i1]
                        + w2 * vertex_density[i2] + w3 * vertex_density[i3]
                    )
                    containing[row] = tet_id
                    return
    return kernel


def _locate_chunk_gpu(
    *,
    kernel,
    d_points,
    d_tets,
    d_density,
    d_offsets,
    d_incident,
    queries: np.ndarray,
    nearest_vertex: np.ndarray,
    epsilon: float,
    threads: int,
) -> tuple[np.ndarray, np.ndarray]:
    from numba import cuda

    nearest = np.asarray(nearest_vertex, dtype=np.int32)
    if nearest.ndim == 1:
        nearest = nearest[:, None]
    d_queries = cuda.to_device(np.ascontiguousarray(queries, dtype=np.float64))
    d_nearest = cuda.to_device(np.ascontiguousarray(nearest))
    d_values = cuda.to_device(np.full(len(queries), np.nan, dtype=np.float32))
    d_containing = cuda.to_device(np.full(len(queries), -1, dtype=np.int32))
    blocks = (len(queries) + threads - 1) // threads
    kernel[blocks, threads](
        d_points, d_tets, d_density, d_offsets, d_incident,
        d_queries, d_nearest, epsilon, d_values, d_containing,
    )
    cuda.synchronize()
    values = d_values.copy_to_host()
    containing = d_containing.copy_to_host()
    del d_queries, d_nearest, d_values, d_containing
    return values, containing


# This second definition intentionally replaces the legacy AABB implementation
# above while retaining it as an auditable record of the failed preflight.
def rasterize_cap(
    *,
    points: np.ndarray,
    tets: np.ndarray,
    vertex_density: np.ndarray,
    incident_offsets: np.ndarray,
    incident_tetrahedron: np.ndarray,
    cap: int,
    field_path: Path,
    origin: np.ndarray,
    shape: tuple[int, int, int],
    cell_mpc: float,
    output: Path,
    threads: int,
    raster_slab: int,
    tree_workers: int,
    fallback_neighbors: tuple[int, ...],
    epsilon: float,
) -> dict:
    """Rasterize exact DTFE values with nearest-vertex incident-tet search.

    The nearest input site is usually a vertex of the containing simplex but is
    not guaranteed to be one for an arbitrary Delaunay simplex.  We therefore
    test incident stars progressively (K=1, then the registered wider K values)
    in exact barycentric coordinates.  Unresolved voxels are counted explicitly;
    no approximate density is substituted.
    """
    from numba import cuda

    output.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output.with_suffix(".progress.json")
    if output.exists():
        field = np.lib.format.open_memmap(output, mode="r+")
        if tuple(field.shape) != tuple(shape):
            raise RuntimeError(f"existing DTFE field has wrong shape: {field.shape} != {shape}")
    else:
        field = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32, shape=shape)
        field.fill(np.nan)
        field.flush()
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress.get("method") != "nearest_vertex_incident_tetrahedra":
            raise RuntimeError("existing DTFE progress uses an incompatible locator")
    else:
        progress = {
            "schema_version": 1,
            "method": "nearest_vertex_incident_tetrahedra",
            "cap": int(cap),
            "shape": list(shape),
            "completed_x": 0,
            "slabs": [],
        }

    cap_point_id = np.flatnonzero(np.asarray(points[:, 3], dtype=np.int8) == cap)
    cap_xyz = np.asarray(points[cap_point_id, :3], dtype=np.float64)
    tree_started = time.time()
    tree = cKDTree(cap_xyz, leafsize=32, compact_nodes=True, balanced_tree=True)
    tree_seconds = time.time() - tree_started

    d_points = cuda.to_device(np.ascontiguousarray(points[:, :3], dtype=np.float64))
    d_tets = cuda.to_device(np.ascontiguousarray(tets, dtype=np.int32))
    d_density = cuda.to_device(np.ascontiguousarray(vertex_density, dtype=np.float32))
    d_offsets = cuda.to_device(np.ascontiguousarray(incident_offsets, dtype=np.int64))
    d_incident = cuda.to_device(np.ascontiguousarray(incident_tetrahedron, dtype=np.int32))
    kernel = _incident_locator_kernel()
    started = time.time()
    completed_x = int(progress["completed_x"])

    with h5py.File(field_path, "r") as handle:
        for left in range(completed_x, shape[0], raster_slab):
            right = min(left + raster_slab, shape[0])
            slab_started = time.time()
            supported = np.asarray(
                handle["exposure_apodized"][left:right], dtype=np.float32
            ) > 0.0
            local_index = np.argwhere(supported)
            slab_values = np.full(supported.shape, np.nan, dtype=np.float32)
            primary_found = 0
            fallback_found: list[dict] = []
            if len(local_index):
                global_index = local_index.astype(np.float64)
                global_index[:, 0] += left
                queries = origin[None, :] + (global_index + 0.5) * cell_mpc
                _, nearest_local = tree.query(queries, k=1, workers=tree_workers)
                nearest_global = cap_point_id[np.asarray(nearest_local, dtype=np.int64)]
                values, containing = _locate_chunk_gpu(
                    kernel=kernel,
                    d_points=d_points,
                    d_tets=d_tets,
                    d_density=d_density,
                    d_offsets=d_offsets,
                    d_incident=d_incident,
                    queries=queries,
                    nearest_vertex=nearest_global,
                    epsilon=epsilon,
                    threads=threads,
                )
                primary_found = int(np.sum(containing >= 0))
                missing = containing < 0
                for requested_k in fallback_neighbors:
                    if not np.any(missing):
                        break
                    k = min(int(requested_k), len(cap_point_id))
                    _, fallback_local = tree.query(
                        queries[missing], k=k, workers=tree_workers
                    )
                    fallback_global = cap_point_id[np.asarray(fallback_local, dtype=np.int64)]
                    retry_values, retry_containing = _locate_chunk_gpu(
                        kernel=kernel,
                        d_points=d_points,
                        d_tets=d_tets,
                        d_density=d_density,
                        d_offsets=d_offsets,
                        d_incident=d_incident,
                        queries=queries[missing],
                        nearest_vertex=fallback_global,
                        epsilon=epsilon,
                        threads=threads,
                    )
                    found_now = int(np.sum(retry_containing >= 0))
                    fallback_found.append(
                        {"k": int(k), "queries": int(missing.sum()), "found": found_now}
                    )
                    values[missing] = retry_values
                    containing[missing] = retry_containing
                    missing = containing < 0
                slab_values[tuple(local_index.T)] = values
            field[left:right] = slab_values
            field.flush()
            finite = np.isfinite(slab_values) & supported
            slab_report = {
                "left": int(left),
                "right": int(right),
                "supported_voxels": int(supported.sum()),
                "primary_found": primary_found,
                "fallback_stages": fallback_found,
                "finite_supported_voxels": int(finite.sum()),
                "elapsed_seconds": time.time() - slab_started,
            }
            progress["slabs"].append(slab_report)
            progress["completed_x"] = int(right)
            atomic_json(progress_path, progress)
            print(json.dumps({"dtfe_slab": slab_report}), flush=True)

    supported_voxels = int(sum(row["supported_voxels"] for row in progress["slabs"]))
    finite_supported = int(sum(row["finite_supported_voxels"] for row in progress["slabs"]))
    finite_values = np.asarray(field[np.isfinite(field)], dtype=np.float32)
    report = {
        "cap": int(cap),
        "cap_name": CAP_NAME[cap],
        "shape": list(shape),
        "field_path": str(field_path),
        "output": str(output),
        "locator": "nearest vertex -> exact incident-tetrahedron barycentric test",
        "fallback_neighbors": [int(value) for value in fallback_neighbors],
        "barycentric_epsilon": float(epsilon),
        "cap_points": int(len(cap_point_id)),
        "tree_build_seconds": tree_seconds,
        "supported_voxels": supported_voxels,
        "finite_supported_voxels": finite_supported,
        "supported_coverage": float(finite_supported / max(supported_voxels, 1)),
        "finite_density_min": float(finite_values.min(initial=np.inf)),
        "finite_density_max": float(finite_values.max(initial=-np.inf)),
        "elapsed_seconds": time.time() - started,
        "progress": str(progress_path),
    }
    del (
        d_points, d_tets, d_density, d_offsets, d_incident,
        field, cap_xyz, cap_point_id, tree, finite_values,
    )
    torch.cuda.empty_cache()
    return report


def _load_dtfe_delta(
    *,
    density_path: Path,
    field_path: Path,
    shape: tuple[int, int, int],
    origin: np.ndarray,
    cell_mpc: float,
    curve: dict,
    cosmology: dict,
    device: str,
    slab: int,
) -> tuple[torch.Tensor, dict]:
    density = np.load(density_path, mmap_mode="r")
    delta = torch.zeros(shape, dtype=torch.float32, device=device)
    radius_grid = np.asarray(cosmology["radius_grid_mpc"], dtype=np.float64)
    redshift_grid = np.asarray(cosmology["redshift_grid"], dtype=np.float64)
    grid_z = np.asarray(curve["grid_z"], dtype=np.float64)
    ntilde = np.asarray(curve["ntilde"], dtype=np.float64)
    y = origin[1] + (np.arange(shape[1], dtype=np.float64) + 0.5) * cell_mpc
    z = origin[2] + (np.arange(shape[2], dtype=np.float64) + 0.5) * cell_mpc
    used, finite = 0, 0
    with h5py.File(field_path, "r") as handle:
        for left in range(0, shape[0], slab):
            right = min(left + slab, shape[0])
            exposure = np.asarray(handle["exposure_apodized"][left:right], dtype=np.float32)
            rho = np.asarray(density[left:right], dtype=np.float32)
            x = origin[0] + (np.arange(left, right, dtype=np.float64) + 0.5) * cell_mpc
            radius = np.sqrt(x[:, None, None]**2 + y[None, :, None]**2 + z[None, None, :]**2)
            redshift = np.interp(radius, radius_grid, redshift_grid)
            radial_density = np.interp(np.clip(redshift, grid_z[0], grid_z[-1]), grid_z, ntilde)
            nbar = radial_density * exposure
            valid = np.isfinite(rho) & (nbar > 0)
            values = np.zeros_like(rho, dtype=np.float32)
            values[valid] = np.clip(rho[valid] / nbar[valid] - 1.0, -1.0, 200.0) * exposure[valid]
            delta[left:right].copy_(torch.from_numpy(values).to(device))
            used += int(valid.sum())
            finite += int(np.isfinite(rho).sum())
    return delta, {"used_voxels": used, "finite_dtfe_voxels": finite}


def evaluate_rotation(rotation: int, args, adapter: dict, selection: dict) -> dict:
    assignment = np.load(args.assignment, mmap_mode="r")
    truth = np.load(args.p8_root / "parent_eigenvalues.npy", mmap_mode="r")
    points = np.load(args.points, mmap_mode="r")
    rotations = json.loads(args.rotations.read_text())
    train_folds, validation_fold, _ = fold_roles(rotations, rotation)
    auth = authoritative_mask(assignment)
    row_fold = np.asarray(assignment["fold"], dtype=np.int8)
    active_rows = np.flatnonzero(auth & np.isin(row_fold, (*train_folds, validation_fold)))
    parent = np.asarray(assignment["parent_node_id"][active_rows], dtype=np.int64)
    train = np.isin(row_fold[active_rows], train_folds)
    validation = row_fold[active_rows] == validation_fold
    cap_id = np.asarray(points[parent, 3], dtype=np.int8)
    positions = np.asarray(points[parent, :3], dtype=np.float64)
    raw = np.empty((len(parent), 3), dtype=np.float32)
    cap_reports = {}
    for cap in (0, 1):
        cap_name = CAP_NAME[cap]
        selected = cap_id == cap
        field_row = adapter["caps"][cap_name]
        shape = tuple(int(v) for v in field_row["shape"])
        delta, load_report = _load_dtfe_delta(
            density_path=args.output_root / f"dtfe_density_{cap_name}.npy",
            field_path=Path(field_row["field_path"]), shape=shape,
            origin=np.asarray(field_row["origin_mpc"], dtype=np.float64),
            cell_mpc=float(field_row["cell_mpc"]),
            curve=selection["rotations"][str(rotation)]["caps"][cap_name],
            cosmology=selection["cosmology"], device=args.device, slab=args.slab,
        )
        prediction, fft = _sample_tidal_eigenvalues(
            delta, positions=positions[selected],
            origin=np.asarray(field_row["origin_mpc"], dtype=np.float64),
            cell_mpc=float(field_row["cell_mpc"]), padding_voxels=args.padding_voxels,
            rsmooth_mpc=args.rsmooth_mpc,
        )
        raw[selected] = prediction
        cap_reports[cap_name] = {"n_sampled": int(selected.sum()), "load": load_report, "fft": fft}
    calibrated, affine = fit_affine_on_training(raw, np.asarray(truth[parent]), train)
    validation_parent = parent[validation]
    runtime = {"device": args.device, "padding_voxels": args.padding_voxels}
    raw_report = evaluate_complete_fold(
        parent_node_id=validation_parent, predicted_eigenvalues=raw[validation],
        truth_by_parent=truth, assignment=assignment, validation_fold=validation_fold,
        runtime=runtime,
    )
    calibrated_report = evaluate_complete_fold(
        parent_node_id=validation_parent, predicted_eigenvalues=calibrated[validation],
        truth_by_parent=truth, assignment=assignment, validation_fold=validation_fold,
        runtime=runtime,
    )
    out = args.output_root / f"rotation_{rotation}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "validation_parent_node_id.npy", validation_parent)
    np.save(out / "dtfe_raw_eigenvalues.npy", raw[validation])
    np.save(out / "dtfe_train_affine_eigenvalues.npy", calibrated[validation].astype(np.float32))
    report = {
        "schema_version": 1, "estimator": "exact_piecewise_linear_dtfe",
        "rotation": rotation, "train_folds": list(train_folds),
        "validation_fold": validation_fold, "affine": affine,
        "raw": raw_report, "train_affine": calibrated_report, "caps": cap_reports,
    }
    atomic_json(out / "dtfe_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "build", "evaluate", "all"), default="all")
    parser.add_argument("--p8-root", type=Path, default=P8_ROOT)
    parser.add_argument("--output-root", type=Path, default=P8_ROOT / "classical/dtfe_fullcap_v1")
    parser.add_argument("--points", type=Path, default=POINTS)
    parser.add_argument("--tets", type=Path, default=TETS)
    parser.add_argument("--volumes", type=Path, default=VOLUMES)
    parser.add_argument("--field-adapter", type=Path, default=FIELD_ADAPTER)
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--assignment", type=Path, default=ASSIGNMENT)
    parser.add_argument("--rotations", type=Path, default=ROTATIONS)
    parser.add_argument("--screen-rotations", type=int, nargs="+", default=(0, 2))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk", type=int, default=1_000_000)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--raster-slab", type=int, default=2,
                        help="number of grid-x planes per resumable locator chunk")
    parser.add_argument("--tree-workers", type=int, default=-1,
                        help="cKDTree workers; -1 uses all CPUs in the allocation")
    parser.add_argument("--fallback-neighbors", type=int, nargs="+", default=(8, 32, 128),
                        help="progressive K-nearest incident-star retries after K=1 misses")
    parser.add_argument("--barycentric-epsilon", type=float, default=1.0e-8)
    parser.add_argument("--slab", type=int, default=8)
    parser.add_argument("--padding-voxels", type=int, default=20)
    parser.add_argument("--rsmooth-mpc", type=float, default=RSMOOTH_MPC)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    adapter = json.loads((args.field_adapter / "adapter_manifest.json").read_text())
    points = np.load(args.points, mmap_mode="r")
    tets = np.load(args.tets, mmap_mode="r")
    git_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if args.mode in ("preflight", "all"):
        report = {
            CAP_NAME[cap]: preflight_aabb_visits(
                points, tets, cap=cap,
                origin=np.asarray(adapter["caps"][CAP_NAME[cap]]["origin_mpc"], dtype=np.float64),
                shape=tuple(adapter["caps"][CAP_NAME[cap]]["shape"]),
                cell_mpc=float(adapter["caps"][CAP_NAME[cap]]["cell_mpc"]), chunk=args.chunk,
            ) for cap in (0, 1)
        }
        atomic_json(args.output_root / "raster_preflight.json", report)
        print(json.dumps(report, indent=2), flush=True)
        if args.mode == "preflight":
            return
    if args.mode in ("build", "all"):
        if not torch.cuda.is_available():
            raise RuntimeError("DTFE build requires a CUDA interactive allocation")
        volumes = np.load(args.volumes, mmap_mode="r")
        vertex_path = args.output_root / "vertex_density.npy"
        if vertex_path.exists():
            vertex_density = np.load(vertex_path, mmap_mode="r")
        else:
            vertex_density = compute_vertex_density(
                tets, volumes, len(points), device=args.device, chunk=args.chunk
            )
            np.save(vertex_path, vertex_density)
        incident_offsets, incident_tetrahedron, incident_manifest = build_incident_csr(
            tets,
            len(points),
            output_root=args.output_root,
            device=args.device,
        )
        cap_reports = {}
        for cap in (0, 1):
            name = CAP_NAME[cap]
            row = adapter["caps"][name]
            cap_output = args.output_root / f"dtfe_density_{name}.npy"
            cap_report_path = args.output_root / f"dtfe_density_{name}_report.json"
            if cap_output.exists() and cap_report_path.exists():
                cap_reports[name] = json.loads(cap_report_path.read_text())
            else:
                cap_reports[name] = rasterize_cap(
                    points=points, tets=tets, vertex_density=vertex_density, cap=cap,
                    incident_offsets=incident_offsets,
                    incident_tetrahedron=incident_tetrahedron,
                    field_path=Path(row["field_path"]),
                    origin=np.asarray(row["origin_mpc"], dtype=np.float64),
                    shape=tuple(row["shape"]), cell_mpc=float(row["cell_mpc"]),
                    output=cap_output, threads=args.threads,
                    raster_slab=args.raster_slab,
                    tree_workers=args.tree_workers,
                    fallback_neighbors=tuple(args.fallback_neighbors),
                    epsilon=args.barycentric_epsilon,
                )
                atomic_json(cap_report_path, cap_reports[name])
        build_report = {
            "schema_version": 1, "estimator": "exact_piecewise_linear_dtfe",
            "git_revision": git_revision,
            "vertex_density": "4/sum incident tetrahedron volumes",
            "rasterization": (
                "voxel-centric nearest-vertex incident-tetrahedron point location; "
                "barycentric interpolation in immutable global Delaunay tetrahedra"
            ),
            "incident_csr": incident_manifest,
            "caps": cap_reports,
            "inputs": {
                "points": str(args.points), "points_sha256": sha256(args.points),
                "tets": str(args.tets), "tets_sha256": sha256(args.tets),
                "volumes": str(args.volumes), "volumes_sha256": sha256(args.volumes),
                "field_adapter": str(args.field_adapter / "adapter_manifest.json"),
                "field_adapter_sha256": sha256(args.field_adapter / "adapter_manifest.json"),
            },
        }
        atomic_json(args.output_root / "dtfe_build_report.json", build_report)
        if min(v["supported_coverage"] for v in cap_reports.values()) < 0.99:
            raise RuntimeError("exact DTFE raster covers less than 99% of supported cap voxels")
        (args.output_root / "DTFE_FIELD_READY").write_text("exact piecewise-linear full-cap DTFE\n")
    if args.mode in ("evaluate", "all"):
        if not torch.cuda.is_available():
            raise RuntimeError("DTFE evaluation requires a CUDA interactive allocation")
        selection = json.loads(args.selection.read_text())
        reports = [evaluate_rotation(r, args, adapter, selection) for r in args.screen_rotations]
        scores = [r["train_affine"]["primary_macro_r2_lambda1"] for r in reports]
        summary = {
            "schema_version": 1, "stage": "P8 matched exact full-cap DTFE",
            "estimator": "exact_piecewise_linear_dtfe",
            "git_revision": git_revision,
            "screen_rotations": list(args.screen_rotations),
            "primary_score_by_rotation": scores, "primary_score_mean": float(np.mean(scores)),
            "calibration": "three scalar affine maps fit on registered training folds only",
            "adoption_row_ready": True,
        }
        atomic_json(args.output_root / "dtfe_summary.json", summary)
        (args.output_root / "P8_EXACT_DTFE_READY").write_text(
            f"mean_macro_r2_lambda1={summary['primary_score_mean']:.8f}\n"
        )
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
