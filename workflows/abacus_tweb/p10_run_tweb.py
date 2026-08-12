#!/usr/bin/env python3
"""Run the frozen P10 CACTUS T-web solve on a verified phase density grid.

The full float32 density ``.npy`` is memory-mapped independently by every MPI
rank.  Each rank copies only its shift-compatible x slab, applies the existing
memory-optimized CACTUS implementation, and writes one atomic rank product.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = Path(__file__).resolve().parent
for import_root in (REPO_ROOT, WORKFLOW_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from p10_phase_assets import DEFAULT_REGISTRY, load_registry, sha256_file  # noqa: E402


class TWebBuildError(RuntimeError):
    """The P10 density input or T-web output contract failed."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def balanced_slice(ngrid: int, rank: int, size: int) -> tuple[int, int]:
    """Return the x slab used by ``shift``'s quotient/remainder split."""
    if size <= 0 or rank < 0 or rank >= size:
        raise ValueError(f"invalid rank/size: {rank}/{size}")
    base, remainder = divmod(ngrid, size)
    if rank < remainder:
        start = rank * (base + 1)
        return start, start + base + 1
    start = rank * base + remainder
    return start, start + base


def default_density_paths(registry: dict[str, Any], phase: str) -> tuple[Path, Path]:
    root = Path(registry["path_templates"]["phase_output"].format(phase=phase))
    stem = f"AbacusSummit_base_c000_{phase}_z0.200_ngrid2048_ab10_tsc_counts"
    return root / "targets/density" / f"{stem}.npy", root / "targets/density" / f"{stem}.manifest.json"


def default_output_dir(registry: dict[str, Any], phase: str) -> Path:
    root = Path(registry["path_templates"]["phase_output"].format(phase=phase))
    return root / "targets/tweb/backend_optimized_ngrid_2048_rsmooth_7"


def validate_density_input(
    density_path: Path,
    density_manifest_path: Path,
    registry: dict[str, Any],
    phase: str,
) -> dict[str, Any]:
    if not density_path.is_file() or not density_manifest_path.is_file():
        raise TWebBuildError("density grid and manifest must both exist")
    manifest = json.loads(density_manifest_path.read_text())
    build = manifest.get("build")
    target = registry["target_contract"]
    if manifest.get("phase") != phase or not isinstance(build, dict):
        raise TWebBuildError("density manifest phase/build mismatch")
    if Path(build.get("output", "")).resolve() != density_path.resolve():
        raise TWebBuildError("density manifest output does not identify the input grid")
    if build.get("processed_file_count") != 136:
        raise TWebBuildError("production density must contain all 136 A+B slabs")
    if build.get("max_files_per_directory") is not None:
        raise TWebBuildError("technical canary density is not a production T-web input")
    if float(build.get("relative_count_error", 1.0)) > 2e-6:
        raise TWebBuildError("density count-conservation gate failed")
    array = np.load(density_path, mmap_mode="r")
    expected_shape = (target["grid_size"],) * 3
    if array.shape != expected_shape or array.dtype != np.float32:
        raise TWebBuildError(
            f"density shape/dtype mismatch: {array.shape}/{array.dtype}"
        )
    return {
        "path": str(density_path.resolve()),
        "bytes": density_path.stat().st_size,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "manifest": str(density_manifest_path.resolve()),
        "manifest_sha256": sha256_file(density_manifest_path),
        "particle_count": int(build["particle_count"]),
        "relative_count_error": float(build["relative_count_error"]),
        "verified": True,
    }


def write_rank_output(
    path: Path,
    *,
    cweb: np.ndarray,
    eig_vals: np.ndarray,
    x_start: int,
    x_end: int,
    ngrid: int,
    boxsize: float,
    threshold: float,
    rsmooth: float,
) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            cweb=cweb.astype(np.uint8),
            eig_vals=eig_vals.astype(np.float32),
            x_start=x_start,
            x_end=x_end,
            ngrid=ngrid,
            boxsize=boxsize,
            threshold=threshold,
            Rsmooth=rsmooth,
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_rank_outputs(
    output_dir: Path,
    *,
    expected_ranks: int,
    ngrid: int,
    boxsize: float,
    threshold: float,
    rsmooth: float,
) -> dict[str, Any]:
    paths = sorted(output_dir.glob("abacus_cactus_tweb_rank*.npz"))
    if len(paths) != expected_ranks:
        raise TWebBuildError(
            f"expected {expected_ranks} rank outputs, found {len(paths)}"
        )
    records: list[dict[str, Any]] = []
    previous_end = 0
    for rank, path in enumerate(paths):
        expected_name = f"abacus_cactus_tweb_rank{rank:04d}.npz"
        if path.name != expected_name:
            raise TWebBuildError(f"rank filename mismatch: {path.name}")
        with np.load(path, mmap_mode="r") as data:
            start, end = int(data["x_start"]), int(data["x_end"])
            eig_shape = tuple(data["eig_vals"].shape)
            cweb_shape = tuple(data["cweb"].shape)
            metadata = {
                "ngrid": int(data["ngrid"]),
                "boxsize": float(data["boxsize"]),
                "threshold": float(data["threshold"]),
                "rsmooth": float(data["Rsmooth"]),
            }
        local_shape = (end - start, ngrid, ngrid)
        if start != previous_end or eig_shape != (3, *local_shape) or cweb_shape != local_shape:
            raise TWebBuildError(f"rank {rank}: non-contiguous or invalid output shape")
        if metadata != {
            "ngrid": ngrid,
            "boxsize": boxsize,
            "threshold": threshold,
            "rsmooth": rsmooth,
        }:
            raise TWebBuildError(f"rank {rank}: T-web metadata mismatch")
        records.append(
            {
                "rank": rank,
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "x_start": start,
                "x_end": end,
            }
        )
        previous_end = end
    if previous_end != ngrid:
        raise TWebBuildError(f"rank outputs cover x=[0,{previous_end}), not [0,{ngrid})")
    return {
        "rank_count": expected_ranks,
        "total_bytes": sum(record["bytes"] for record in records),
        "x_coverage": [0, previous_end],
        "records": records,
        "verified": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--density", type=Path)
    parser.add_argument("--density-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    registry = load_registry(args.registry)
    if args.phase not in registry["phases"]:
        raise TWebBuildError(f"unregistered phase: {args.phase}")
    density_default, manifest_default = default_density_paths(registry, args.phase)
    density_path = args.density or density_default
    density_manifest_path = args.density_manifest or manifest_default
    output_dir = args.output_dir or default_output_dir(registry, args.phase)
    density_report = validate_density_input(
        density_path, density_manifest_path, registry, args.phase
    )
    if args.preflight_only:
        print(json.dumps(density_report, indent=2, sort_keys=True))
        return 0

    from shift import mpiutils
    from shared.resource_requirements import require_cpu_mpi_slurm
    from abacus_process_particles2 import run_tweb_memory_optimized

    require_cpu_mpi_slurm("p10_run_tweb.py", min_tasks=2)
    mpi = mpiutils.MPI()
    rank, size = mpi.rank, mpi.size
    target = registry["target_contract"]
    ngrid = int(target["grid_size"])
    boxsize = float(target["box_size_mpc_h"])
    threshold = float(target["web_threshold"])
    rsmooth = float(target["tidal_smoothing_mpc_h"])

    existing = sorted(output_dir.glob("abacus_cactus_tweb_rank*.npz")) if output_dir.exists() else []
    if existing:
        raise TWebBuildError(
            f"refusing to mix with {len(existing)} existing rank outputs in {output_dir}"
        )
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    mpi.wait()

    x_start, x_end = balanced_slice(ngrid, rank, size)
    density = np.load(density_path, mmap_mode="r")
    density_local = np.ascontiguousarray(density[x_start:x_end], dtype=np.float32)
    del density
    cweb, eig_vals = run_tweb_memory_optimized(
        density_local,
        boxsize,
        ngrid,
        threshold,
        mpi,
        Rsmooth=rsmooth,
        verbose=True,
    )
    rank_path = output_dir / f"abacus_cactus_tweb_rank{rank:04d}.npz"
    write_rank_output(
        rank_path,
        cweb=cweb,
        eig_vals=eig_vals,
        x_start=x_start,
        x_end=x_end,
        ngrid=ngrid,
        boxsize=boxsize,
        threshold=threshold,
        rsmooth=rsmooth,
    )
    print(f"[rank {rank}] wrote {rank_path}", flush=True)
    mpi.wait()

    if rank == 0:
        outputs = validate_rank_outputs(
            output_dir,
            expected_ranks=size,
            ngrid=ngrid,
            boxsize=boxsize,
            threshold=threshold,
            rsmooth=rsmooth,
        )
        git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        payload = {
            "schema_version": "p10-tweb-complete-v1",
            "created_utc": utc_now(),
            "phase": args.phase,
            "role": registry["phases"][args.phase]["role"],
            "registry": str(args.registry.resolve()),
            "registry_sha256": sha256_file(args.registry),
            "git_sha": git_sha,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "mpi_size": size,
            "density": density_report,
            "target_contract": target,
            "outputs": outputs,
        }
        atomic_write_json(output_dir / "TWEB_COMPLETE.json", payload)
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    mpi.wait()
    mpi.end()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (TWebBuildError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
