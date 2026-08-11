#!/usr/bin/env python3
"""Build a P10 10-percent A+B particle density field by streaming ASDF slabs.

Unlike the legacy ph000 helper, this command is phase-explicit, registry-driven,
and accepts A and B from different storage roots.  It writes count density with
TSC assignment; normalization and the tidal solve remain downstream frozen
operations.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = Path(__file__).resolve().parent
for import_root in (REPO_ROOT, WORKFLOW_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from p10_phase_assets import (  # noqa: E402
    DEFAULT_REGISTRY,
    RegistryError,
    expand_phase,
    load_registry,
    sha256_file,
)
from p10_stage_particle_b import phase_staging_paths  # noqa: E402


class DensityBuildError(RuntimeError):
    """The density input or build contract failed."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def manifest_fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(f"{path}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def resolve_particle_roots(
    registry_path: Path,
    registry: dict[str, Any],
    phase: str,
    staging_root: Path,
) -> tuple[Path, Path, Path | None]:
    expanded = expand_phase(registry, phase)
    a_root = Path(expanded["assets"]["snapshot_root"])
    source = expanded["particle_b"]
    marker: Path | None = None
    if source["kind"] == "cfs":
        b_root = Path(source["root"])
    else:
        _, b_root, marker = phase_staging_paths(staging_root, phase)
        if not marker.is_file():
            raise DensityBuildError(
                f"{phase}: missing verified B-stage marker {marker}; run "
                "p10_stage_particle_b.py --mode stage first"
            )
        marker_payload = json.loads(marker.read_text())
        if marker_payload.get("phase") != phase:
            raise DensityBuildError(f"{marker}: phase mismatch")
        if marker_payload.get("registry_sha256") != sha256_file(registry_path):
            raise DensityBuildError(
                f"{marker}: registry hash differs; reverify the staged payload"
            )
        if marker_payload.get("verification", {}).get("verified") is not True:
            raise DensityBuildError(f"{marker}: payload is not verified")
    return a_root, b_root, marker


def _slab_record(
    directory: Path,
    pattern: str,
    expected_count: int,
    checksum_filename: str,
) -> dict[str, Any]:
    files = sorted(directory.glob(pattern)) if directory.is_dir() else []
    checksum = directory / checksum_filename
    ready = (
        len(files) == expected_count
        and checksum.is_file()
        and checksum.stat().st_size > 0
    )
    return {
        "directory": str(directory),
        "pattern": pattern,
        "expected_count": expected_count,
        "file_count": len(files),
        "files": [str(path) for path in files],
        "checksum_manifest": str(checksum),
        "checksum_manifest_exists": checksum.is_file(),
        "total_bytes": sum(path.stat().st_size for path in files),
        "fingerprint": manifest_fingerprint(files) if files else None,
        "ready": ready,
    }


def inspect_particle_inputs(
    registry: dict[str, Any],
    phase: str,
    a_root: Path,
    b_root: Path,
    *,
    inspect_headers: bool,
) -> dict[str, Any]:
    contract = registry["staging_contract"]
    records = {
        "field_A": _slab_record(
            a_root / "field_rv_A",
            "field_rv_A_*.asdf",
            contract["expected_field_a_slabs"],
            contract["checksum_filename"],
        ),
        "halo_A": _slab_record(
            a_root / "halo_rv_A",
            "halo_rv_A_*.asdf",
            contract["expected_halo_a_slabs"],
            contract["checksum_filename"],
        ),
        "field_B": _slab_record(
            b_root / "field_rv_B",
            "field_rv_B_*.asdf",
            contract["expected_field_b_slabs"],
            contract["checksum_filename"],
        ),
        "halo_B": _slab_record(
            b_root / "halo_rv_B",
            "halo_rv_B_*.asdf",
            contract["expected_halo_b_slabs"],
            contract["checksum_filename"],
        ),
    }
    if not all(record["ready"] for record in records.values()):
        failed = [name for name, record in records.items() if not record["ready"]]
        raise DensityBuildError(f"{phase}: particle directories failed: {failed}")

    header_report = None
    if inspect_headers:
        try:
            import asdf
        except ImportError as exc:
            raise DensityBuildError("asdf is required for header checks") from exc
        expected_sim = f"AbacusSummit_base_c000_{phase}"
        expected_z = registry["target_contract"]["redshift"]
        checked = []
        for sample_name, record in records.items():
            sample_fraction_key = (
                "ParticleSubsampleA" if sample_name.endswith("_A") else "ParticleSubsampleB"
            )
            expected_fraction = (
                registry["target_contract"]["particle_subsamples"]["A_fraction"]
                if sample_name.endswith("_A")
                else registry["target_contract"]["particle_subsamples"]["B_fraction"]
            )
            paths = [Path(path) for path in record["files"]]
            for path in (paths[0], paths[-1]):
                with asdf.open(path, lazy_load=True) as af:
                    header = dict(af.tree.get("header", {}))
                if header.get("SimName") != expected_sim:
                    raise DensityBuildError(f"{path}: unexpected SimName")
                if abs(float(header.get("Redshift")) - expected_z) > 1e-8:
                    raise DensityBuildError(f"{path}: unexpected Redshift")
                if abs(float(header.get(sample_fraction_key)) - expected_fraction) > 1e-8:
                    raise DensityBuildError(
                        f"{path}: unexpected {sample_fraction_key}"
                    )
                if abs(float(header.get("BoxSizeHMpc")) - registry["target_contract"]["box_size_mpc_h"]) > 1e-8:
                    raise DensityBuildError(f"{path}: unexpected BoxSizeHMpc")
                checked.append(str(path))
        header_report = {
            "checked_files": checked,
            "file_count": len(checked),
            "sim_name": expected_sim,
            "redshift": expected_z,
            "verified": True,
        }
    return {
        "phase": phase,
        "a_root": str(a_root),
        "b_root": str(b_root),
        "directories": records,
        "headers": header_report,
        "ready": True,
    }


def default_output(registry: dict[str, Any], phase: str, ngrid: int) -> Path:
    expanded = expand_phase(registry, phase)
    return (
        Path(expanded["assets"]["phase_output"])
        / "targets/density"
        / (
            f"AbacusSummit_base_c000_{phase}_z0.200_ngrid{ngrid}_"
            "ab10_tsc_counts.npy"
        )
    )


def build_density(
    input_report: dict[str, Any],
    output: Path,
    *,
    ngrid: int,
    boxsize: float,
    threads: int,
    max_files_per_directory: int | None,
) -> dict[str, Any]:
    from abacusnbody.analysis.tsc import tsc_parallel
    from abacusnbody.data.read_abacus import read_asdf

    if output.exists():
        raise DensityBuildError(f"refusing to overwrite existing density field: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.npy")
    if temporary.exists():
        raise DensityBuildError(f"temporary output already exists: {temporary}")

    grid = np.zeros((ngrid, ngrid, ngrid), dtype=np.float32)
    particles = 0
    processed_files = []
    started = time.monotonic()
    try:
        for sample_name in ("field_A", "halo_A", "field_B", "halo_B"):
            paths = [
                Path(path)
                for path in input_report["directories"][sample_name]["files"]
            ]
            if max_files_per_directory is not None:
                paths = paths[:max_files_per_directory]
            for path in paths:
                data = read_asdf(path, verbose=False)
                positions = np.asarray(data["pos"], dtype=np.float32)
                positions = np.mod(positions, boxsize)
                tsc_parallel(positions, grid, boxsize, nthread=threads)
                particles += int(len(positions))
                processed_files.append(str(path))
                del data, positions

        deposited = float(np.sum(grid, dtype=np.float64))
        relative_error = abs(deposited - particles) / max(particles, 1)
        if relative_error > 2e-6:
            raise DensityBuildError(
                f"TSC count conservation failed: particles={particles}, "
                f"deposited={deposited}, relative_error={relative_error}"
            )
        np.save(temporary, grid)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    elapsed = time.monotonic() - started
    return {
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "ngrid": ngrid,
        "boxsize_mpc_h": boxsize,
        "dtype": str(grid.dtype),
        "particle_count": particles,
        "deposited_count": deposited,
        "relative_count_error": relative_error,
        "processed_file_count": len(processed_files),
        "processed_files": processed_files,
        "max_files_per_directory": max_files_per_directory,
        "wall_seconds": elapsed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--phase", required=True)
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase/particle_b"),
    )
    parser.add_argument("--ngrid", type=int)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-header-checks", action="store_true")
    parser.add_argument(
        "--max-files-per-directory",
        type=int,
        help="technical canary only; never label the result as a production target",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    registry = load_registry(args.registry)
    if args.phase not in registry["phases"]:
        raise RegistryError(f"phase is not registered: {args.phase}")
    ngrid = args.ngrid or registry["target_contract"]["grid_size"]
    boxsize = registry["target_contract"]["box_size_mpc_h"]
    output = args.output or default_output(registry, args.phase, ngrid)
    manifest = args.manifest or output.with_suffix(".manifest.json")
    a_root, b_root, stage_marker = resolve_particle_roots(
        args.registry, registry, args.phase, args.staging_root
    )
    inputs = inspect_particle_inputs(
        registry,
        args.phase,
        a_root,
        b_root,
        inspect_headers=not args.skip_header_checks,
    )
    payload: dict[str, Any] = {
        "schema_version": "p10-density-build-v1",
        "created_utc": utc_now(),
        "phase": args.phase,
        "role": registry["phases"][args.phase]["role"],
        "registry": str(args.registry.resolve()),
        "registry_sha256": sha256_file(args.registry),
        "target_contract": registry["target_contract"],
        "stage_marker": str(stage_marker) if stage_marker else None,
        "inputs": inputs,
        "preflight_only": args.preflight_only,
        "build": None,
    }
    if not args.preflight_only:
        payload["build"] = build_density(
            inputs,
            output,
            ngrid=ngrid,
            boxsize=boxsize,
            threads=args.threads,
            max_files_per_directory=args.max_files_per_directory,
        )
    atomic_write_json(manifest, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RegistryError, DensityBuildError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
