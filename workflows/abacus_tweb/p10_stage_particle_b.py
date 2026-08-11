#!/usr/bin/env python3
"""List, stage, and verify the registered P10 particle-B products.

The workflow is idempotent and intentionally has no delete operation.  A staged
payload becomes usable only after counts, POSIX checksums, ASDF readability, and
phase/redshift headers pass and an atomic completion marker is written.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from p10_phase_assets import (
    DEFAULT_REGISTRY,
    RegistryError,
    expand_phase,
    inspect_hpss_b,
    load_registry,
    sha256_file,
    utc_now,
)


DEFAULT_STAGING_ROOT = Path(
    "/pscratch/sd/d/dkololgi/abacus/p10_multiphase/particle_b"
)


class StageError(RuntimeError):
    """The staging or verification contract failed."""


def parse_checksum_manifest(path: Path) -> dict[str, tuple[int, int]]:
    records: dict[str, tuple[int, int]] = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 3:
            raise StageError(f"{path}:{line_number}: expected CRC SIZE FILE")
        crc, size, name = parts
        try:
            record = (int(crc), int(size))
        except ValueError as exc:
            raise StageError(f"{path}:{line_number}: invalid checksum record") from exc
        if name in records:
            raise StageError(f"{path}:{line_number}: duplicate file {name}")
        records[name] = record
    return records


def posix_cksum(paths: list[Path]) -> dict[str, tuple[int, int]]:
    if not paths:
        return {}
    result = subprocess.run(
        ["/usr/bin/cksum", *map(str, paths)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise StageError(f"cksum failed: {result.stderr[-4000:]}")
    records: dict[str, tuple[int, int]] = {}
    for raw_line in result.stdout.splitlines():
        crc, size, name = raw_line.split(maxsplit=2)
        records[Path(name).name] = (int(crc), int(size))
    return records


def verify_checksums(directory: Path, pattern: str, manifest_name: str) -> dict[str, Any]:
    manifest = directory / manifest_name
    expected = parse_checksum_manifest(manifest)
    files = sorted(directory.glob(pattern))
    expected_names = set(expected)
    actual_names = {path.name for path in files}
    if expected_names != actual_names:
        raise StageError(
            f"{directory}: checksum/file mismatch; "
            f"missing={sorted(expected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names)}"
        )
    actual = posix_cksum(files)
    mismatches = {
        name: {"expected": expected[name], "actual": actual.get(name)}
        for name in sorted(expected)
        if expected[name] != actual.get(name)
    }
    if mismatches:
        raise StageError(f"{directory}: POSIX checksum mismatch: {mismatches}")
    return {
        "manifest": str(manifest),
        "file_count": len(files),
        "bytes": sum(size for _, size in actual.values()),
        "verified": True,
    }


def verify_asdf_headers(paths: list[Path], phase: str, redshift: float) -> dict[str, Any]:
    try:
        import asdf
    except ImportError as exc:
        raise StageError("asdf is required for particle-B readability checks") from exc

    expected_sim = f"AbacusSummit_base_c000_{phase}"
    phase_mismatches: list[dict[str, Any]] = []
    redshift_mismatches: list[dict[str, Any]] = []
    unreadable: list[dict[str, str]] = []
    for path in paths:
        try:
            with asdf.open(path, lazy_load=True) as af:
                header = dict(af.tree.get("header", {}))
                sim_name = header.get("SimName")
                file_redshift = header.get("Redshift")
                if sim_name != expected_sim:
                    phase_mismatches.append(
                        {"path": str(path), "SimName": sim_name}
                    )
                if file_redshift is None or abs(float(file_redshift) - redshift) > 1e-8:
                    redshift_mismatches.append(
                        {"path": str(path), "Redshift": file_redshift}
                    )
        except Exception as exc:  # pragma: no cover - external corruption path
            unreadable.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    if unreadable or phase_mismatches or redshift_mismatches:
        raise StageError(
            "ASDF metadata verification failed: "
            f"unreadable={len(unreadable)}, phase={len(phase_mismatches)}, "
            f"redshift={len(redshift_mismatches)}"
        )
    return {
        "file_count": len(paths),
        "expected_sim_name": expected_sim,
        "expected_redshift": redshift,
        "readable": True,
    }


def verify_b_tree(
    root: Path,
    *,
    phase: str,
    registry: dict[str, Any],
    checksums: bool,
    asdf_headers: bool,
) -> dict[str, Any]:
    contract = registry["staging_contract"]
    field_dir = root / "field_rv_B"
    halo_dir = root / "halo_rv_B"
    field = sorted(field_dir.glob("field_rv_B_*.asdf")) if field_dir.is_dir() else []
    halo = sorted(halo_dir.glob("halo_rv_B_*.asdf")) if halo_dir.is_dir() else []
    if len(field) != contract["expected_field_b_slabs"]:
        raise StageError(
            f"{field_dir}: expected {contract['expected_field_b_slabs']} ASDF slabs, "
            f"got {len(field)}"
        )
    if len(halo) != contract["expected_halo_b_slabs"]:
        raise StageError(
            f"{halo_dir}: expected {contract['expected_halo_b_slabs']} ASDF slabs, "
            f"got {len(halo)}"
        )
    for directory in (field_dir, halo_dir):
        manifest = directory / contract["checksum_filename"]
        if not manifest.is_file() or manifest.stat().st_size == 0:
            raise StageError(f"missing checksum manifest: {manifest}")

    report: dict[str, Any] = {
        "phase": phase,
        "root": str(root),
        "field_asdf_count": len(field),
        "halo_asdf_count": len(halo),
        "payload_bytes": sum(path.stat().st_size for path in field + halo),
        "checksums": None,
        "asdf_headers": None,
    }
    if checksums:
        report["checksums"] = {
            "field": verify_checksums(
                field_dir, "field_rv_B_*.asdf", contract["checksum_filename"]
            ),
            "halo": verify_checksums(
                halo_dir, "halo_rv_B_*.asdf", contract["checksum_filename"]
            ),
        }
    if asdf_headers:
        report["asdf_headers"] = verify_asdf_headers(
            field + halo, phase, registry["target_contract"]["redshift"]
        )
    report["verified"] = True
    return report


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def phase_staging_paths(
    staging_root: Path, phase: str
) -> tuple[Path, Path, Path]:
    phase_root = staging_root / f"AbacusSummit_base_c000_{phase}"
    particle_root = phase_root / "halos/z0.200"
    marker = phase_root / "B_STAGE_COMPLETE.json"
    return phase_root, particle_root, marker


def list_source(
    registry: dict[str, Any], phase: str, *, htar: str
) -> dict[str, Any]:
    expanded = expand_phase(registry, phase)
    source = expanded["particle_b"]
    if source["kind"] == "hpss":
        report = inspect_hpss_b(source, registry["staging_contract"], htar=htar)
        if not report["ready"]:
            raise StageError(f"{phase}: HPSS B listing failed: {report.get('errors')}")
        return report
    return {
        "kind": "cfs",
        "root": source["root"],
        "ready": True,
        "note": "online source; use verify mode for full validation",
    }


def stage_hpss(
    registry_path: Path,
    registry: dict[str, Any],
    phase: str,
    staging_root: Path,
    *,
    htar: str,
    headroom_gib: int,
    checksums: bool,
    asdf_headers: bool,
) -> dict[str, Any]:
    expanded = expand_phase(registry, phase)
    source = expanded["particle_b"]
    if source["kind"] != "hpss":
        raise StageError(f"{phase}: stage mode is only valid for an HPSS source")
    listing = list_source(registry, phase, htar=htar)
    payload = listing.get("payload_bytes")
    if payload is None:
        raise StageError("HTAR verbose listing did not expose payload sizes")
    phase_root, particle_root, marker = phase_staging_paths(staging_root, phase)

    if marker.is_file():
        verification = verify_b_tree(
            particle_root,
            phase=phase,
            registry=registry,
            checksums=checksums,
            asdf_headers=asdf_headers,
        )
        return {
            "phase": phase,
            "action": "reuse_verified_stage",
            "marker": str(marker),
            "verification": verification,
        }

    phase_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(phase_root).free
    required = int(payload) + headroom_gib * 1024**3
    if free < required:
        raise StageError(
            f"insufficient scratch: free={free}, payload={payload}, "
            f"headroom_gib={headroom_gib}, required={required}"
        )

    command = [htar, "-xvf", source["archive"], *source["members"]]
    result = subprocess.run(command, cwd=phase_root, check=False)
    if result.returncode != 0:
        raise StageError(f"HTAR restore failed with return code {result.returncode}")
    verification = verify_b_tree(
        particle_root,
        phase=phase,
        registry=registry,
        checksums=checksums,
        asdf_headers=asdf_headers,
    )
    marker_payload = {
        "schema_version": "p10-b-stage-complete-v1",
        "created_utc": utc_now(),
        "phase": phase,
        "source": source,
        "registry": str(registry_path.resolve()),
        "registry_sha256": sha256_file(registry_path),
        "listing": listing,
        "verification": verification,
        "cleanup_authorized": False,
    }
    atomic_write_json(marker, marker_payload)
    return {
        "phase": phase,
        "action": "restored_and_verified",
        "marker": str(marker),
        "verification": verification,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--mode", choices=("list", "stage", "verify"), required=True)
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING_ROOT)
    parser.add_argument("--htar", default="htar")
    parser.add_argument("--headroom-gib", type=int)
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument("--skip-asdf-headers", action="store_true")
    parser.add_argument("--out", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    registry = load_registry(args.registry)
    if args.phase not in registry["phases"]:
        raise RegistryError(f"phase is not registered: {args.phase}")
    source = expand_phase(registry, args.phase)["particle_b"]
    checksums = not args.skip_checksums
    asdf_headers = not args.skip_asdf_headers
    if args.mode == "list":
        result = list_source(registry, args.phase, htar=args.htar)
    elif args.mode == "stage":
        result = stage_hpss(
            args.registry,
            registry,
            args.phase,
            args.staging_root,
            htar=args.htar,
            headroom_gib=(
                args.headroom_gib
                if args.headroom_gib is not None
                else registry["staging_contract"]["headroom_gib"]
            ),
            checksums=checksums,
            asdf_headers=asdf_headers,
        )
    else:
        if source["kind"] == "cfs":
            root = Path(source["root"])
        else:
            _, root, _ = phase_staging_paths(args.staging_root, args.phase)
        result = verify_b_tree(
            root,
            phase=args.phase,
            registry=registry,
            checksums=checksums,
            asdf_headers=asdf_headers,
        )
    payload = {
        "schema_version": "p10-b-stage-result-v1",
        "created_utc": utc_now(),
        "phase": args.phase,
        "mode": args.mode,
        "result": result,
    }
    if args.out:
        atomic_write_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RegistryError, StageError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
