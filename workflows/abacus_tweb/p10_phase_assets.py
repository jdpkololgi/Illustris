#!/usr/bin/env python3
"""Validate the frozen P10 phase registry and audit source readiness.

This command is deliberately read-only.  It proves that the inputs required to
build an independent phase exist, while keeping target generation, HPSS restore,
and blind-phase truth access as separate explicit operations.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "configs/p10_phase_registry_v1.json"
EXPECTED_PHASES = tuple(f"ph{i:03d}" for i in range(1, 7))
ASDF_RE = re.compile(r"(field|halo)_rv_B_\d{3}\.asdf$")
FITS_REQUIRED_COLUMNS = {
    "cutsky": {
        "RA",
        "DEC",
        "Z",
        "Z_COSMO",
        "FILE_NUM",
        "HALO_INDEX",
        "BOX_INDEX",
        "R_MAG_APP",
        "IN_Y1",
        "IN_Y5",
    },
    "forfa": {"TARGETID", "RA", "DEC", "TRUEZ", "RSDZ", "BGS_TARGET"},
    "pota": {
        "TARGETID",
        "RA",
        "DEC",
        "TRUEZ",
        "RSDZ",
        "BGS_TARGET",
        "COLLISION",
    },
    "fiberassign": {"TARGETID", "RA", "DEC", "BGS_TARGET"},
}


class RegistryError(ValueError):
    """The frozen phase contract is internally inconsistent."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    with path.open() as stream:
        registry = json.load(stream)
    validate_registry(registry)
    return registry


def validate_registry(registry: dict[str, Any]) -> None:
    if registry.get("schema_version") != "p10-phase-registry-v1":
        raise RegistryError("unexpected registry schema")
    phases = registry.get("phases", {})
    if tuple(sorted(phases)) != EXPECTED_PHASES:
        raise RegistryError(
            f"phase registry must contain exactly {EXPECTED_PHASES}, got {tuple(sorted(phases))}"
        )
    target = registry["target_contract"]
    samples = target["particle_subsamples"]
    if abs(samples["A_fraction"] + samples["B_fraction"] - samples["total_fraction"]) > 1e-12:
        raise RegistryError("A and B fractions do not sum to the registered total")
    if samples["total_fraction"] != 0.1:
        raise RegistryError("P10 requires the frozen 10 percent particle contract")
    if target["redshift"] != 0.2 or target["grid_size"] != 2048:
        raise RegistryError("target epoch or grid differs from the frozen contract")
    if target.get("phase_is_model_input") is not False:
        raise RegistryError("phase identity must remain provenance-only")
    roles = {phase: cfg["role"] for phase, cfg in phases.items()}
    if roles["ph001"] != "sealed_blind":
        raise RegistryError("ph001 must remain sealed blind")
    if roles["ph006"] != "validation_and_selection":
        raise RegistryError("ph006 must remain validation/selection")
    for phase, cfg in phases.items():
        source = cfg["particle_b"]
        kind = source.get("kind")
        if phase == "ph006" and kind != "cfs":
            raise RegistryError("ph006 B must use the verified online CFS source")
        if phase != "ph006" and kind != "hpss":
            raise RegistryError(f"{phase} B must use its registered HPSS source")
        if kind == "hpss":
            members = source.get("members", [])
            if members != [
                "./halos/z0.200/field_rv_B",
                "./halos/z0.200/halo_rv_B",
            ]:
                raise RegistryError(f"{phase} has unexpected HPSS members")


def expand_phase(registry: dict[str, Any], phase: str) -> dict[str, Any]:
    if phase not in registry["phases"]:
        raise RegistryError(f"unregistered phase: {phase}")
    cfg = registry["phases"][phase]
    values = {"phase": phase, "mock": cfg["mock"]}
    assets = {
        name: template.format(**values)
        for name, template in registry["path_templates"].items()
    }
    snapshot = Path(assets["snapshot_root"])
    assets.update(
        {
            "field_rv_A": str(snapshot / "field_rv_A"),
            "halo_rv_A": str(snapshot / "halo_rv_A"),
            "halo_info": str(snapshot / "halo_info"),
        }
    )
    return {
        "phase": phase,
        "mock": cfg["mock"],
        "role": cfg["role"],
        "truth_access": cfg["truth_access"],
        "particle_b": cfg["particle_b"],
        "assets": assets,
    }


def path_record(path: Path, *, require_nonempty_dir: bool = True) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "kind": None,
        "size_bytes": None,
        "entries": None,
        "ready": False,
    }
    if not path.exists():
        return record
    stat = path.stat()
    record["mtime_ns"] = stat.st_mtime_ns
    if path.is_file():
        record.update(kind="file", size_bytes=stat.st_size, ready=stat.st_size > 0)
    elif path.is_dir():
        entries = sum(1 for _ in path.iterdir())
        record.update(
            kind="directory",
            entries=entries,
            ready=(entries > 0 or not require_nonempty_dir),
        )
    return record


def inspect_fits(path: Path, required: set[str]) -> dict[str, Any]:
    record = path_record(path)
    record.update(rows=None, columns=None, required_columns=sorted(required), missing_columns=[])
    if not record["ready"]:
        return record
    try:
        import fitsio

        with fitsio.FITS(path) as fits:
            hdu = fits[1]
            columns = list(hdu.get_colnames())
            record["rows"] = int(hdu.get_nrows())
            record["columns"] = columns
            record["missing_columns"] = sorted(required.difference(columns))
            record["ready"] = record["rows"] > 0 and not record["missing_columns"]
    except Exception as exc:  # pragma: no cover - exercised on damaged external files
        record["ready"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def _classify_b_name(name: str) -> str | None:
    normalized = name.lstrip("./")
    if "/field_rv_B/" in f"/{normalized}" and ASDF_RE.search(normalized):
        return "field"
    if "/halo_rv_B/" in f"/{normalized}" and ASDF_RE.search(normalized):
        return "halo"
    return None


def parse_htar_listing(text: str) -> list[dict[str, Any]]:
    """Parse verbose or name-only HTAR listings without depending on locale."""
    records: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("HTAR:"):
            line = line.removeprefix("HTAR:").strip()
        if line.startswith("HTAPE:"):
            line = line.removeprefix("HTAPE:").strip()
        if not line or line.startswith(("Listing complete", "HTAR SUCCESSFUL")):
            continue
        parts = line.split()
        name = parts[-1]
        if "/" not in name:
            continue
        size = None
        for token in reversed(parts[:-1]):
            if token.isdigit():
                size = int(token)
                break
        records.append({"name": name, "size_bytes": size})
    return records


def summarize_b_listing(
    records: Iterable[dict[str, Any]], expected_field: int, expected_halo: int
) -> dict[str, Any]:
    records = list(records)
    field = [r for r in records if _classify_b_name(r["name"]) == "field"]
    halo = [r for r in records if _classify_b_name(r["name"]) == "halo"]
    checksums = [
        r for r in records if Path(r["name"]).name == "checksums.crc32"
    ]
    sizes_known = all(r["size_bytes"] is not None for r in field + halo)
    payload = sum(int(r["size_bytes"]) for r in field + halo) if sizes_known else None
    ready = (
        len(field) == expected_field
        and len(halo) == expected_halo
        and len(checksums) == 2
    )
    return {
        "field_asdf_count": len(field),
        "halo_asdf_count": len(halo),
        "checksum_manifest_count": len(checksums),
        "payload_bytes": payload,
        "listing_entry_count": len(records),
        "ready": ready,
        "errors": []
        if ready
        else [
            "expected "
            f"{expected_field}/{expected_halo}/2 field/halo/checksum entries, got "
            f"{len(field)}/{len(halo)}/{len(checksums)}"
        ],
    }


def inspect_hpss_b(
    source: dict[str, Any],
    staging: dict[str, Any],
    *,
    htar: str = "htar",
) -> dict[str, Any]:
    command = [htar, "-tvf", source["archive"], *source["members"]]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    record: dict[str, Any] = {
        "kind": "hpss",
        "archive": source["archive"],
        "members": source["members"],
        "command": command,
        "returncode": result.returncode,
        "stderr": result.stderr[-4000:],
    }
    if result.returncode != 0:
        record.update(ready=False, errors=["HTAR listing failed"])
        return record
    records = parse_htar_listing(result.stdout)
    record.update(
        summarize_b_listing(
            records,
            staging["expected_field_b_slabs"],
            staging["expected_halo_b_slabs"],
        )
    )
    return record


def inspect_local_b(root: Path, staging: dict[str, Any]) -> dict[str, Any]:
    field_dir = root / "field_rv_B"
    halo_dir = root / "halo_rv_B"
    field = sorted(field_dir.glob("field_rv_B_*.asdf")) if field_dir.is_dir() else []
    halo = sorted(halo_dir.glob("halo_rv_B_*.asdf")) if halo_dir.is_dir() else []
    manifests = [
        field_dir / staging["checksum_filename"],
        halo_dir / staging["checksum_filename"],
    ]
    ready = (
        len(field) == staging["expected_field_b_slabs"]
        and len(halo) == staging["expected_halo_b_slabs"]
        and all(path.is_file() and path.stat().st_size > 0 for path in manifests)
    )
    return {
        "kind": "cfs",
        "root": str(root),
        "field_asdf_count": len(field),
        "halo_asdf_count": len(halo),
        "checksum_manifests": [str(path) for path in manifests],
        "checksum_manifest_count": sum(path.is_file() for path in manifests),
        "payload_bytes": sum(path.stat().st_size for path in field + halo),
        "ready": ready,
        "errors": []
        if ready
        else [
            "local B source does not satisfy registered slab/checksum counts"
        ],
    }


def audit_phase(
    registry: dict[str, Any],
    phase: str,
    *,
    check_hpss: bool,
    inspect_fits_schema: bool,
) -> dict[str, Any]:
    expanded = expand_phase(registry, phase)
    assets: dict[str, Any] = {}
    for name, value in expanded["assets"].items():
        if name == "phase_output":
            continue
        path = Path(value)
        if inspect_fits_schema and name in FITS_REQUIRED_COLUMNS:
            assets[name] = inspect_fits(path, FITS_REQUIRED_COLUMNS[name])
        else:
            assets[name] = path_record(path)

    source = expanded["particle_b"]
    if source["kind"] == "hpss":
        particle_b = (
            inspect_hpss_b(source, registry["staging_contract"])
            if check_hpss
            else {
                "kind": "hpss",
                "archive": source["archive"],
                "members": source["members"],
                "ready": None,
                "not_checked": True,
            }
        )
    else:
        particle_b = inspect_local_b(Path(source["root"]), registry["staging_contract"])

    required_assets = [
        "cutsky",
        "field_rv_A",
        "halo_rv_A",
        "halo_info",
        *registry["required_observation_assets"][1:],
    ]
    source_ready = all(assets[name]["ready"] for name in required_assets)
    b_ready = particle_b["ready"] is True
    return {
        "phase": phase,
        "role": expanded["role"],
        "truth_access": expanded["truth_access"],
        "mock": expanded["mock"],
        "assets": assets,
        "particle_b": particle_b,
        "source_ready": source_ready and b_ready,
        "source_assets_ready": source_ready,
        "particle_b_ready": b_ready,
        "output_root": expanded["assets"]["phase_output"],
    }


def parse_phases(raw: list[str] | None) -> list[str]:
    if not raw:
        return list(EXPECTED_PHASES)
    phases = []
    for item in raw:
        phases.extend(part.strip() for part in item.split(",") if part.strip())
    invalid = sorted(set(phases).difference(EXPECTED_PHASES))
    if invalid:
        raise RegistryError(f"invalid phase(s): {invalid}")
    return phases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--phases", nargs="*", help="phase keys or comma-separated keys")
    parser.add_argument("--check-hpss", action="store_true")
    parser.add_argument("--inspect-fits", action="store_true")
    parser.add_argument("--out", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    registry = load_registry(args.registry)
    phases = parse_phases(args.phases)
    report = {
        "schema_version": "p10-phase-asset-inventory-v1",
        "created_utc": utc_now(),
        "registry": str(args.registry.resolve()),
        "registry_sha256": sha256_file(args.registry),
        "check_hpss": args.check_hpss,
        "inspect_fits": args.inspect_fits,
        "target_contract": registry["target_contract"],
        "phases": {
            phase: audit_phase(
                registry,
                phase,
                check_hpss=args.check_hpss,
                inspect_fits_schema=args.inspect_fits,
            )
            for phase in phases
        },
    }
    report["gate"] = {
        "all_sources_ready": all(
            record["source_ready"] for record in report["phases"].values()
        ),
        "phase_count": len(phases),
        "ready_phase_count": sum(
            record["source_ready"] for record in report["phases"].values()
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_suffix(args.out.suffix + ".tmp")
        temporary.write_text(rendered)
        temporary.replace(args.out)
    print(rendered, end="")
    return 0 if report["gate"]["all_sources_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
