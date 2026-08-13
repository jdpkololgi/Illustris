#!/usr/bin/env python3
"""Normalize the frozen legacy ph000 development products under P10.

ph000 is an established development/reference data set, not a member of the
ph002--ph005 training pool.  This importer copies the immutable legacy products
into the phase-shaped P10 hierarchy, records source and destination hashes, and
writes explicit *reference import* markers.  It deliberately preserves the
original scientific manifests byte-for-byte because they are historical
evidence and contain their original absolute paths and hashes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase")
LEGACY = Path("/pscratch/sd/d/dkololgi/abacus")
DENSITY = Path(
    "/pscratch/sd/d/dkololgi/AbacusSummit_densities/density_fields/"
    "AbacusSummit_base_c000_ph000_z0.200_ngrid_2048_10pc_density_field.npy"
)
TWEB = Path(
    "/pscratch/sd/d/dkololgi/AbacusSummit_densities/tweb_rank_outputs_fullgrid_v3/"
    "dens_AbacusSummit_base_c000_ph000_z0.200_ngrid2048_box2000_thr0p2/"
    "backend_optimized_ngrid_2048_rsmooth_7"
)
OBSERVED = LEGACY / (
    "mocks_with_eigs_05062026_rsmooth_7/"
    "mock_bgs_maglim_path1_fiberassign_graph_ready_with_tweb_eigs_rs7_"
    "ngrid2048_thr0p2_halo_xcom.fits"
)
GRAPH_ROOT = LEGACY / "graph_constructions"
GRAPH_SOURCE_PREFIX = "path1_fiberassign_mock_bgs_maglim_rs7"
GRAPH_DEST_PREFIX = "ph000_bgs_bright_full_delaunay"
TARGET_CONTRACT = {
    "simulation_family": "AbacusSummit",
    "simulation_name_pattern": "AbacusSummit_base_c000_{phase}",
    "cosmology": "c000",
    "redshift": 0.2,
    "box_size_mpc_h": 2000.0,
    "grid_size": 2048,
    "mass_assignment": "TSC",
    "particle_subsamples": {
        "A_fraction": 0.03,
        "B_fraction": 0.07,
        "total_fraction": 0.1,
        "required_directories": ["field_rv_A", "halo_rv_A", "field_rv_B", "halo_rv_B"],
    },
    "tidal_smoothing_mpc_h": 7.0,
    "eigenvalue_order": "lambda1<=lambda2<=lambda3",
    "web_threshold": 0.2,
    "coordinate_units": "Mpc/h",
    "phase_is_model_input": False,
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def copy_verified(source: Path, destination: Path) -> dict:
    if not source.is_file() or source.stat().st_size <= 0:
        raise RuntimeError(f"missing or empty source: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(source)
    copied = False
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size or sha256(destination) != source_hash:
            raise RuntimeError(f"existing destination differs from source: {destination}")
    else:
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        if temporary.exists():
            temporary.unlink()
        shutil.copy2(source, temporary)
        if temporary.stat().st_size != source.stat().st_size or sha256(temporary) != source_hash:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"copy verification failed: {source} -> {destination}")
        os.replace(temporary, destination)
        copied = True
    return {
        "source": str(source.resolve()), "destination": str(destination.resolve()),
        "bytes": source.stat().st_size, "sha256": source_hash, "copied_now": copied,
    }


def copy_tree(source: Path, destination: Path) -> list[dict]:
    if not source.is_dir():
        raise RuntimeError(f"missing source directory: {source}")
    return [
        copy_verified(path, destination / path.relative_to(source))
        for path in sorted(source.rglob("*")) if path.is_file()
    ]


def verify_tree(source: Path, destination: Path) -> list[dict]:
    if not source.is_dir():
        raise RuntimeError(f"missing source directory: {source}")
    records = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        imported = destination / path.relative_to(source)
        if not imported.is_file() or imported.stat().st_size != path.stat().st_size:
            raise RuntimeError(f"missing/mismatched imported file: {imported}")
        source_hash = sha256(path)
        if sha256(imported) != source_hash:
            raise RuntimeError(f"imported checksum differs: {imported}")
        records.append({
            "source": str(path.resolve()), "destination": str(imported.resolve()),
            "bytes": path.stat().st_size, "sha256": source_hash, "copied_now": False,
        })
    return records


def graph_sources() -> Iterable[tuple[Path, Path]]:
    skip = {f"{GRAPH_SOURCE_PREFIX}_points.npy", f"{GRAPH_SOURCE_PREFIX}_points_xyz.npy"}
    for source in sorted(GRAPH_ROOT.glob(f"{GRAPH_SOURCE_PREFIX}_*")):
        if source.is_file() and source.name not in skip:
            suffix = source.name.removeprefix(GRAPH_SOURCE_PREFIX)
            yield source, Path("p2_graph") / f"{GRAPH_DEST_PREFIX}{suffix}"


def validate_tweb_rank(path: Path, rank: int, previous_end: int) -> tuple[int, dict]:
    # Do not materialize the large eig_vals/cweb members merely to inspect
    # their shapes.  The ph000 reference is sharded into 128 archives, so a
    # naive np.load(...)["eig_vals"] validation would re-read roughly 100 GiB
    # and briefly allocate an 800 MiB array for every rank.  Read only the NPY
    # headers for large members and deserialize the six scalar members.
    with zipfile.ZipFile(path) as archive:
        def scalar(name: str):
            return np.load(io.BytesIO(archive.read(f"{name}.npy")), allow_pickle=False).item()

        def shape(name: str) -> list[int]:
            with archive.open(f"{name}.npy") as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    array_shape, _, _ = np.lib.format.read_array_header_1_0(stream)
                elif version in {(2, 0), (3, 0)}:
                    array_shape, _, _ = np.lib.format._read_array_header(stream, version)
                else:
                    raise RuntimeError(f"unsupported NPY version {version} in {path}:{name}")
            return list(array_shape)

        start, end = int(scalar("x_start")), int(scalar("x_end"))
        ngrid = int(scalar("ngrid"))
        boxsize = float(scalar("boxsize"))
        threshold = float(scalar("threshold"))
        rsmooth = float(scalar("Rsmooth"))
        eig_shape = shape("eig_vals")
        cweb_shape = shape("cweb")
    expected = [end - start, 2048, 2048]
    gates = {
        "filename_rank": path.name == f"abacus_cactus_tweb_rank{rank:04d}.npz",
        "contiguous": start == previous_end,
        "ngrid": ngrid == 2048,
        "boxsize_mpc_h": np.isclose(boxsize, 2000.0),
        "threshold": np.isclose(threshold, 0.2),
        "rsmooth_mpc_h": np.isclose(rsmooth, 7.0),
        "eig_shape": eig_shape == [3, *expected],
        "cweb_shape": cweb_shape == expected,
    }
    if not all(gates.values()):
        raise RuntimeError(f"invalid legacy T-web rank {path}: {gates}")
    return end, {"rank": rank, "x_start": start, "x_end": end, "gates": gates}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    phase_root = args.root / "ph000"
    density_dest = phase_root / "targets/density/AbacusSummit_base_c000_ph000_z0.200_ngrid2048_ab10_tsc_counts.npy"
    tweb_dest = phase_root / "targets/tweb/backend_optimized_ngrid_2048_rsmooth_7"
    observed_dest = phase_root / "catalogues/observed/ph000_bgs_bright_full_observed_with_tweb.fits"

    mappings: list[tuple[Path, Path]] = [
        (DENSITY, density_dest),
        (OBSERVED, observed_dest),
        (GRAPH_ROOT / f"{GRAPH_SOURCE_PREFIX}_points.npy", phase_root / "p1_canonical/points.npy"),
        (LEGACY / "p1b_full_footprint/canonical_index.npz", phase_root / "p1_canonical/canonical_index.npz"),
        (LEGACY / "p1b_full_footprint/manifest.json", phase_root / "p1_canonical/manifest.json"),
        (LEGACY / "p1b_full_footprint/full_footprint_audit.json", phase_root / "p1_canonical/full_footprint_audit.json"),
    ]
    mappings.extend((source, phase_root / relative) for source, relative in graph_sources())

    records = []
    if args.verify_only:
        for source, destination in mappings:
            if not destination.is_file() or destination.stat().st_size != source.stat().st_size:
                raise RuntimeError(f"missing/mismatched imported file: {destination}")
            records.append(copy_verified(source, destination))
    else:
        records.extend(copy_verified(source, destination) for source, destination in mappings)
    tree_records = {}
    for name, source, destination in (
        ("p2_union", LEGACY / "p2b_full_footprint", phase_root / "p2_union"),
        ("p3_fields", LEGACY / "p3_full_footprint", phase_root / "p3_fields"),
        ("p4_patches", LEGACY / "p4_spatial_manifest", phase_root / "p4_patches"),
        ("p4_rebuild", LEGACY / "p4_spatial_manifest_determinism", phase_root / "p4_rebuild"),
        ("tweb", TWEB, tweb_dest),
    ):
        tree_records[name] = (
            verify_tree(source, destination) if args.verify_only
            else copy_tree(source, destination)
        )

    ranks = sorted(tweb_dest.glob("abacus_cactus_tweb_rank*.npz"))
    if len(ranks) != 128:
        raise RuntimeError(f"expected 128 legacy T-web ranks, found {len(ranks)}")
    previous_end, rank_records = 0, []
    for rank, path in enumerate(ranks):
        previous_end, rank_record = validate_tweb_rank(path, rank, previous_end)
        rank_records.append(rank_record)
    if previous_end != 2048:
        raise RuntimeError(f"T-web x coverage ends at {previous_end}, expected 2048")

    # Frozen phase schemas: only catalogue_id differs from the tracked templates.
    repo = Path(__file__).resolve().parents[2]
    schema_records = {}
    for key, base in (
        ("p3", repo / "docs/evidence/p3/p3_field_schema_v1.json"),
        ("p4", repo / "docs/evidence/p4/p4_spatial_schema_v1.json"),
    ):
        payload = json.loads(base.read_text())
        payload["catalogue_id"] = "ph000_path1_full_ngc_sgc_v1"
        destination = phase_root / f"contracts/{key}_schema_v1.json"
        atomic_json(destination, payload)
        schema_records[key] = {"base": str(base.resolve()), "output": str(destination.resolve())}
    atomic_json(phase_root / "contracts/SCHEMAS_COMPLETE.json", {
        "schema_version": "p10-ph000-reference-schemas-v1", "phase": "ph000",
        "catalogue_id": "ph000_path1_full_ngc_sgc_v1", "records": schema_records, "pass": True,
    })

    density_record = next(record for record in records if record["destination"] == str(density_dest.resolve()))
    density_manifest = {
        "schema_version": "p10-density-legacy-import-v1", "created_utc": utc_now(),
        "phase": "ph000", "role": "development_reference", "target_contract": TARGET_CONTRACT,
        "legacy_source": density_record, "build": {
            "output": str(density_dest.resolve()), "output_bytes": density_dest.stat().st_size,
            "ngrid": 2048, "boxsize_mpc_h": 2000.0, "dtype": "float32",
            "mass_assignment": "TSC", "particle_fraction": 0.1,
            "particle_count": None, "deposited_count": None,
            "processed_file_count": 136, "relative_count_error": None,
            "note": "legacy generator used all 34 slabs in each A/B field/halo directory; exact run count manifest predates P10",
        }, "verified": True,
    }
    atomic_json(density_dest.with_suffix(".manifest.json"), density_manifest)
    atomic_json(tweb_dest / "TWEB_COMPLETE.json", {
        "schema_version": "p10-tweb-legacy-import-v1", "created_utc": utc_now(),
        "phase": "ph000", "role": "development_reference", "target_contract": TARGET_CONTRACT,
        "legacy_source": str(TWEB.resolve()), "mpi_size": 128,
        "outputs": {"rank_count": 128, "x_coverage": [0, 2048],
                    "total_bytes": sum(path.stat().st_size for path in ranks),
                    "verified": True, "records": rank_records},
    })
    observed_record = next(record for record in records if record["destination"] == str(observed_dest.resolve()))
    atomic_json(Path(f"{observed_dest}.complete.json"), {
        "schema_version": "p10-ph000-observed-import-v1", "phase": "ph000",
        "role": "development_reference", "artifact": observed_record, "verified": True,
    })
    stage_markers = {
        phase_root / "p1_canonical/CATALOGUE_COMPLETE.json": "P1",
        phase_root / "p2_graph/GRAPH_COMPLETE.json": "P2_GRAPH",
        phase_root / "p2_graph/P2_COMPLETE.json": "P2",
    }
    for marker, stage in stage_markers.items():
        atomic_json(marker, {"schema_version": "p10-ph000-reference-stage-v1", "phase": "ph000",
                             "stage": stage, "legacy_import": True, "verified": True})
    marker = phase_root / "REFERENCE_PHASE_COMPLETE.json"
    atomic_json(marker, {
        "schema_version": "p10-ph000-reference-complete-v1", "created_utc": utc_now(),
        "phase": "ph000", "role": "development_reference",
        "training_pool_eligible": False, "selection_phase_eligible": False,
        "purpose": "frozen P8 development and P10 convention reference",
        "target_contract": TARGET_CONTRACT,
        "copied_file_count": len(records) + sum(map(len, tree_records.values())),
        "copied_bytes": sum(record["bytes"] for record in records)
                        + sum(record["bytes"] for group in tree_records.values() for record in group),
        "source_destination_records": records, "tree_records": tree_records,
        "tweb_rank_validation": {"rank_count": 128, "x_coverage": [0, previous_end]},
        "historical_manifests_preserved_byte_exact": True,
        "historical_manifest_paths_remain_legacy_absolute_paths": True,
        "pass": True,
    })
    print(marker.read_text())


if __name__ == "__main__":
    main()
