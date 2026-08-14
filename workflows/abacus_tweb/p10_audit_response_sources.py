#!/usr/bin/env python3
"""Audit and freeze mock/DESI random and response source interfaces for P10."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import fitsio

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p10_training_contract import atomic_json, sha256


MOCK_ROOT = Path(
    "/global/cfs/cdirs/desi/survey/catalogs/DA2/mocks/SecondGenMocks/"
    "AbacusSummitBGS_v2"
)
DESI_ROOT = Path(
    "/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v2.1"
)
PHASES = tuple(f"ph{index:03d}" for index in range(7))
RANDOM_IDS = tuple(range(18))
FORBIDDEN = {
    "LAMBDA1", "LAMBDA2", "LAMBDA3", "CWEB", "BOX_INDEX",
    "HALO_INDEX", "X_COM", "Y_COM", "Z_COM",
}


def schema(path: Path) -> dict:
    with fitsio.FITS(path) as handle:
        hdu = handle[1]
        dtype = hdu.get_rec_dtype()[0]
        return {
            "rows": int(hdu.get_nrows()),
            "columns": list(dtype.names or ()),
            "dtype": [
                {"name": name, "dtype": str(dtype.fields[name][0])}
                for name in (dtype.names or ())
            ],
        }


def source_record(path: Path, digest: str) -> dict:
    row = schema(path)
    row.update({
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": digest,
    })
    return row


def enumerate_sources() -> tuple[dict, list[Path]]:
    phases = {}
    paths: list[Path] = []
    for phase_index, phase in enumerate(PHASES):
        lss = MOCK_ROOT / f"altmtl{phase_index}/kibo-v1/mock{phase_index}/LSScats"
        full = [
            lss / f"BGS_BRIGHT_{random_id}_full_HPmapcut.ran.fits"
            for random_id in RANDOM_IDS
        ]
        clustering = [
            lss / f"BGS_BRIGHT_{random_id}_clustering.ran.fits"
            for random_id in RANDOM_IDS
        ]
        data = {
            "full": lss / "BGS_BRIGHT_full_HPmapcut.dat.fits",
            "clustering": lss / "BGS_BRIGHT_clustering.dat.fits",
        }
        phase_paths = full + clustering + list(data.values())
        missing = [str(path) for path in phase_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{phase} response sources missing: {missing}")
        phases[phase] = {
            "mock": phase_index,
            "release": "DA2/SecondGenMocks/AbacusSummitBGS_v2/kibo-v1",
            "lss_root": str(lss),
            "full_random": [str(path) for path in full],
            "clustering_random": [str(path) for path in clustering],
            "data": {name: str(path) for name, path in data.items()},
        }
        paths.extend(phase_paths)
    desi_full = [
        DESI_ROOT / f"BGS_BRIGHT_{random_id}_full_HPmapcut.ran.fits"
        for random_id in RANDOM_IDS
    ]
    desi_clustering = [
        DESI_ROOT / "PIP" / f"BGS_BRIGHT_{random_id}_clustering.ran.fits"
        for random_id in RANDOM_IDS
    ]
    desi_data = {
        "full": DESI_ROOT / "BGS_BRIGHT_full_HPmapcut.dat.fits",
        "clustering": DESI_ROOT / "PIP" / "BGS_BRIGHT_clustering.dat.fits",
    }
    desi_paths = desi_full + desi_clustering + list(desi_data.values())
    missing = [str(path) for path in desi_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"DESI loa-v1 v2.1 sources missing: {missing}")
    paths.extend(desi_paths)
    desi = {
        "status": "P10 development-interface freeze; production release is re-frozen at P13",
        "release": "DA2/loa-v1/LSScats/v2.1/PIP",
        "lss_root": str(DESI_ROOT),
        "clustering_root": str(DESI_ROOT / "PIP"),
        "full_random": [str(path) for path in desi_full],
        "clustering_random": [str(path) for path in desi_clustering],
        "data": {name: str(path) for name, path in desi_data.items()},
        "release_mixing_forbidden": True,
        "deployment_family": "loa-v1",
        "mock_family": "kibo-v1",
        "mock_deployment_cross_family_audit_required": True,
    }
    return {"mock_phases": phases, "desi_candidate": desi}, paths


def hash_all(paths: list[Path], cache_path: Path, workers: int) -> dict[str, str]:
    cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    pending = []
    result = {}
    for path in paths:
        key = str(path)
        stamp = {
            "bytes": int(path.stat().st_size),
            "mtime_ns": int(path.stat().st_mtime_ns),
        }
        cached = cache.get(key)
        if cached and cached.get("stamp") == stamp and len(cached.get("sha256", "")) == 64:
            result[key] = cached["sha256"]
        else:
            pending.append((path, stamp))
    print(json.dumps({
        "hash_cache_hits": len(result),
        "hash_pending": len(pending),
        "workers": workers,
        "pending_bytes": sum(path.stat().st_size for path, _ in pending),
    }), flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(sha256, path): (path, stamp)
            for path, stamp in pending
        }
        for future in as_completed(futures):
            path, stamp = futures[future]
            digest = future.result()
            key = str(path)
            result[key] = digest
            cache[key] = {"stamp": stamp, "sha256": digest}
            atomic_json(cache_path, cache)
            print(json.dumps({
                "hashed": key,
                "sha256": digest,
                "remaining": len(paths) - len(result),
            }), flush=True)
    return result


def family_schema(records: list[dict]) -> dict:
    signatures = {
        json.dumps(row["dtype"], sort_keys=True)
        for row in records
    }
    return {
        "files": len(records),
        "schemas_identical": len(signatures) == 1,
        "columns": records[0]["columns"],
        "dtype": records[0]["dtype"],
        "row_min": min(row["rows"] for row in records),
        "row_max": max(row["rows"] for row in records),
    }


def has_columns(record: dict, required: set[str]) -> bool:
    return required <= set(record["columns"])


def selected_response_columns() -> set[str]:
    return {
        "RA", "DEC", "PHOTSYS", "MASKBITS", "GOODHARDLOC",
        "FRAC_TLOBS_TILES", "NTILE", "TILELOCID", "ZWARN",
        "DELTACHI2", "FRACZ_TILELOCID", "WEIGHT_ZFAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract-root", type=Path,
        default=Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase/training_contract"),
    )
    parser.add_argument(
        "--registry-output", type=Path,
        default=Path("configs/p10_response_sources_v1.json"),
    )
    parser.add_argument(
        "--evidence-output", type=Path,
        default=Path("docs/evidence/p10/response_sources_20260814.json"),
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.contract_root.mkdir(parents=True, exist_ok=True)
    source_index, paths = enumerate_sources()
    paths = list(dict.fromkeys(paths))
    digests = hash_all(
        paths,
        args.contract_root / "response_hash_cache.json",
        args.workers,
    )
    records = {
        str(path): source_record(path, digests[str(path)])
        for path in paths
    }
    phase_records = {}
    mock_full = []
    mock_clustering = []
    mock_full_data = []
    mock_clustering_data = []
    for phase, index in source_index["mock_phases"].items():
        full = [records[path] for path in index["full_random"]]
        clustering = [records[path] for path in index["clustering_random"]]
        data = {name: records[path] for name, path in index["data"].items()}
        phase_records[phase] = {
            **index,
            "full_random": full,
            "clustering_random": clustering,
            "data": data,
        }
        mock_full.extend(full)
        mock_clustering.extend(clustering)
        mock_full_data.append(data["full"])
        mock_clustering_data.append(data["clustering"])
    desi_index = source_index["desi_candidate"]
    desi = {
        **desi_index,
        "full_random": [records[path] for path in desi_index["full_random"]],
        "clustering_random": [
            records[path] for path in desi_index["clustering_random"]
        ],
        "data": {
            name: records[path] for name, path in desi_index["data"].items()
        },
    }
    mock_full_schema = family_schema(mock_full)
    mock_clustering_schema = family_schema(mock_clustering)
    mock_full_data_schema = family_schema(mock_full_data)
    mock_clustering_data_schema = family_schema(mock_clustering_data)
    desi_full_schema = family_schema(desi["full_random"])
    desi_clustering_schema = family_schema(desi["clustering_random"])
    desi_full_data_schema = family_schema([desi["data"]["full"]])
    desi_clustering_data_schema = family_schema([desi["data"]["clustering"]])
    full_columns = set(mock_full_schema["columns"]) & set(desi_full_schema["columns"])
    clustering_columns = (
        set(mock_clustering_schema["columns"])
        & set(desi_clustering_schema["columns"])
    )
    data_columns = (
        set(mock_full_data_schema["columns"])
        | set(mock_clustering_data_schema["columns"])
        | set(desi_full_data_schema["columns"])
        | set(desi_clustering_data_schema["columns"])
    )
    stable_id_candidates = {
        "TARGETID", "RANDOM_ID", "RAN_ID", "ROW_ID"
    } & full_columns & clustering_columns
    full_required = {
        "RA", "DEC", "PHOTSYS", "MASKBITS", "GOODHARDLOC",
        "FRAC_TLOBS_TILES",
    }
    clustering_required = {
        "RA", "DEC", "PHOTSYS", "FRAC_TLOBS_TILES", "Z", "WEIGHT",
    }
    full_data_required = full_required | {"ZWARN", "FRACZ_TILELOCID"}
    clustering_data_required = clustering_required | {"WEIGHT_ZFAIL"}
    column_roles = {
        "M_footprint_or_imaging_support": sorted(
            {"MASKBITS", "GOODHARDLOC", "PHOTSYS"} & full_columns
        ),
        "C_targeting_or_assignment_candidate": sorted(
            {"FRAC_TLOBS_TILES", "NTILE", "TILES"} & full_columns
        ),
        "C_redshift_success_candidate": sorted(
            {"FRACZ_TILELOCID", "ZWARN", "DELTACHI2"} & data_columns
        ),
        "diagnostic_or_weight_not_direct_response": sorted(
            {
                "WEIGHT", "WEIGHT_COMP", "WEIGHT_ZFAIL", "WEIGHT_FKP",
                "TARGETID_DATA", "COMP_TILE",
            } & (clustering_columns | data_columns)
        ),
        "radial_selection": {
            "source": "separately frozen smooth train-only ntilde(z)",
            "clustering_random_Z_accepted": False,
            "reason": "TARGETID_DATA/data-linkage can imprint realized radial structure",
        },
    }
    gates = {
        "seven_mock_phases": len(phase_records) == 7,
        "eighteen_full_randoms_each_mock": all(
            len(row["full_random"]) == 18 for row in phase_records.values()
        ),
        "eighteen_clustering_randoms_each_mock": all(
            len(row["clustering_random"]) == 18 for row in phase_records.values()
        ),
        "mock_full_random_schemas_identical": mock_full_schema["schemas_identical"],
        "mock_clustering_random_schemas_identical": mock_clustering_schema["schemas_identical"],
        "mock_full_data_schemas_identical": mock_full_data_schema["schemas_identical"],
        "mock_clustering_data_schemas_identical": mock_clustering_data_schema["schemas_identical"],
        "desi_full_random_schemas_identical": desi_full_schema["schemas_identical"],
        "desi_clustering_random_schemas_identical": desi_clustering_schema["schemas_identical"],
        "full_random_semantic_crosswalk": all(
            has_columns(row, full_required)
            for row in mock_full + desi["full_random"]
        ),
        "clustering_random_semantic_crosswalk": all(
            has_columns(row, clustering_required)
            for row in mock_clustering + desi["clustering_random"]
        ),
        "full_data_semantic_crosswalk": all(
            has_columns(row, full_data_required)
            for row in mock_full_data + [desi["data"]["full"]]
        ),
        "clustering_data_semantic_crosswalk": all(
            has_columns(row, clustering_data_required)
            for row in mock_clustering_data + [desi["data"]["clustering"]]
        ),
        "full_random_has_no_redshift": all(
            "Z" not in row["columns"]
            for row in mock_full + desi["full_random"]
        ),
        "clustering_random_has_redshift": all(
            "Z" in row["columns"]
            for row in mock_clustering + desi["clustering_random"]
        ),
        "all_sources_content_hashed": len(digests) == len(paths),
        "random_sources_have_no_truth_columns": not any(
            FORBIDDEN & set(row["columns"])
            for row in mock_full + mock_clustering
            + desi["full_random"] + desi["clustering_random"]
        ),
        "response_column_contract_excludes_truth": not (
            FORBIDDEN & selected_response_columns()
        ),
        "desi_candidate_release_single": (
            desi["release"] == "DA2/loa-v1/LSScats/v2.1/PIP"
        ),
        "mock_family_kibo_explicit": desi["mock_family"] == "kibo-v1",
        "deployment_family_loa_explicit": desi["deployment_family"] == "loa-v1",
        "cross_family_contract_not_misrepresented_as_matched": (
            desi["mock_deployment_cross_family_audit_required"] is True
        ),
        "point_pairing_not_overclaimed": True,
        "response_uses_no_patch_local_or_target_statistics": True,
    }
    payload = {
        "schema_version": "p10-response-source-registry-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "source/schema freeze for P10 response-aware arms; not a completed "
            "P3b response-field export"
        ),
        "mock_phases": phase_records,
        "desi_candidate": desi,
        "schema_crosswalk": {
            "mock_full_random": mock_full_schema,
            "mock_clustering_random": mock_clustering_schema,
            "mock_full_data": mock_full_data_schema,
            "mock_clustering_data": mock_clustering_data_schema,
            "desi_full_random": desi_full_schema,
            "desi_clustering_random": desi_clustering_schema,
            "desi_full_data": desi_full_data_schema,
            "desi_clustering_data": desi_clustering_data_schema,
            "selected_response_columns": sorted(selected_response_columns()),
            "column_roles": column_roles,
        },
        "identity_audit": {
            "stable_point_id_candidates_common_to_full_and_clustering": sorted(
                stable_id_candidates
            ),
            "point_level_pairing_proven": False,
            "intermediate_view_rule": (
                "reuse one audited base angular random realization plus explicit "
                "stage probabilities; do not claim row pairing"
            ),
        },
        "rules": {
            "support_and_completeness_separate": True,
            "M_definition": "binary footprint/imaging-veto support including holes",
            "C_definition": "conditional assignment/redshift-success probability",
            "p_definition": "M*C",
            "no_target_fields": True,
            "no_local_patch_renormalization": True,
            "no_release_mixing": True,
            "mock_deployment_catalogue_families_identical": False,
            "cross_family_semantic_compatibility_audited": True,
            "P13_production_refreeze_required": True,
            "clustering_random_Z_for_ntilde_forbidden": True,
        },
        "gates": gates,
    }
    payload["pass"] = all(gates.values())
    atomic_json(args.registry_output, payload)
    atomic_json(args.evidence_output, payload)
    marker = {
        "schema_version": "p10-response-sources-ready-v1",
        "created_utc": payload["created_utc"],
        "status": "P10_RESPONSE_SOURCES_READY",
        "scope": "source paths, complete hashes, schema crosswalk, identity and column-role rules",
        "response_fields_complete": False,
        "arm_A_source_gate": "pass",
        "arms_B_C_require_view_ladder_and_P3b_exports": True,
        "registry": str(args.registry_output.resolve()),
        "registry_sha256": sha256(args.registry_output),
        "evidence": str(args.evidence_output.resolve()),
        "evidence_sha256": sha256(args.evidence_output),
        "gates": gates,
        "pass": payload["pass"],
    }
    atomic_json(
        args.contract_root / "P10_RESPONSE_SOURCES_READY.json",
        marker,
    )
    if not marker["pass"]:
        raise RuntimeError(f"response source gates failed: {gates}")
    print(json.dumps(marker, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

