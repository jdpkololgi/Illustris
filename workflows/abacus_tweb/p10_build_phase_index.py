#!/usr/bin/env python3
"""Build the phase-generic P10/P1 canonical index and Cartesian point contract.

The successful-redshift observed-truth FITS table remains the immutable canonical
row table.  This stage adds only representation-neutral indexing metadata:
identity row/node IDs, Galactic-cap labels, reporting shells, active/context masks,
and observer-frame Planck18 Cartesian positions in Mpc.  P2 graph construction and
P3 field construction must consume these exact points and masks.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import fitsio
import numpy as np
from astropy.cosmology import Planck18


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = Path(__file__).resolve().parent
for import_root in (REPO_ROOT, WORKFLOW_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from p10_phase_assets import DEFAULT_REGISTRY, load_registry, sha256_file  # noqa: E402


SHELLS = ((0.15, 0.25), (0.25, 0.35), (0.35, 0.45), (0.45, 0.55))
CONTEXT_RANGE = (0.10, 0.60)
SENTINEL = (0.585, 0.595)

# ICRS -> Galactic rotation (IAU 1958/J2000 realization used by Astropy).
ICRS_TO_GAL = np.asarray(
    [
        [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
        [0.4941094278755837, -0.4448296299600112, 0.7469822444972189],
        [-0.8676661490190047, -0.1980763734312015, 0.4559837761750669],
    ],
    dtype=np.float64,
)


class PhaseIndexError(RuntimeError):
    """The phase catalogue cannot be promoted to the P1 canonical contract."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def default_observed(registry: dict[str, Any], phase: str) -> Path:
    root = Path(registry["path_templates"]["phase_output"].format(phase=phase))
    return root / "catalogues/observed" / f"{phase}_bgs_bright_full_observed_with_tweb.fits"


def default_blind_observed(registry: dict[str, Any], phase: str) -> Path:
    root = Path(registry["path_templates"]["phase_output"].format(phase=phase))
    return root / "catalogues/blind_observed" / f"{phase}_bgs_bright_full_observed_geometry.fits"


def default_output_dir(registry: dict[str, Any], phase: str) -> Path:
    root = Path(registry["path_templates"]["phase_output"].format(phase=phase))
    return root / "p1_canonical"


def comoving_distance_mpc(z: np.ndarray) -> np.ndarray:
    grid_z = np.linspace(0.0, 0.85, 17001, dtype=np.float64)
    grid_r = Planck18.comoving_distance(grid_z).value.astype(np.float64)
    return np.interp(np.asarray(z, dtype=np.float64), grid_z, grid_r)


def cartesian_points(ra_deg: np.ndarray, dec_deg: np.ndarray,
                     redshift: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
    radius = comoving_distance_mpc(redshift)
    cos_dec = np.cos(dec)
    unit = np.column_stack((cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)))
    gal_z = unit @ ICRS_TO_GAL[2]
    points = np.empty((len(unit), 4), dtype=np.float64)
    points[:, :3] = radius[:, None] * unit
    points[:, 3] = gal_z > 0.0
    return points


def shell_and_masks(redshift: np.ndarray, valid_target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.asarray(redshift, dtype=np.float64)
    shell = np.full(len(z), -1, dtype=np.int8)
    for shell_id, (lo, hi) in enumerate(SHELLS):
        shell[(z >= lo) & (z < hi)] = shell_id
    sentinel = (z >= SENTINEL[0]) & (z < SENTINEL[1])
    context = (z >= CONTEXT_RANGE[0]) & (z < CONTEXT_RANGE[1]) & ~sentinel
    active = np.asarray(valid_target, dtype=bool) & (shell >= 0)
    return shell, active, context


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    os.replace(temporary, path)


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--observed", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--blind-geometry-only", action="store_true",
        help=("Build a sealed-phase input/index contract without reading tidal labels. "
              "This is accepted only for the registered sealed-blind phase."),
    )
    parser.add_argument(
        "--reuse-validated", action="store_true",
        help="Validate an existing complete P1 product and add the compatibility manifest.",
    )
    return parser


def write_compatibility_manifest(
    *, marker: Path, manifest: Path, index_path: Path, points_path: Path,
) -> dict[str, Any]:
    payload = json.loads(marker.read_text())
    index = np.load(index_path)
    points = np.load(points_path, mmap_mode="r")
    n = int(payload["counts"]["total"])
    if points.shape != (n, 4) or len(index["parent_node_id"]) != n:
        raise PhaseIndexError("existing P1 points/index rows do not match completion marker")
    if not np.array_equal(index["parent_node_id"], np.arange(n, dtype=np.int64)):
        raise PhaseIndexError("existing P1 parent-node IDs are not row identities")
    if sha256_file(index_path) != payload["artifacts"]["canonical_index_sha256"]:
        raise PhaseIndexError("existing P1 canonical-index hash differs from marker")
    if sha256_file(points_path) != payload["artifacts"]["points_sha256"]:
        raise PhaseIndexError("existing P1 points hash differs from marker")
    catalogue_id = payload.get("catalogue_id", f"{payload['phase']}_bgs_bright_full_ngc_sgc_v1")
    legacy = {
        "schema_version": "1.1",
        "stage": "P10/P1 phase-generic canonical catalogue",
        "catalogue_id": catalogue_id,
        "phase": payload["phase"],
        "role": payload["role"],
        "parent": payload["canonical_parent"]["path"],
        "parent_sha256": payload["canonical_parent"]["sha256"],
        "parent_rows_are_canonical_rows": True,
        "points": str(points_path.resolve()),
        "index": str(index_path.resolve()),
        "index_sha256": sha256_file(index_path),
        "counts": payload["counts"],
        "mapping_contract": {
            "galaxy_id": "TARGETID",
            "graph_node_id": "PARENT_NODE_ID == parent FITS row == full graph row",
            "halo_group": ["FILE_NUM", "BOX_INDEX", "HALO_INDEX"],
            "p4_rule": "no repeated TARGETID or underlying halo group may cross supervised folds",
        },
        "scope": {
            "footprint": "full successful-redshift BGS BRIGHT NGC+SGC",
            "components": {"0": "SGC", "1": "NGC"},
            "z_context": list(CONTEXT_RANGE), "z_core": [0.15, 0.55],
            "sentinel_excluded_from_context": list(SENTINEL),
        },
        "target_truth_present": bool(payload.get("target_truth_present", True)),
        "blind_contract": payload.get("blind_contract"),
        "target_convention": payload.get("target_contract"),
        "no_train_fitted_normalisation": True,
        "no_split_filtering": True,
        "source_completion_marker": str(marker.resolve()),
        "source_completion_marker_sha256": sha256_file(marker),
    }
    atomic_json(manifest, legacy)
    return legacy


def main() -> int:
    args = build_parser().parse_args()
    registry = load_registry(args.registry)
    if args.phase not in registry["phases"]:
        raise PhaseIndexError(f"unregistered phase: {args.phase}")
    cfg = registry["phases"][args.phase]
    is_blind = cfg["role"] == "sealed_blind"
    if is_blind != bool(args.blind_geometry_only):
        raise PhaseIndexError(
            "sealed ph001 requires --blind-geometry-only; development phases forbid it"
        )
    observed = args.observed or (
        default_blind_observed(registry, args.phase) if is_blind
        else default_observed(registry, args.phase)
    )
    out_dir = args.out_dir or default_output_dir(registry, args.phase)
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "CATALOGUE_COMPLETE.json"
    manifest = out_dir / "manifest.json"
    index_path = out_dir / "canonical_index.npz"
    points_path = out_dir / "points.npy"
    if args.reuse_validated:
        if not (marker.is_file() and index_path.is_file() and points_path.is_file()):
            raise PhaseIndexError("--reuse-validated requires all existing P1 artifacts")
        legacy = write_compatibility_manifest(
            marker=marker, manifest=manifest, index_path=index_path, points_path=points_path,
        )
        print(json.dumps(legacy, indent=2, sort_keys=True))
        return 0
    if marker.exists() or manifest.exists() or index_path.exists() or points_path.exists():
        raise PhaseIndexError(f"refusing to overwrite P1 artifacts in {out_dir}")

    required = ["TARGETID", "RA", "DEC", "Z", "BOX_INDEX"]
    if not is_blind:
        required += ["LAMBDA1", "LAMBDA2", "LAMBDA3", "CWEB"]
    table = fitsio.read(observed, columns=required)
    n = len(table)
    if not n:
        raise PhaseIndexError("observed catalogue is empty")
    targetid = np.asarray(table["TARGETID"], dtype=np.int64)
    if len(np.unique(targetid)) != n:
        raise PhaseIndexError("TARGETID is not unique")
    box_valid = np.asarray(table["BOX_INDEX"]) >= 0
    valid_target = box_valid.copy()
    if not is_blind:
        eigenvalues = np.column_stack(
            (table["LAMBDA1"], table["LAMBDA2"], table["LAMBDA3"])
        ).astype(np.float64)
        finite = np.isfinite(eigenvalues).all(axis=1)
        ordered = ((eigenvalues[:, 0] <= eigenvalues[:, 1])
                   & (eigenvalues[:, 1] <= eigenvalues[:, 2]))
        class_expected = np.sum(eigenvalues > 0.2, axis=1).astype(np.int8)
        valid_target &= finite & ordered
        if not np.all(finite & ordered):
            raise PhaseIndexError(
                f"{int((~(finite & ordered)).sum())} observed rows lack finite ordered truth"
            )
        if not np.array_equal(class_expected, np.asarray(table["CWEB"], dtype=np.int8)):
            raise PhaseIndexError("CWEB disagrees with thresholded eigenvalues")

    z = np.asarray(table["Z"], dtype=np.float64)
    points = cartesian_points(table["RA"], table["DEC"], z)
    if not np.isfinite(points).all() or np.any(np.linalg.norm(points[:, :3], axis=1) <= 0):
        raise PhaseIndexError("non-finite or non-positive Cartesian geometry")
    cap = np.asarray(points[:, 3], dtype=np.uint8)
    shell, active, context = shell_and_masks(z, valid_target)
    if not active.any() or not (active & (cap == 0)).any() or not (active & (cap == 1)).any():
        raise PhaseIndexError("active catalogue does not cover both Galactic caps")

    parent_node_id = np.arange(n, dtype=np.int64)
    atomic_save_npy(points_path, points)
    atomic_savez(
        index_path,
        parent_node_id=parent_node_id,
        targetid=targetid,
        cap=cap,
        shell=shell,
        active=active,
        context=context,
        valid_target=valid_target,
    )
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    by_shell = {
        f"{lo:.2f}_{hi:.2f}": {
            "all": int(np.count_nonzero(active & (shell == shell_id))),
            "NGC": int(np.count_nonzero(active & (shell == shell_id) & (cap == 1))),
            "SGC": int(np.count_nonzero(active & (shell == shell_id) & (cap == 0))),
        }
        for shell_id, (lo, hi) in enumerate(SHELLS)
    }
    payload = {
        "schema_version": "p10-p1-canonical-v1",
        "created_utc": utc_now(),
        "phase": args.phase,
        "catalogue_id": f"{args.phase}_bgs_bright_full_ngc_sgc_v1",
        "role": cfg["role"],
        "git_sha": git_sha,
        "registry": str(args.registry.resolve()),
        "registry_sha256": sha256_file(args.registry),
        "canonical_parent": {
            "path": str(observed.resolve()), "rows": n,
            "bytes": observed.stat().st_size, "sha256": sha256_file(observed),
            "row_contract": "PARENT_NODE_ID == observed FITS row == future graph row",
        },
        "artifacts": {
            "canonical_index": str(index_path.resolve()),
            "canonical_index_sha256": sha256_file(index_path),
            "points": str(points_path.resolve()),
            "points_sha256": sha256_file(points_path),
        },
        "geometry": {
            "coordinates": "observer-frame ICRS Cartesian from RA, DEC and observed Z",
            "cosmology": "Astropy Planck18",
            "units": "Mpc",
            "cap": {"0": "SGC (Galactic b<=0)", "1": "NGC (Galactic b>0)"},
        },
        "selection": {
            "successful_redshift_selection_applied_upstream": True,
            "active_z": [0.15, 0.55],
            "context_z": [0.10, 0.60],
            "sentinel_excluded_from_context": list(SENTINEL),
        },
        "counts": {
            "total": n,
            "NGC": int(np.count_nonzero(cap == 1)),
            "SGC": int(np.count_nonzero(cap == 0)),
            "active": int(np.count_nonzero(active)),
            "context": int(np.count_nonzero(context)),
            "valid_target": int(np.count_nonzero(valid_target)),
            "context_only_invalid_box_index": int(np.count_nonzero(~box_valid)),
            "by_shell": by_shell,
        },
        "box_index_policy": (
            "BOX_INDEX<0 rows retain observed geometry as context but never become "
            "authoritative supervised/evaluation rows, matching the frozen ph000 P1b contract"
        ),
        "target_contract": registry["target_contract"],
        "target_truth_present": not is_blind,
        "blind_contract": ({
            "sealed": True,
            "truth_columns_read": [],
            "valid_target_is_geometry_linkage_only": True,
            "unsealing_required_before_scored_evaluation": True,
        } if is_blind else None),
        "no_train_fitted_normalisation": True,
        "no_split_filtering": True,
        "pass": True,
    }
    atomic_json(marker, payload)
    write_compatibility_manifest(
        marker=marker, manifest=manifest, index_path=index_path, points_path=points_path,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PhaseIndexError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
