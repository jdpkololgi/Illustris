#!/usr/bin/env python3
"""Audit whether the current ph000 mock products can support BGS_FAINT context.

This is an evidence audit, not a mock builder.  It inventories the target classes
that survive each staged-mock product, checks the columns needed for a response-aware
multitracer reconstruction, and emits an explicit readiness decision.  Large FITS
tables are read in bounded chunks; TARGETID uniqueness is checked exactly.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import fitsio
import numpy as np


DEFAULT_STAGE3 = Path(
    "/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/stage_3/"
    "path1_fiberassign_20260604_083322"
)
DEFAULT_STAGES = {
    "forfa_targets": Path(
        "/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/stage_2/"
        "SecondGenMocks/AbacusSummitBGS_v2/forFA0.fits"
    ),
    "fiberassign_input": DEFAULT_STAGE3 / "inputs/targ.fits",
    "fiberassign_assigned": DEFAULT_STAGE3 / "fba0/datcomb_brightassignwdup.fits",
    "fiberassign_all_targets": DEFAULT_STAGE3 / "fba0/datcomb_brightwdup.fits",
    "spectroscopic_join": (
        DEFAULT_STAGE3 / "loa-v1/mock0/datcomb_bright_tarspecwdup_zdone.fits"
    ),
    "lss_bright_full": (
        DEFAULT_STAGE3 / "loa-v1/mock0/LSScats/BGS_BRIGHT_full_noveto.dat.fits"
    ),
    "lss_bright_injected": (
        DEFAULT_STAGE3 / "loa-v1/mock0/LSScats/BGS_BRIGHT_full_noveto_loa_spec.fits"
    ),
    "graphweb_bright_final": DEFAULT_STAGE3 / "mock_bgs_maglim_graph_ready.fits",
}

SHELL_EDGES = np.asarray([0.15, 0.25, 0.35, 0.45, 0.55], dtype=np.float64)
SHELL_NAMES = ("0p15_0p25", "0p25_0p35", "0p35_0p45", "0p45_0p55")
AUDIT_COLUMNS = (
    "TARGETID",
    "BGS_TARGET",
    "R_MAG_APP",
    "RA",
    "DEC",
    "Z",
    "Z_COSMO",
    "Z_NOT4CLUS",
    "ZWARN",
    "DELTACHI2",
    "SPECTYPE",
    "FILE_NUM",
    "BOX_INDEX",
    "HALO_INDEX",
    "LOCATION_ASSIGNED",
    "TILELOCID",
    "FRACZ_TILELOCID",
    "WEIGHT_COMP",
)


def target_masks() -> tuple[int, int, dict[str, int], str]:
    """Return unions of installed DESI Bright/Faint bits with a safe fallback."""
    try:
        from desitarget.targetmask import bgs_mask

        names = list(bgs_mask.names())
        selected = {
            name: int(bgs_mask[name])
            for name in names
            if name.startswith("BGS_BRIGHT") or name.startswith("BGS_FAINT")
        }
        bright = 0
        faint = 0
        for name, value in selected.items():
            if name.startswith("BGS_BRIGHT"):
                bright |= value
            elif name.startswith("BGS_FAINT"):
                faint |= value
        if bright and faint:
            return bright, faint, selected, "desitarget.targetmask.bgs_mask"
    except (ImportError, AttributeError, KeyError, TypeError):
        pass

    # The current upstream preparation script writes 2**1 for BGS_BRIGHT,
    # 2**0 for BGS_FAINT, and adds 2**3 to promoted faint targets.
    return 2, 1 | 8, {"BGS_BRIGHT": 2, "BGS_FAINT": 1, "BGS_FAINT_HIP": 8}, "script_fallback"


def classify_targets(
    values: np.ndarray, bright_mask: int, faint_mask: int
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.int64)
    return (values & bright_mask) != 0, (values & faint_mask) != 0


def shell_counts(z: np.ndarray, mask: np.ndarray) -> dict[str, int]:
    z = np.asarray(z, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(z)
    return {
        name: int(np.count_nonzero(mask & (z >= lo) & (z < hi)))
        for name, lo, hi in zip(SHELL_NAMES, SHELL_EDGES[:-1], SHELL_EDGES[1:])
    }


def summarize_chunk(
    tab: np.ndarray, bright_mask: int, faint_mask: int
) -> dict[str, object]:
    """Summarize one structured-array chunk; kept pure for unit testing."""
    names = set(tab.dtype.names or ())
    n = int(tab.size)
    if "BGS_TARGET" in names:
        bright, faint = classify_targets(tab["BGS_TARGET"], bright_mask, faint_mask)
    else:
        bright = np.zeros(n, dtype=bool)
        faint = np.zeros(n, dtype=bool)

    out: dict[str, object] = {
        "rows": n,
        "bright_rows": int(np.count_nonzero(bright)),
        "faint_rows": int(np.count_nonzero(faint)),
        "bright_and_faint_rows": int(np.count_nonzero(bright & faint)),
        "neither_rows": int(np.count_nonzero(~(bright | faint))),
    }

    if "R_MAG_APP" in names:
        mag = np.asarray(tab["R_MAG_APP"], dtype=np.float64)
        out["magnitude_rows"] = {
            "r_lt_19p5": int(np.count_nonzero(np.isfinite(mag) & (mag < 19.5))),
            "r_19p5_to_20p175": int(
                np.count_nonzero(np.isfinite(mag) & (mag >= 19.5) & (mag <= 20.175))
            ),
            "r_gt_20p175": int(np.count_nonzero(np.isfinite(mag) & (mag > 20.175))),
        }

    z_name = next((name for name in ("Z", "Z_NOT4CLUS", "Z_COSMO") if name in names), None)
    if z_name is not None:
        z = np.asarray(tab[z_name], dtype=np.float64)
        out["redshift_column"] = z_name
        out["finite_positive_redshift_rows"] = int(np.count_nonzero(np.isfinite(z) & (z > 0)))
        out["shell_rows"] = {
            "all": shell_counts(z, np.ones(n, dtype=bool)),
            "bright": shell_counts(z, bright),
            "faint": shell_counts(z, faint),
        }

    if {"ZWARN", "DELTACHI2", "SPECTYPE"}.issubset(names):
        spectype = np.char.strip(np.asarray(tab["SPECTYPE"]).astype("S16"))
        good = (
            (np.asarray(tab["ZWARN"]) == 0)
            & (np.asarray(tab["DELTACHI2"], dtype=np.float64) >= 25.0)
            & (spectype == b"GALAXY")
        )
        out["good_spectrum_rows"] = {
            "all": int(np.count_nonzero(good)),
            "bright": int(np.count_nonzero(good & bright)),
            "faint": int(np.count_nonzero(good & faint)),
        }

    truth_columns = {"FILE_NUM", "BOX_INDEX", "HALO_INDEX"}
    if truth_columns.issubset(names):
        valid_truth = (
            (np.asarray(tab["FILE_NUM"], dtype=np.int64) >= 0)
            & (np.asarray(tab["BOX_INDEX"], dtype=np.int64) >= 0)
            & (np.asarray(tab["HALO_INDEX"], dtype=np.int64) >= 0)
        )
        out["valid_truth_link_rows"] = {
            "all": int(np.count_nonzero(valid_truth)),
            "bright": int(np.count_nonzero(valid_truth & bright)),
            "faint": int(np.count_nonzero(valid_truth & faint)),
        }

    return out


def _merge_counts(total: dict[str, object], chunk: dict[str, object]) -> None:
    for key, value in chunk.items():
        if key == "redshift_column":
            total.setdefault(key, value)
        elif isinstance(value, dict):
            child = total.setdefault(key, {})
            assert isinstance(child, dict)
            _merge_counts(child, value)
        elif isinstance(value, (int, np.integer)):
            total[key] = int(total.get(key, 0)) + int(value)


def audit_fits(
    path: Path,
    bright_mask: int,
    faint_mask: int,
    chunk_rows: int,
    exact_targetid_uniqueness: bool = True,
) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False}

    stat = path.stat()
    with fitsio.FITS(str(path), "r") as f:
        if len(f) < 2:
            raise ValueError(f"No table extension in {path}")
        hdu = f[1]
        columns = list(hdu.get_colnames())
        read_columns = [name for name in AUDIT_COLUMNS if name in columns]
        nrows = int(hdu.get_nrows())
        summary: dict[str, object] = {}
        targetids: list[np.ndarray] = []
        for start in range(0, nrows, chunk_rows):
            stop = min(start + chunk_rows, nrows)
            rows = np.arange(start, stop, dtype=np.int64)
            tab = hdu.read(rows=rows, columns=read_columns)
            _merge_counts(summary, summarize_chunk(tab, bright_mask, faint_mask))
            if exact_targetid_uniqueness and "TARGETID" in read_columns:
                targetids.append(np.asarray(tab["TARGETID"], dtype=np.int64))

    result: dict[str, object] = {
        "path": str(path),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "hdu": 1,
        "rows_header": nrows,
        "columns": columns,
        "read_columns": read_columns,
        "summary": summary,
    }
    if targetids:
        values = np.concatenate(targetids)
        unique = np.unique(values).size
        result["targetid_uniqueness"] = {
            "checked_exactly": True,
            "unique_rows": int(unique),
            "duplicate_rows": int(values.size - unique),
        }
    else:
        result["targetid_uniqueness"] = {
            "checked_exactly": False,
            "reason": "TARGETID absent or --skip-targetid-uniqueness",
        }
    return result


def feasibility_decision(stages: dict[str, dict[str, object]]) -> dict[str, object]:
    def faint_rows(name: str) -> int:
        summary = stages.get(name, {}).get("summary", {})
        return int(summary.get("faint_rows", 0)) if isinstance(summary, dict) else 0

    upstream_faint = faint_rows("forfa_targets") > 0 and faint_rows("fiberassign_input") > 0
    assigned_faint = faint_rows("fiberassign_assigned") > 0
    joined_faint = faint_rows("spectroscopic_join") > 0
    final_faint = faint_rows("graphweb_bright_final") > 0

    reusable = []
    for name in ("fiberassign_assigned", "fiberassign_all_targets", "spectroscopic_join"):
        stage = stages.get(name, {})
        columns = set(stage.get("columns", []))
        if faint_rows(name) and {"TARGETID", "BGS_TARGET", "RA", "DEC"}.issubset(columns):
            reusable.append(name)

    blockers = []
    if not upstream_faint:
        blockers.append("BGS_FAINT is absent before fibre assignment")
    if not assigned_faint:
        blockers.append("no assigned BGS_FAINT rows were found in the current assigned product")
    if not joined_faint:
        blockers.append("no BGS_FAINT rows were found in the current spectroscopic-join product")
    if final_faint:
        blockers.append("unexpected BGS_FAINT contamination in the frozen Bright-only final catalogue")
    blockers.extend(
        [
            "upstream BGS_FAINT is magnitude-selected then randomly retained at 0.695 without a frozen RNG seed",
            "the current LOA spectroscopic-success calibration is fitted to BGS_BRIGHT only",
            "the current LSS build and GraphWeb export explicitly select BGS_BRIGHT",
        ]
    )

    if upstream_faint:
        verdict = "CONDITIONAL_GO_BUILD_RESPONSE_COMPLETE_FAINT"
    else:
        verdict = "NO_GO_REGENERATE_UPSTREAM_TARGETS"

    return {
        "f0_audit_complete": True,
        "verdict": verdict,
        "multitracer_training_ready": False,
        "bright_only_reference_frozen": True,
        "upstream_faint_exists": upstream_faint,
        "assigned_faint_exists": assigned_faint,
        "spectroscopic_join_faint_exists": joined_faint,
        "final_graphweb_faint_exists": final_faint,
        "engineering_reuse_candidates": reusable,
        "blocking_findings": blockers,
        "next_gate": (
            "construct deterministic selection- and response-complete Bright+Faint catalogues, "
            "then rerun this audit before building fields or union graphs"
        ),
    }


def parse_stage_overrides(values: Iterable[str]) -> dict[str, Path]:
    stages = dict(DEFAULT_STAGES)
    for value in values:
        if "=" not in value:
            raise ValueError(f"--stage expects NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        stages[name] = Path(path).expanduser().resolve()
    return stages


def git_sha(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--chunk-rows", type=int, default=500_000)
    parser.add_argument("--skip-targetid-uniqueness", action="store_true")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    if args.chunk_rows <= 0:
        raise ValueError("--chunk-rows must be positive")

    bright_mask, faint_mask, bit_values, mask_source = target_masks()
    stage_paths = parse_stage_overrides(args.stage)
    stage_results = {}
    for name, path in stage_paths.items():
        print(f"auditing {name}: {path}", flush=True)
        stage_results[name] = audit_fits(
            path,
            bright_mask=bright_mask,
            faint_mask=faint_mask,
            chunk_rows=args.chunk_rows,
            exact_targetid_uniqueness=not args.skip_targetid_uniqueness,
        )

    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_sha": git_sha(args.repo),
        "scope": "ph000 current staged mock; BGS_FAINT context feasibility only",
        "target_bits": {
            "source": mask_source,
            "bright_union": bright_mask,
            "faint_union": faint_mask,
            "values": bit_values,
        },
        "known_code_contract": {
            "upstream_target_selection": (
                "R_MAG_APP<19.5 -> BGS_BRIGHT; 19.5<=R_MAG_APP<=20.175 -> "
                "BGS_FAINT; retain 0.695 uniformly and promote 0.2 of retained faint"
            ),
            "upstream_rng": "np.random.uniform without an explicit seed in the current script",
            "lss_enforcement": "run_path1_mkcat.sh invokes --tracer BGS_BRIGHT",
            "spectroscopic_calibration": "inject_loa_spec_from_zall.py calibrates BRIGHT_BITS only",
            "graphweb_enforcement": (
                "build_mock_bgs_maglim_catalog.py defaults to --bright-only and applies the Bright bit mask"
            ),
            "scientific_warning": (
                "The current faint target recipe is an engineering approximation, not the final "
                "DESI BGS_FAINT colour selection and response model."
            ),
        },
        "stages": stage_results,
    }
    payload["decision"] = feasibility_decision(stage_results)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    marker = args.out.parent / "F0_FEASIBILITY_COMPLETE"
    marker.write_text(
        f"{payload['created_utc']}\n{payload['decision']['verdict']}\n{args.out}\n"
    )
    print(json.dumps(payload["decision"], indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)
    print(f"wrote {marker}", flush=True)


if __name__ == "__main__":
    main()
