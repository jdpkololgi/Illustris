#!/usr/bin/env python3
"""Build ph000/ph006 angular assignment-response canaries for P10 R2.

The output remains a canary rather than a frozen R2 overlay.  It establishes
whether target-level assignment completeness covers the independently defined
random-supported footprint at nside=256 without smoothing across holes.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitsio
import healpy as hp
import numpy as np

from p10_audit_r2_response_ladder import atomic_json, sha256


def _phase_canary(phase: str, registry_path: str, root: str, nside: int) -> dict[str, Any]:
    registry = json.loads(Path(registry_path).read_text())
    full_meta = registry["mock_phases"][phase]["data"]["full"]
    full = fitsio.read(
        full_meta["path"],
        columns=[
            "RA",
            "DEC",
            "GOODPRI",
            "GOODHARDLOC",
            "FRACZ_TILELOCID",
            "FRAC_TLOBS_TILES",
        ],
        ext=1,
    )
    eligible = np.asarray(full["GOODPRI"], dtype=bool) & np.asarray(
        full["GOODHARDLOC"], dtype=bool
    )
    ra = np.asarray(full["RA"], dtype=np.float64)[eligible]
    dec = np.asarray(full["DEC"], dtype=np.float64)[eligible]
    tileloc = np.asarray(full["FRACZ_TILELOCID"], dtype=np.float64)[eligible]
    tiles = np.asarray(full["FRAC_TLOBS_TILES"], dtype=np.float64)[eligible]
    pixel = hp.ang2pix(nside, ra, dec, lonlat=True, nest=False)
    npix = hp.nside2npix(nside)
    counts = np.bincount(pixel, minlength=npix).astype(np.int64)
    sum_tileloc = np.bincount(pixel, weights=tileloc, minlength=npix)
    sum_tiles = np.bincount(pixel, weights=tiles, minlength=npix)
    sum_product = np.bincount(pixel, weights=tileloc * tiles, minlength=npix)
    has_target = counts > 0

    mean_tileloc = np.zeros(npix, dtype=np.float32)
    mean_tiles = np.zeros(npix, dtype=np.float32)
    mean_product = np.zeros(npix, dtype=np.float32)
    mean_tileloc[has_target] = (sum_tileloc[has_target] / counts[has_target]).astype(np.float32)
    mean_tiles[has_target] = (sum_tiles[has_target] / counts[has_target]).astype(np.float32)
    mean_product[has_target] = (sum_product[has_target] / counts[has_target]).astype(np.float32)

    random_map_path = (
        Path(root).parent
        / phase
        / "p3b_random_response_v1"
        / "angular"
        / "randoms_n18.npz"
    )
    with np.load(random_map_path) as random_map:
        support = np.asarray(random_map["support"], dtype=np.uint8)
        domain = np.asarray(random_map["domain"], dtype=np.int8)
    if support.size != npix:
        raise ValueError(f"{phase}: random support has {support.size} pixels, expected {npix}")
    supported = support == 1
    covered = supported & has_target
    uncovered = supported & ~has_target
    coverage = float(covered.sum() / max(supported.sum(), 1))

    output_dir = Path(root) / phase
    output_dir.mkdir(parents=True, exist_ok=True)
    map_path = output_dir / "assignment_angular_canary.npz"
    np.savez_compressed(
        map_path,
        support=support,
        domain=domain,
        target_count=counts,
        has_target=has_target.astype(np.uint8),
        frac_tileloc_mean=mean_tileloc,
        frac_tlobs_tiles_mean=mean_tiles,
        completeness_product_mean=mean_product,
    )
    report = {
        "schema_version": "p10-r2-assignment-angular-canary-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "blind_phase_opened": False,
        "nside": nside,
        "ordering": "RING",
        "source": {
            "path": full_meta["path"],
            "registered_sha256": full_meta["sha256"],
            "random_support_path": str(random_map_path),
            "random_support_sha256": sha256(random_map_path),
        },
        "map": {"path": str(map_path), "sha256": sha256(map_path)},
        "counts": {
            "eligible_targets": int(eligible.sum()),
            "supported_pixels": int(supported.sum()),
            "covered_supported_pixels": int(covered.sum()),
            "uncovered_supported_pixels": int(uncovered.sum()),
        },
        "coverage_fraction": coverage,
        "uncovered_by_domain": {
            str(int(value)): int(np.count_nonzero(uncovered & (domain == value)))
            for value in np.unique(domain[supported])
        },
        "response_quantiles_on_covered_support": {
            "FRACZ_TILELOCID": np.quantile(
                mean_tileloc[covered], [0.0, 0.01, 0.5, 0.99, 1.0]
            ).tolist(),
            "FRAC_TLOBS_TILES": np.quantile(
                mean_tiles[covered], [0.0, 0.01, 0.5, 0.99, 1.0]
            ).tolist(),
            "product": np.quantile(
                mean_product[covered], [0.0, 0.01, 0.5, 0.99, 1.0]
            ).tolist(),
        },
        "gates": {
            "coverage_ge_0p999": coverage >= 0.999,
            "finite_on_covered_support": bool(
                np.all(np.isfinite(mean_tileloc[covered]))
                and np.all(np.isfinite(mean_tiles[covered]))
                and np.all(np.isfinite(mean_product[covered]))
            ),
            "probability_ranges": bool(
                np.all((mean_tileloc[covered] >= 0.0) & (mean_tileloc[covered] <= 1.0))
                and np.all((mean_tiles[covered] >= 0.0) & (mean_tiles[covered] <= 1.0))
                and np.all((mean_product[covered] >= 0.0) & (mean_product[covered] <= 1.0))
            ),
        },
    }
    report["pass"] = bool(all(report["gates"].values()))
    report_path = output_dir / "assignment_angular_canary.json"
    atomic_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="configs/p10_response_sources_v1.json")
    parser.add_argument(
        "--output-root",
        default="/pscratch/sd/d/dkololgi/abacus/p10_multiphase/r2_assignment_canary_v1",
    )
    parser.add_argument("--nside", type=int, default=256)
    parser.add_argument("--phases", nargs="+", default=["ph000", "ph006"])
    args = parser.parse_args()
    if "ph001" in args.phases:
        raise SystemExit("ph001 is sealed")

    results: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=len(args.phases)) as pool:
        futures = {
            pool.submit(
                _phase_canary,
                phase,
                str(Path(args.registry).resolve()),
                args.output_root,
                args.nside,
            ): phase
            for phase in args.phases
        }
        for future in as_completed(futures):
            phase = futures[future]
            results[phase] = future.result()
            print(
                f"[{phase}] coverage={results[phase]['coverage_fraction']:.8f} "
                f"pass={results[phase]['pass']}",
                flush=True,
            )

    summary = {
        "schema_version": "p10-r2-assignment-angular-canary-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phases": list(args.phases),
        "blind_phase_opened": False,
        "reports": {
            phase: {
                "path": str(Path(args.output_root) / phase / "assignment_angular_canary.json"),
                "sha256": sha256(
                    Path(args.output_root) / phase / "assignment_angular_canary.json"
                ),
                "pass": results[phase]["pass"],
            }
            for phase in args.phases
        },
        "pass": bool(all(results[phase]["pass"] for phase in args.phases)),
        "view_ladder_marker_written": False,
    }
    path = Path(args.output_root) / "R2_ASSIGNMENT_ANGULAR_CANARY.json"
    atomic_json(path, summary)
    print(path)


if __name__ == "__main__":
    main()
