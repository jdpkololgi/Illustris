#!/usr/bin/env python3
"""Apply the registered neutral/defined policy to P10 R2 angular response.

Undefined target-derived response inside random support is assigned the neutral
no-competition value one and accompanied by an explicit definition flag. Values
outside support remain zero.  No spatial smoothing or interpolation is used.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from p10_audit_r2_response_ladder import atomic_json, sha256


VISIBLE_PHASES = ("ph000", "ph002", "ph003", "ph004", "ph005", "ph006")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        default="/pscratch/sd/d/dkololgi/abacus/p10_multiphase/r2_assignment_canary_all_v1",
    )
    parser.add_argument(
        "--output-root",
        default="/pscratch/sd/d/dkololgi/abacus/p10_multiphase/r2_assignment_response_v1",
    )
    parser.add_argument("--phases", nargs="+", default=list(VISIBLE_PHASES))
    args = parser.parse_args()
    if "ph001" in args.phases:
        raise SystemExit("ph001 is sealed")

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    phase_reports: dict[str, dict[str, Any]] = {}
    schema = None
    for phase in args.phases:
        source = input_root / phase / "assignment_angular_canary.npz"
        with np.load(source) as archive:
            support = np.asarray(archive["support"], dtype=np.uint8)
            domain = np.asarray(archive["domain"], dtype=np.int8)
            target_count = np.asarray(archive["target_count"], dtype=np.int64)
            defined = np.asarray(archive["has_target"], dtype=bool) & (support == 1)
            tileloc_raw = np.asarray(archive["frac_tileloc_mean"], dtype=np.float32)
            tiles_raw = np.asarray(archive["frac_tlobs_tiles_mean"], dtype=np.float32)
            product_raw = np.asarray(archive["completeness_product_mean"], dtype=np.float32)

        supported = support == 1
        undefined = supported & ~defined
        outside = ~supported

        tileloc = np.zeros_like(tileloc_raw)
        tiles = np.zeros_like(tiles_raw)
        product = np.zeros_like(product_raw)
        tileloc[defined] = tileloc_raw[defined]
        tiles[defined] = tiles_raw[defined]
        product[defined] = product_raw[defined]
        tileloc[undefined] = 1.0
        tiles[undefined] = 1.0
        product[undefined] = 1.0
        c_z = support.astype(np.float32)
        c_z_informative = np.zeros_like(support, dtype=np.uint8)

        phase_dir = output_root / phase
        phase_dir.mkdir(parents=True, exist_ok=True)
        output = phase_dir / "assignment_response_angular.npz"
        np.savez_compressed(
            output,
            support=support,
            domain=domain,
            target_count=target_count,
            c_fibre_defined=defined.astype(np.uint8),
            c_fibre_tileloc=tileloc,
            c_fibre_tiles=tiles,
            c_fibre_product=product,
            c_z=c_z,
            c_z_informative=c_z_informative,
        )

        gate = bool(
            np.all(tileloc[outside] == 0.0)
            and np.all(tiles[outside] == 0.0)
            and np.all(product[outside] == 0.0)
            and np.all(tileloc[undefined] == 1.0)
            and np.all(tiles[undefined] == 1.0)
            and np.all(product[undefined] == 1.0)
            and np.all((tileloc[supported] >= 0.0) & (tileloc[supported] <= 1.0))
            and np.all((tiles[supported] >= 0.0) & (tiles[supported] <= 1.0))
            and np.all((product[supported] >= 0.0) & (product[supported] <= 1.0))
            and np.all(c_z[supported] == 1.0)
            and np.all(c_z[outside] == 0.0)
            and np.all(c_z_informative == 0)
        )
        report = {
            "schema_version": "p10-r2-assignment-response-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "blind_phase_opened": False,
            "source": {"path": str(source), "sha256": sha256(source)},
            "output": {"path": str(output), "sha256": sha256(output)},
            "policy": {
                "undefined_inside_support": "neutral_no_competition_value_1",
                "undefined_flag": "c_fibre_defined=0",
                "outside_support": "all_response_values_0",
                "spatial_interpolation": "none",
                "C_z": "1_on_support",
                "C_z_informative": False,
            },
            "counts": {
                "supported_pixels": int(supported.sum()),
                "defined_pixels": int(defined.sum()),
                "undefined_pixels": int(undefined.sum()),
            },
            "defined_fraction": float(defined.sum() / max(supported.sum(), 1)),
            "pass": gate,
        }
        report_path = phase_dir / "assignment_response_angular.json"
        atomic_json(report_path, report)
        phase_reports[phase] = report

        with np.load(output) as check:
            this_schema = tuple(check.files)
        if schema is None:
            schema = this_schema
        elif this_schema != schema:
            raise ValueError(f"{phase}: output schema drift")
        print(f"[{phase}] defined={report['defined_fraction']:.8f} pass={gate}")

    summary = {
        "schema_version": "p10-r2-assignment-response-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phases": list(args.phases),
        "blind_phase_opened": False,
        "channels": list(schema or ()),
        "phase_reports": {
            phase: {
                "path": str(output_root / phase / "assignment_response_angular.json"),
                "sha256": sha256(output_root / phase / "assignment_response_angular.json"),
                "pass": phase_reports[phase]["pass"],
            }
            for phase in args.phases
        },
        "pass": bool(all(phase_reports[phase]["pass"] for phase in args.phases)),
        "three_dimensional_overlays_ready": False,
        "view_ladder_marker_written": False,
    }
    marker = output_root / "R2_ASSIGNMENT_POLICY_READY.json"
    atomic_json(marker, summary)
    print(marker)


if __name__ == "__main__":
    main()
