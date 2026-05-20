#!/usr/bin/env python3
"""Build stage-3 mock wedge NPZ + wedge_targets.fits for multiple sky wedges.

Each product is ready for ``build_abacus_sbi_cache.py`` once a matching
``*_cugraph_gnn_metadata.json`` exists (graph subset is a separate step).

Outputs per wedge (under --out-dir):
  staged_mock_wedge_stage3_<tag>_rs7.npz
  staged_mock_wedge_stage3_<tag>_wedge_targets.fits
  staged_mock_wedge_stage3_<tag>_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from build_staged_mock_wedge_truth_npz import (
    ANNOTATED_DEFAULT,
    STAGE3_DEFAULT,
    OUT_DIR_DEFAULT,
    _Stage3Mask,
    collect_needed_triples,
    extract_stage_npz,
    save_npz,
    scan_annotated_wedge_truth,
    stable_sort_arrays,
    triples_to_packed_set,
    write_product_manifest,
    write_wedge_targets_fits,
)

WEDGE1 = dict(
    tag="ra120_140_dec16p5_26p7_z0p2_0p3",
    ra_min=120.0,
    ra_max=140.0,
    dec_min=16.5,
    dec_max=26.7,
    z_min=0.2,
    z_max=0.3,
)

WEDGE2_DEFAULTS = dict(
    ra_min=128.0,
    ra_max=138.0,
    dec_min=18.0,
    dec_max=25.0,
    z_min=0.0,
    z_max=0.5,
)


def _tag_from_wedge(wedge: dict[str, float]) -> str:
    def _f(x: float) -> str:
        s = f"{x:g}".replace(".", "p")
        return s

    return (
        f"ra{_f(wedge['ra_min'])}_{_f(wedge['ra_max'])}"
        f"_dec{_f(wedge['dec_min'])}_{_f(wedge['dec_max'])}"
        f"_z{_f(wedge['z_min'])}_{_f(wedge['z_max'])}"
    )


def load_wedge2_spec(out_dir: Path, json_path: Path | None) -> dict:
    wedge = dict(WEDGE2_DEFAULTS)
    candidates: list[Path] = []
    if json_path is not None:
        candidates.append(json_path.expanduser().resolve())
    candidates.extend(
        [
            out_dir / "wedge2_ra_dec_recommendation.json",
            Path(__file__).resolve().parent / "wedge2_ra_dec_recommendation.json",
        ]
    )
    source_json = None
    raw: dict | None = None
    for p in candidates:
        if p.is_file():
            with p.open(encoding="utf-8") as f:
                raw = json.load(f)
            for k in WEDGE2_DEFAULTS:
                if k in raw:
                    wedge[k] = float(raw[k])
            source_json = str(p)
            break
    tag = str(raw["tag"]) if isinstance(raw, dict) and raw.get("tag") else _tag_from_wedge(wedge)
    if source_json is None:
        tag = f"wedge2_{tag}"
    return {"tag": tag, **wedge, "wedge2_json": source_json}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotated-fits", type=Path, default=Path(ANNOTATED_DEFAULT))
    p.add_argument("--stage3-fits", type=Path, default=Path(STAGE3_DEFAULT))
    p.add_argument("--out-dir", type=Path, default=Path(OUT_DIR_DEFAULT))
    p.add_argument("--chunk-size", type=int, default=1_000_000)
    p.add_argument("--eig-threshold", type=float, default=0.2)
    p.add_argument("--redshift-col", default="Z")
    p.add_argument(
        "--wedge",
        choices=("wedge1", "wedge2", "both"),
        default="both",
        help="Which preset wedge(s) to build.",
    )
    p.add_argument("--wedge-name", default=None, help="Override output tag for single custom wedge.")
    p.add_argument("--wedge2-json", type=Path, default=None)
    p.add_argument("--no-write-xyz", action="store_true")
    for k in ("ra-min", "ra-max", "dec-min", "dec-max", "z-min", "z-max"):
        p.add_argument(f"--{k}", type=float, default=None)
    return p.parse_args()


def _specs_from_args(args: argparse.Namespace) -> list[dict]:
    out_dir = args.out_dir.expanduser().resolve()
    specs: list[dict] = []
    if args.wedge in ("wedge1", "both"):
        specs.append(dict(WEDGE1))
    if args.wedge in ("wedge2", "both"):
        specs.append(load_wedge2_spec(out_dir, args.wedge2_json))

    custom_bounds = [args.ra_min, args.ra_max, args.dec_min, args.dec_max, args.z_min, args.z_max]
    if any(v is not None for v in custom_bounds):
        if not all(v is not None for v in custom_bounds):
            raise SystemExit("Custom wedge requires all of --ra-min ... --z-max.")
        wedge = dict(
            ra_min=args.ra_min,
            ra_max=args.ra_max,
            dec_min=args.dec_min,
            dec_max=args.dec_max,
            z_min=args.z_min,
            z_max=args.z_max,
        )
        tag = args.wedge_name or _tag_from_wedge(wedge)
        specs = [{"tag": tag, **wedge}]
    elif args.wedge_name:
        raise SystemExit("--wedge-name requires custom RA/Dec/z bounds.")

    return specs


def build_one_stage3_wedge(
    spec: dict,
    *,
    annotated: Path,
    stage3: Path,
    out_dir: Path,
    chunk_size: int,
    eig_thr: float,
    z_col: str,
    write_xyz: bool,
) -> dict:
    tag = spec["tag"]
    wedge = {k: spec[k] for k in ("ra_min", "ra_max", "dec_min", "dec_max", "z_min", "z_max")}
    t0 = time.time()

    stage_specs = [("stage3", stage3, _Stage3Mask())]
    needed = collect_needed_triples(
        stage_specs,
        wedge=wedge,
        z_col=z_col,
        chunk_size=chunk_size,
    )
    needed_packed = triples_to_packed_set(needed)
    _truth, lookup = scan_annotated_wedge_truth(
        annotated,
        needed_packed,
        wedge=wedge,
        z_col=z_col,
        chunk_size=chunk_size,
        eig_thr=eig_thr,
        write_xyz=write_xyz,
    )

    npz_name = f"staged_mock_wedge_stage3_{tag}_rs7.npz"
    arrays, stats = extract_stage_npz(
        stage3,
        lookup,
        stage_name=npz_name,
        wedge=wedge,
        z_col=z_col,
        chunk_size=chunk_size,
        eig_thr=eig_thr,
        write_xyz=write_xyz,
        extra_mask_fn=_Stage3Mask(),
        dedupe_triple=True,
    )
    arrays = stable_sort_arrays(arrays)

    n_gal = int(arrays.get("ra", np.array([])).size)
    filters = {
        "stage": "stage3_datcomb_brightwdup",
        "COLLISION": "== 0",
        "dedupe_halo_triple": True,
        "eig_join": "annotated_cutsky (FILE_NUM, HALO_INDEX, BOX_INDEX)",
    }
    meta = {
        "product": npz_name,
        "n_gal": n_gal,
        "wedge": wedge,
        "bounds": wedge,
        "filters": filters,
        "redshift_col": z_col,
        "eig_threshold": eig_thr,
        "class_fractions": stats.get("class_fractions"),
        **stats,
        "annotated_fits": str(annotated),
        "stage3_fits": str(stage3),
        "gnn_metadata_note": (
            "Next step: subset mock graph to this wedge and write "
            f"<prefix>_cugraph_gnn_metadata.json aligned to {npz_name} row order "
            "(or wedge_targets.fits stable sort by TARGETID / halo triple)."
        ),
    }
    if spec.get("wedge2_json"):
        meta["wedge2_ra_dec_recommendation_json"] = spec["wedge2_json"]

    out_npz = out_dir / npz_name
    save_npz(out_npz, arrays, meta)

    fits_out = out_dir / f"staged_mock_wedge_stage3_{tag}_wedge_targets.fits"
    write_wedge_targets_fits(fits_out, arrays)

    manifest_path = out_dir / f"staged_mock_wedge_stage3_{tag}_manifest.json"
    write_product_manifest(manifest_path, meta)

    meta["elapsed_sec"] = time.time() - t0
    meta["outputs"] = {
        "npz": str(out_npz),
        "wedge_targets_fits": str(fits_out),
        "manifest": str(manifest_path),
    }
    return meta


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = _specs_from_args(args)
    if not specs:
        raise SystemExit("No wedge specs to build.")

    summary = []
    for spec in specs:
        print(f"=== Building stage3 wedge tag={spec['tag']} ===", flush=True)
        summary.append(
            build_one_stage3_wedge(
                spec,
                annotated=args.annotated_fits.expanduser().resolve(),
                stage3=args.stage3_fits.expanduser().resolve(),
                out_dir=out_dir,
                chunk_size=args.chunk_size,
                eig_thr=args.eig_threshold,
                z_col=args.redshift_col,
                write_xyz=not args.no_write_xyz,
            )
        )

    summary_path = out_dir / "staged_mock_wedge_stage3_variants_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
