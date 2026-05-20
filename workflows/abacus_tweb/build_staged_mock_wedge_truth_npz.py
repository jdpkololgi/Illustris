#!/usr/bin/env python3
"""Build wedge-truth NPZs for staged Abacus SecondGen mocks.

Funnel (4 NPZs):
  1) Wedge subset of annotated CutSky (T-Web eigenvalues in FITS).
  2) Stage-1 mag cut, 3) forFA, 4) Stage-3 datcomb post-collision unique halos.

Eigenvalues for stages 2–4 are joined from the annotated CutSky via
(FILE_NUM, HALO_INDEX, BOX_INDEX) using one streaming pass over the annotated
catalog to populate a lookup for the union of triples required by those stages.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import fitsio
import numpy as np
from astropy.cosmology import Planck18 as cosmo

WEDGE_DEFAULTS = dict(
    ra_min=120.0,
    ra_max=140.0,
    dec_min=16.5,
    dec_max=26.7,
    z_min=0.25,
    z_max=0.30,
)

ANNOTATED_DEFAULT = (
    "/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_24042026_rsmooth_7/"
    "cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_with_tweb_eigs_rs7_ngrid2048_thr0p2.fits"
)
STAGE1_DEFAULT = (
    "/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/stage_1/"
    "cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000_subset_ALL_rmaglt19.5.fits"
)
FORFA_DEFAULT = (
    "/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/stage_2/"
    "SecondGenMocks/AbacusSummitBGS_v2/forFA0_nomask.fits"
)
STAGE3_DEFAULT = (
    "/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/stage_3/fba0/"
    "datcomb_brightwdup.fits"
)
OUT_DIR_DEFAULT = "/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/wedge/"

EIG_THR = 0.2
TRIPLE_COLS = ("FILE_NUM", "HALO_INDEX", "BOX_INDEX")
POS_COLS = ("RA", "DEC", "Z")
EIG_COLS = ("LAMBDA1", "LAMBDA2", "LAMBDA3")


def pack_triple(fn: np.ndarray, hi: np.ndarray, bi: np.ndarray) -> np.ndarray:
    return (
        fn.astype(np.uint64, copy=False) << np.uint64(42)
        | (hi.astype(np.uint64, copy=False) & np.uint64((1 << 21) - 1)) << np.uint64(21)
        | (bi.astype(np.uint64, copy=False) & np.uint64((1 << 21) - 1))
    )


def triples_to_packed_set(triples: set[tuple[int, int, int]]) -> set[np.uint64]:
    if not triples:
        return set()
    fn = np.fromiter((t[0] for t in triples), dtype=np.int64)
    hi = np.fromiter((t[1] for t in triples), dtype=np.int64)
    bi = np.fromiter((t[2] for t in triples), dtype=np.int64)
    return set(pack_triple(fn, hi, bi).tolist())


def unpack_triple(packed: int) -> tuple[int, int, int]:
    p = int(packed)
    bi = p & ((1 << 21) - 1)
    hi = (p >> 21) & ((1 << 21) - 1)
    fn = p >> 42
    return fn, hi, bi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotated-fits", type=Path, default=Path(ANNOTATED_DEFAULT))
    p.add_argument("--stage1-fits", type=Path, default=Path(STAGE1_DEFAULT))
    p.add_argument("--forfa-fits", type=Path, default=Path(FORFA_DEFAULT))
    p.add_argument("--stage3-fits", type=Path, default=Path(STAGE3_DEFAULT))
    p.add_argument("--out-dir", type=Path, default=Path(OUT_DIR_DEFAULT))
    p.add_argument("--chunk-size", type=int, default=1_000_000)
    p.add_argument("--eig-threshold", type=float, default=EIG_THR)
    p.add_argument("--write-xyz", action="store_true", help="Include x,y,z comoving Mpc arrays.")
    p.add_argument("--no-write-xyz", action="store_false", dest="write_xyz")
    p.set_defaults(write_xyz=True)
    for k, v in WEDGE_DEFAULTS.items():
        p.add_argument(f"--{k.replace('_', '-')}", type=float, default=v)
    p.add_argument("--redshift-col", default="Z")
    return p.parse_args()


def _resolve_col(colnames: list[str], name: str) -> str:
    m = {c.upper(): c for c in colnames}
    r = m.get(name.upper())
    if r is None:
        raise KeyError(f"Column {name!r} not in FITS (have {len(colnames)} cols).")
    return r


def wedge_mask(
    ra: np.ndarray,
    dec: np.ndarray,
    z: np.ndarray,
    *,
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    z_min: float,
    z_max: float,
) -> np.ndarray:
    if ra_min <= ra_max:
        m_ra = (ra >= ra_min) & (ra <= ra_max)
    else:
        m_ra = (ra >= ra_min) | (ra <= ra_max)
    return (
        m_ra
        & (dec >= dec_min)
        & (dec <= dec_max)
        & (z >= z_min)
        & (z <= z_max)
    )


def classify_eigs(l1: np.ndarray, l2: np.ndarray, l3: np.ndarray, thr: float) -> np.ndarray:
    return (
        (l1 > thr).astype(np.int8)
        + (l2 > thr).astype(np.int8)
        + (l3 > thr).astype(np.int8)
    )


def sky_to_xyz_mpc(ra_deg: np.ndarray, dec_deg: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.asarray(cosmo.comoving_distance(z).value, dtype=np.float64)
    ra = np.deg2rad(ra_deg.astype(np.float64, copy=False))
    dec = np.deg2rad(dec_deg.astype(np.float64, copy=False))
    x = r * np.cos(dec) * np.cos(ra)
    y = r * np.cos(dec) * np.sin(ra)
    zc = r * np.sin(dec)
    return x, y, zc


def _class_fractions(cls: np.ndarray) -> dict[str, float]:
    n = cls.size
    if n == 0:
        return {str(k): 0.0 for k in range(4)}
    counts = np.bincount(cls.astype(np.int64), minlength=4)
    return {str(k): float(counts[k] / n) for k in range(4)}


def _iter_fits_chunks(path: Path, columns: list[str], chunk_size: int):
    with fitsio.FITS(str(path)) as ff:
        hdu = ff[1]
        n = int(hdu.get_nrows())
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            yield start, hdu[columns][start:end]


def collect_needed_triples(
    stage_specs: list[tuple[str, Path, dict]],
    *,
    wedge: dict[str, float],
    z_col: str,
    chunk_size: int,
) -> set[tuple[int, int, int]]:
    needed: set[tuple[int, int, int]] = set()
    for _name, path, extra_mask_fn in stage_specs:
        with fitsio.FITS(str(path)) as ff:
            cols = [str(c) for c in ff[1].get_colnames()]
        ra_c = _resolve_col(cols, "RA")
        dec_c = _resolve_col(cols, "DEC")
        z_c = _resolve_col(cols, z_col)
        fn_c, hi_c, bi_c = (_resolve_col(cols, c) for c in TRIPLE_COLS)
        read_cols = list(dict.fromkeys([ra_c, dec_c, z_c, fn_c, hi_c, bi_c]))
        if extra_mask_fn is not None:
            extra_cols = extra_mask_fn.required_columns(cols)
            read_cols = list(dict.fromkeys(read_cols + extra_cols))

        for _start, tab in _iter_fits_chunks(path, read_cols, chunk_size):
            m = wedge_mask(
                tab[ra_c],
                tab[dec_c],
                tab[z_c],
                **wedge,
            )
            if extra_mask_fn is not None:
                m &= extra_mask_fn(tab, cols)
            if not np.any(m):
                continue
            fn = tab[fn_c][m].astype(np.int64, copy=False)
            hi = tab[hi_c][m].astype(np.int64, copy=False)
            bi = tab[bi_c][m].astype(np.int64, copy=False)
            for f, h, b in zip(fn, hi, bi, strict=False):
                needed.add((int(f), int(h), int(b)))
    return needed


class _Stage3Mask:
    @staticmethod
    def required_columns(colnames: list[str]) -> list[str]:
        return [_resolve_col(colnames, "COLLISION")]

    @staticmethod
    def __call__(tab: dict, colnames: list[str]) -> np.ndarray:
        col = _resolve_col(colnames, "COLLISION")
        return tab[col] == 0



def _append_arrays(buf: dict[str, list], **kwargs) -> None:
    for k, v in kwargs.items():
        buf[k].append(np.asarray(v))


def _concat(buf: dict[str, list]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for k, parts in buf.items():
        if not parts:
            out[k] = np.array([], dtype=parts[0].dtype if parts else np.float64)
        else:
            out[k] = np.concatenate(parts)
    return out


def scan_annotated_wedge_truth(
    annotated: Path,
    needed_packed: set[np.uint64],
    *,
    wedge: dict[str, float],
    z_col: str,
    chunk_size: int,
    eig_thr: float,
    write_xyz: bool,
) -> tuple[dict[str, np.ndarray], dict[tuple[int, int, int], tuple[float, float, float]]]:
    with fitsio.FITS(str(annotated)) as ff:
        cols = [str(c) for c in ff[1].get_colnames()]
    ra_c, dec_c = _resolve_col(cols, "RA"), _resolve_col(cols, "DEC")
    z_c = _resolve_col(cols, z_col)
    fn_c, hi_c, bi_c = (_resolve_col(cols, c) for c in TRIPLE_COLS)
    l1_c, l2_c, l3_c = (_resolve_col(cols, c) for c in EIG_COLS)
    read_cols = list(dict.fromkeys([ra_c, dec_c, z_c, fn_c, hi_c, bi_c, l1_c, l2_c, l3_c]))

    buf: dict[str, list] = {k: [] for k in ("ra", "dec", "z", "lambda1", "lambda2", "lambda3", "cls")}
    if write_xyz:
        buf.update(x=[], y=[], zc=[])
    lookup: dict[tuple[int, int, int], tuple[float, float, float]] = {}
    needed_arr = (
        np.fromiter(needed_packed, dtype=np.uint64)
        if needed_packed
        else np.array([], dtype=np.uint64)
    )

    for _start, tab in _iter_fits_chunks(annotated, read_cols, chunk_size):
        fn = tab[fn_c].astype(np.int64, copy=False)
        hi = tab[hi_c].astype(np.int64, copy=False)
        bi = tab[bi_c].astype(np.int64, copy=False)
        l1 = tab[l1_c].astype(np.float32, copy=False)
        l2 = tab[l2_c].astype(np.float32, copy=False)
        l3 = tab[l3_c].astype(np.float32, copy=False)

        m_wedge = wedge_mask(tab[ra_c], tab[dec_c], tab[z_c], **wedge)
        if np.any(m_wedge):
            cls = classify_eigs(l1[m_wedge], l2[m_wedge], l3[m_wedge], eig_thr)
            kw = dict(
                ra=tab[ra_c][m_wedge].astype(np.float64, copy=False),
                dec=tab[dec_c][m_wedge].astype(np.float64, copy=False),
                z=tab[z_c][m_wedge].astype(np.float64, copy=False),
                lambda1=l1[m_wedge],
                lambda2=l2[m_wedge],
                lambda3=l3[m_wedge],
                cls=cls,
            )
            if write_xyz:
                x, y, zc = sky_to_xyz_mpc(kw["ra"], kw["dec"], kw["z"])
                kw.update(x=x, y=y, zc=zc)
            _append_arrays(buf, **kw)

        if needed_arr is not None and needed_arr.size:
            pk = pack_triple(fn, hi, bi)
            m_need = np.isin(pk, needed_arr)
            if np.any(m_need):
                for i in np.flatnonzero(m_need):
                    key = (int(fn[i]), int(hi[i]), int(bi[i]))
                    lookup[key] = (float(l1[i]), float(l2[i]), float(l3[i]))

    arrays = _concat(buf)
    if write_xyz and "zc" in arrays:
        arrays["z_comoving_cart"] = arrays.pop("zc")
    return arrays, lookup


def extract_stage_npz(
    path: Path,
    lookup: dict[tuple[int, int, int], tuple[float, float, float]],
    *,
    stage_name: str,
    wedge: dict[str, float],
    z_col: str,
    chunk_size: int,
    eig_thr: float,
    write_xyz: bool,
    extra_mask_fn,
    dedupe_triple: bool,
) -> tuple[dict[str, np.ndarray], dict]:
    with fitsio.FITS(str(path)) as ff:
        cols = [str(c) for c in ff[1].get_colnames()]
    ra_c, dec_c = _resolve_col(cols, "RA"), _resolve_col(cols, "DEC")
    z_c = _resolve_col(cols, z_col)
    fn_c, hi_c, bi_c = (_resolve_col(cols, c) for c in TRIPLE_COLS)
    read_cols = list(dict.fromkeys([ra_c, dec_c, z_c, fn_c, hi_c, bi_c]))
    if extra_mask_fn is not None:
        read_cols = list(dict.fromkeys(read_cols + extra_mask_fn.required_columns(cols)))

    buf: dict[str, list] = {k: [] for k in ("ra", "dec", "z", "lambda1", "lambda2", "lambda3", "cls", "file_num", "halo_index", "box_index")}
    if write_xyz:
        buf.update(x=[], y=[], zc=[])
    seen: set[tuple[int, int, int]] = set()

    n_wedge = 0
    n_join_ok = 0
    n_join_miss = 0
    n_dedup_skip = 0

    for _start, tab in _iter_fits_chunks(path, read_cols, chunk_size):
        m = wedge_mask(tab[ra_c], tab[dec_c], tab[z_c], **wedge)
        if extra_mask_fn is not None:
            m &= extra_mask_fn(tab, cols)
        if not np.any(m):
            continue
        n_wedge += int(np.count_nonzero(m))

        ra = tab[ra_c][m].astype(np.float64, copy=False)
        dec = tab[dec_c][m].astype(np.float64, copy=False)
        zz = tab[z_c][m].astype(np.float64, copy=False)
        fn = tab[fn_c][m].astype(np.int64, copy=False)
        hi = tab[hi_c][m].astype(np.int64, copy=False)
        bi = tab[bi_c][m].astype(np.int64, copy=False)

        l1_out = np.empty(fn.size, dtype=np.float32)
        l2_out = np.empty(fn.size, dtype=np.float32)
        l3_out = np.empty(fn.size, dtype=np.float32)
        keep = np.ones(fn.size, dtype=bool)

        for i in range(fn.size):
            key = (int(fn[i]), int(hi[i]), int(bi[i]))
            if dedupe_triple and key in seen:
                keep[i] = False
                n_dedup_skip += 1
                continue
            ev = lookup.get(key)
            if ev is None:
                keep[i] = False
                n_join_miss += 1
                continue
            l1_out[i], l2_out[i], l3_out[i] = ev
            n_join_ok += 1
            if dedupe_triple:
                seen.add(key)

        if not np.any(keep):
            continue
        ra, dec, zz = ra[keep], dec[keep], zz[keep]
        fn, hi, bi = fn[keep], hi[keep], bi[keep]
        l1_out, l2_out, l3_out = l1_out[keep], l2_out[keep], l3_out[keep]
        cls = classify_eigs(l1_out, l2_out, l3_out, eig_thr)
        kw = dict(
            ra=ra,
            dec=dec,
            z=zz,
            lambda1=l1_out,
            lambda2=l2_out,
            lambda3=l3_out,
            cls=cls,
            file_num=fn.astype(np.int32, copy=False),
            halo_index=hi.astype(np.int32, copy=False),
            box_index=bi.astype(np.int32, copy=False),
        )
        if write_xyz:
            x, y, zc = sky_to_xyz_mpc(ra, dec, zz)
            kw.update(x=x, y=y, zc=zc)
        _append_arrays(buf, **kw)

    arrays = _concat(buf)
    if write_xyz and "zc" in arrays:
        arrays["z_comoving_cart"] = arrays.pop("zc")

    denom = max(n_join_ok + n_join_miss, 1)
    stats = {
        "stage": stage_name,
        "n_rows_wedge_and_stage_mask": n_wedge,
        "n_join_ok": n_join_ok,
        "n_join_miss": n_join_miss,
        "join_miss_rate": float(n_join_miss / denom),
        "n_dedup_skip": n_dedup_skip,
        "class_fractions": _class_fractions(arrays.get("cls", np.array([], dtype=np.int8))),
    }
    return arrays, stats



def stable_sort_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Sort rows by (FILE_NUM, HALO_INDEX, BOX_INDEX) for reproducible cache builds."""
    if "file_num" not in arrays:
        return arrays
    order = np.lexsort(
        (
            arrays["box_index"],
            arrays["halo_index"],
            arrays["file_num"],
        )
    )
    return {k: v[order] for k, v in arrays.items()}


def dummy_targetid_from_triple(
    file_num: np.ndarray, halo_index: np.ndarray, box_index: np.ndarray
) -> np.ndarray:
    return pack_triple(
        file_num.astype(np.uint64, copy=False),
        halo_index.astype(np.uint64, copy=False),
        box_index.astype(np.uint64, copy=False),
    ).astype(np.int64, copy=False)


def write_wedge_targets_fits(out_path: Path, arrays: dict[str, np.ndarray]) -> None:
    arrays = stable_sort_arrays(arrays)
    n = int(arrays["ra"].size)
    fn = arrays.get("file_num", np.zeros(n, dtype=np.int32))
    hi = arrays.get("halo_index", np.zeros(n, dtype=np.int32))
    bi = arrays.get("box_index", np.zeros(n, dtype=np.int32))
    targetid = dummy_targetid_from_triple(fn, hi, bi)
    table = np.empty(
        n,
        dtype=[
            ("RA", "f8"),
            ("DEC", "f8"),
            ("Z", "f8"),
            ("LAMBDA1", "f4"),
            ("LAMBDA2", "f4"),
            ("LAMBDA3", "f4"),
            ("FILE_NUM", "i4"),
            ("HALO_INDEX", "i4"),
            ("BOX_INDEX", "i4"),
            ("TARGETID", "i8"),
        ],
    )
    table["RA"] = arrays["ra"]
    table["DEC"] = arrays["dec"]
    table["Z"] = arrays["z"]
    table["LAMBDA1"] = arrays["lambda1"]
    table["LAMBDA2"] = arrays["lambda2"]
    table["LAMBDA3"] = arrays["lambda3"]
    table["FILE_NUM"] = fn
    table["HALO_INDEX"] = hi
    table["BOX_INDEX"] = bi
    table["TARGETID"] = targetid
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fitsio.write(str(out_path), table, clobber=True)
    print(f"Wrote {out_path} (n={n:,})", flush=True)


def write_product_manifest(manifest_path: Path, metadata: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote manifest {manifest_path}", flush=True)


def save_npz(
    out_path: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: arrays[k] for k in ("ra", "dec", "z", "lambda1", "lambda2", "lambda3", "cls") if k in arrays}
    for opt in ("file_num", "halo_index", "box_index", "x", "y", "z_comoving_cart"):
        if opt in arrays:
            payload[opt] = arrays[opt]
    n_gal = int(payload["ra"].size)
    payload["n_gal"] = np.int64(n_gal)
    payload["metadata_json"] = np.array(json.dumps(metadata))
    np.savez_compressed(str(out_path), **payload)
    manifest = out_path.with_suffix(".manifest.json")
    with manifest.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote {out_path} (n_gal={n_gal:,}), manifest {manifest}", flush=True)


def main() -> int:
    args = parse_args()
    wedge = dict(
        ra_min=args.ra_min,
        ra_max=args.ra_max,
        dec_min=args.dec_min,
        dec_max=args.dec_max,
        z_min=args.z_min,
        z_max=args.z_max,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stage_specs = [
        ("stage1", args.stage1_fits, None),
        ("forFA", args.forfa_fits, None),
        ("stage3", args.stage3_fits, _Stage3Mask()),
    ]

    print("Phase 1: collect union of halo triples in wedge for staged mocks...", flush=True)
    needed = collect_needed_triples(
        stage_specs,
        wedge=wedge,
        z_col=args.redshift_col,
        chunk_size=args.chunk_size,
    )
    print(f"  needed triples: {len(needed):,}", flush=True)

    print("Phase 2: annotated pass (wedge truth + eig lookup)...", flush=True)
    needed_packed = triples_to_packed_set(needed)
    truth_arrays, lookup = scan_annotated_wedge_truth(
        args.annotated_fits,
        needed_packed,
        wedge=wedge,
        z_col=args.redshift_col,
        chunk_size=args.chunk_size,
        eig_thr=args.eig_threshold,
        write_xyz=args.write_xyz,
    )

    meta_truth = {
        "stage": "annotated_wedge_truth",
        "wedge": wedge,
        "redshift_col": args.redshift_col,
        "eig_threshold": args.eig_threshold,
        "cosmology": "Planck18",
        "class_fractions": _class_fractions(truth_arrays.get("cls", np.array([], dtype=np.int8))),
        "n_gal": int(truth_arrays.get("ra", np.array([])).size),
        "annotated_fits": str(args.annotated_fits),
    }
    out_truth = args.out_dir / "staged_mock_wedge_truth_annotated_rs7.npz"
    save_npz(out_truth, truth_arrays, meta_truth)
    print(
        f"  annotated wedge truth class fractions: {meta_truth['class_fractions']}",
        flush=True,
    )

    outputs = [
        ("staged_mock_wedge_stage1_rmaglt19p5_rs7.npz", args.stage1_fits, None, False),
        ("staged_mock_wedge_forFA_rs7.npz", args.forfa_fits, None, False),
        (
            "staged_mock_wedge_stage3_postcollision_rs7.npz",
            args.stage3_fits,
            _Stage3Mask(),
            True,
        ),
    ]

    print("Phase 3: build stage NPZs with eigenvalue join...", flush=True)
    all_stats = []
    for fname, fpath, mask_fn, dedupe in outputs:
        arrays, stats = extract_stage_npz(
            fpath,
            lookup,
            stage_name=fname,
            wedge=wedge,
            z_col=args.redshift_col,
            chunk_size=args.chunk_size,
            eig_thr=args.eig_threshold,
            write_xyz=args.write_xyz,
            extra_mask_fn=mask_fn,
            dedupe_triple=dedupe,
        )
        meta = {
            "wedge": wedge,
            "redshift_col": args.redshift_col,
            "eig_threshold": args.eig_threshold,
            "cosmology": "Planck18",
            **stats,
            "source_fits": str(fpath),
        }
        save_npz(args.out_dir / fname, arrays, meta)
        all_stats.append(stats)
        print(
            f"  {fname}: n_gal={arrays['ra'].size:,} join_miss_rate={stats['join_miss_rate']:.6f} "
            f"class_fractions={stats['class_fractions']}",
            flush=True,
        )

    summary_path = args.out_dir / "staged_mock_wedge_pipeline_summary.json"
    summary = {
        "outputs": {
            str(out_truth): meta_truth,
            **{
                str(args.out_dir / o[0]): s
                for o, s in zip(outputs, all_stats, strict=True)
            },
        },
        "needed_triples": len(needed),
        "lookup_size": len(lookup),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
