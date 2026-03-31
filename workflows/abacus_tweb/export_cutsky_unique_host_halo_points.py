#!/usr/bin/env python3
"""Export unique host-halo box-frame positions for a CutSky mock.

This implements "option 2" validation:

- Filter CutSky galaxies by (IN_Y1 == 1) | (IN_Y5 == 1) and BOX_INDEX != -1
- Build a *unique* set of host halos keyed by (FILE_NUM, BOX_INDEX)
- Load halo positions (x_com or x_L2com) from the corresponding CompaSO
  halo_info_<FILE_NUM>.asdf files, indexing by BOX_INDEX
- Save the resulting halo positions as a points array suitable as Gudhi input

Outputs:
- points.npy: float64, shape [N_halo, 3]
- keys.npy:   int32/int64, shape [N_halo, 2] columns [FILE_NUM, BOX_INDEX]

Notes:
- This exports halo positions, not per-galaxy positions. Multiple galaxies can
  share a halo; uniqueness is enforced on (FILE_NUM, BOX_INDEX).
- Abacus stores positions in [-L/2, L/2) in some products; we wrap to [0, L)
  when --wrap-box is enabled (default).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cutsky",
        required=True,
        help="CutSky FITS path (must contain FILE_NUM, BOX_INDEX, IN_Y1/IN_Y5).",
    )
    p.add_argument(
        "--halo-info-dir",
        required=True,
        help="Directory containing halo_info_XXX.asdf (CompaSO halos).",
    )
    p.add_argument(
        "--halo-pos-field",
        choices=("x_com", "x_L2com"),
        default="x_com",
        help="Which halo position field to export.",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=2_000_000,
        help="Rows per FITS chunk when streaming.",
    )
    p.add_argument("--out-points-npy", required=True, help="Output points .npy (float64, [N,3]).")
    p.add_argument("--out-keys-npy", required=True, help="Output keys .npy ([N,2] = FILE_NUM,BOX_INDEX).")
    p.add_argument("--meta-json", default=None, help="Optional metadata JSON sidecar.")
    p.add_argument(
        "--wrap-box",
        action="store_true",
        default=True,
        help="Wrap positions to [0, boxsize) via modulo (default: true).",
    )
    p.add_argument(
        "--no-wrap-box",
        dest="wrap_box",
        action="store_false",
        help="Disable modulo wrap of positions.",
    )
    p.add_argument(
        "--boxsize",
        type=float,
        default=2000.0,
        help="Box size (Mpc/h) used when --wrap-box is enabled.",
    )
    return p.parse_args()


def _resolve_cols(names: list[str]) -> dict[str, str]:
    up = {n.upper(): n for n in names}

    def need(*cands: str) -> str:
        for c in cands:
            if c.upper() in up:
                return up[c.upper()]
        raise KeyError(f"None of {cands} present. Found columns: {names[:25]} ...")

    return {
        "FILE_NUM": need("FILE_NUM"),
        "BOX_INDEX": need("BOX_INDEX"),
        "IN_Y1": need("IN_Y1"),
        "IN_Y5": need("IN_Y5"),
    }


def _unique_pairs(file_num: np.ndarray, box_index: np.ndarray) -> np.ndarray:
    """Return unique pairs as a (N,2) array sorted by FILE_NUM then BOX_INDEX."""
    file_num = np.asarray(file_num, dtype=np.int32)
    box_index = np.asarray(box_index, dtype=np.int64)
    pairs = np.empty(file_num.shape[0], dtype=[("file_num", np.int32), ("box_index", np.int64)])
    pairs["file_num"] = file_num
    pairs["box_index"] = box_index
    uniq = np.unique(pairs)
    out = np.empty((uniq.shape[0], 2), dtype=np.int64)
    out[:, 0] = uniq["file_num"].astype(np.int64)
    out[:, 1] = uniq["box_index"].astype(np.int64)
    return out


def main() -> None:
    args = parse_args()
    t0 = time.time()

    import fitsio

    try:
        from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
    except Exception as e:
        raise RuntimeError("abacusnbody must be importable in this environment.") from e

    cutsky = Path(args.cutsky).expanduser().resolve()
    halo_dir = Path(args.halo_info_dir).expanduser().resolve()
    if not cutsky.exists():
        raise FileNotFoundError(cutsky)
    if not halo_dir.exists():
        raise FileNotFoundError(halo_dir)

    print(f"Streaming CutSky: {cutsky}")
    f = fitsio.FITS(str(cutsky), "r")
    hdu = f[1]
    nrows = hdu.get_nrows()
    cols = _resolve_cols(hdu.get_colnames())
    col_list = [cols["FILE_NUM"], cols["BOX_INDEX"], cols["IN_Y1"], cols["IN_Y5"]]

    file_chunks: list[np.ndarray] = []
    box_chunks: list[np.ndarray] = []
    kept_rows = 0

    for start in range(0, nrows, args.chunk_size):
        stop = min(start + args.chunk_size, nrows)
        chunk = hdu[start:stop][col_list]
        in_y1 = np.asarray(chunk[cols["IN_Y1"]]) == 1
        in_y5 = np.asarray(chunk[cols["IN_Y5"]]) == 1
        box = np.asarray(chunk[cols["BOX_INDEX"]], dtype=np.int64)
        m = (in_y1 | in_y5) & (box != -1)
        if np.any(m):
            file_chunks.append(np.asarray(chunk[cols["FILE_NUM"]][m], dtype=np.int32))
            box_chunks.append(box[m])
            kept_rows += int(np.sum(m))
        if start == 0 or (start // args.chunk_size + 1) % 10 == 0 or stop == nrows:
            print(f"  scanned rows {start:,}-{stop:,}/{nrows:,} | kept so far={kept_rows:,}")

    f.close()

    file_all = np.concatenate(file_chunks) if file_chunks else np.empty((0,), dtype=np.int32)
    box_all = np.concatenate(box_chunks) if box_chunks else np.empty((0,), dtype=np.int64)
    print(f"Filtered galaxy rows: {file_all.size:,}")

    keys = _unique_pairs(file_all, box_all)
    print(f"Unique (FILE_NUM, BOX_INDEX) halos: {keys.shape[0]:,}")
    del file_chunks, box_chunks, file_all, box_all

    # Load positions for each unique pair.
    points = np.empty((keys.shape[0], 3), dtype=np.float64)
    file_nums = keys[:, 0].astype(np.int32)
    box_idx = keys[:, 1].astype(np.int64)

    order = np.argsort(file_nums, kind="mergesort")
    file_sorted = file_nums[order]
    idx_sorted = box_idx[order]

    unique_files, starts = np.unique(file_sorted, return_index=True)
    starts = list(starts) + [len(file_sorted)]

    for uf, s0, s1 in zip(unique_files, starts[:-1], starts[1:]):
        hp = halo_dir / f"halo_info_{int(uf):03d}.asdf"
        if not hp.exists():
            raise FileNotFoundError(f"Missing halo_info file: {hp}")

        # Load minimal halo fields when supported.
        try:
            cat = CompaSOHaloCatalog(str(hp), cleaned=False, fields=[args.halo_pos_field])
        except (TypeError, ValueError):
            cat = CompaSOHaloCatalog(str(hp), cleaned=False)

        arr = np.asarray(cat.halos[args.halo_pos_field], dtype=np.float64)
        nh = arr.shape[0]
        hidx = idx_sorted[s0:s1]
        ok = (hidx >= 0) & (hidx < nh)
        if not np.all(ok):
            bad = int((~ok).sum())
            raise RuntimeError(
                f"FILE_NUM={int(uf):03d}: {bad} BOX_INDEX entries out of range (nhalo={nh})."
            )

        pts = arr[hidx]
        if args.wrap_box:
            pts = pts % float(args.boxsize)
        points[order[s0:s1]] = pts
        del cat, arr, pts
        print(f"  loaded file_num={int(uf):03d} halos={s1-s0:,}")

    out_points = Path(args.out_points_npy).expanduser().resolve()
    out_keys = Path(args.out_keys_npy).expanduser().resolve()
    out_points.parent.mkdir(parents=True, exist_ok=True)
    out_keys.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_points, points)
    np.save(out_keys, keys.astype(np.int64))
    print(f"Saved points: {out_points} shape={points.shape} dtype=float64")
    print(f"Saved keys:   {out_keys} shape={keys.shape} dtype=int64 [FILE_NUM, BOX_INDEX]")

    if args.meta_json:
        meta = {
            "cutsky": str(cutsky),
            "halo_info_dir": str(halo_dir),
            "filters": {"IN_Y1_or_IN_Y5": True, "BOX_INDEX_not_-1": True},
            "halo_key": "(FILE_NUM, BOX_INDEX)",
            "halo_pos_field": args.halo_pos_field,
            "wrap_box": bool(args.wrap_box),
            "boxsize": float(args.boxsize),
            "n_unique_halos": int(keys.shape[0]),
            "outputs": {"points_npy": str(out_points), "keys_npy": str(out_keys)},
            "elapsed_sec": float(time.time() - t0),
        }
        mp = Path(args.meta_json).expanduser().resolve()
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(meta, indent=2, sort_keys=True))
        print(f"Saved meta:   {mp}")


if __name__ == "__main__":
    main()

