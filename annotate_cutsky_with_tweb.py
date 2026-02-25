#!/usr/bin/env python3
"""
Annotate AbacusSummit CutSky mock galaxies with T-Web class/eigenvalues.

This script:
1) Streams the CutSky FITS in chunks and computes periodic base-box grid indices.
2) Stores per-row slab/grid lookup indices in temporary memmaps.
3) Loads each T-Web rank file once, gathers cweb/eigenvalues for matching rows.
4) Writes a new FITS catalog with appended columns:
   - CWEB (uint8)
   - LAMBDA1, LAMBDA2, LAMBDA3 (float32)

Default paths come from `config_paths.py` and can be overridden via env vars:
- `TNG_CUTSKY_Z0200_PATH`
- `TNG_ABACUS_TWEB_OUTPUT_DIR`
- `TNG_ABACUS_MOCKS_WITH_EIGS_DIR`
"""

from __future__ import annotations

import argparse
import glob
import gc
import os
import shutil
import time
from dataclasses import dataclass

import fitsio
import numpy as np
from astropy.cosmology import Planck18 as cosmo
from config_paths import ABACUS_MOCKS_WITH_EIGS_DIR, ABACUS_TWEB_OUTPUT_DIR, CUTSKY_Z0200_PATH

# Workflow status: ACTIVE (CutSky annotation with slabwise T-Web outputs)


DEFAULT_CUTSKY = CUTSKY_Z0200_PATH
DEFAULT_TWEB_DIR = ABACUS_TWEB_OUTPUT_DIR
DEFAULT_OUTPUT_DIR = ABACUS_MOCKS_WITH_EIGS_DIR

OBSERVER_ORIGIN = np.array([-990.0, -990.0, -990.0], dtype=np.float64)


@dataclass(frozen=True)
class SlabMeta:
    slab_id: int
    path: str
    x_start: int
    x_end: int
    ngrid: int
    boxsize: float
    threshold: float
    rsmooth: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append T-Web class/eigenvalues to a CutSky FITS catalog."
    )
    parser.add_argument("--cutsky", default=DEFAULT_CUTSKY, help="Input CutSky FITS path.")
    parser.add_argument(
        "--tweb-dir",
        default=DEFAULT_TWEB_DIR,
        help="Directory containing abacus_cactus_tweb_rank*.npz files.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write annotated FITS output.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output FITS filename. Default: <input_stem>_with_tweb.fits",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Rows per chunk when reading/writing FITS.",
    )
    parser.add_argument(
        "--temp-dir",
        default=None,
        help="Optional temporary directory. Default: <output-dir>/tmp_tweb_<timestamp>",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output FITS if it already exists.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary memmap files after successful completion.",
    )
    return parser.parse_args()


def discover_slabs(tweb_dir: str) -> list[SlabMeta]:
    pattern = os.path.join(tweb_dir, "abacus_cactus_tweb_rank*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No T-Web rank files found with pattern: {pattern}")

    slabs: list[SlabMeta] = []
    for i, path in enumerate(files):
        with np.load(path) as d:
            slabs.append(
                SlabMeta(
                    slab_id=i,
                    path=path,
                    x_start=int(d["x_start"]),
                    x_end=int(d["x_end"]),
                    ngrid=int(d["ngrid"]),
                    boxsize=float(d["boxsize"]),
                    threshold=float(d["threshold"]),
                    rsmooth=float(d["Rsmooth"]),
                )
            )

    slabs = sorted(slabs, key=lambda s: s.x_start)
    for new_id, slab in enumerate(slabs):
        slabs[new_id] = SlabMeta(
            slab_id=new_id,
            path=slab.path,
            x_start=slab.x_start,
            x_end=slab.x_end,
            ngrid=slab.ngrid,
            boxsize=slab.boxsize,
            threshold=slab.threshold,
            rsmooth=slab.rsmooth,
        )
    return slabs


def validate_and_build_maps(slabs: list[SlabMeta]) -> tuple[np.ndarray, np.ndarray, int, float]:
    ngrid_set = {s.ngrid for s in slabs}
    box_set = {s.boxsize for s in slabs}
    thr_set = {s.threshold for s in slabs}
    rsm_set = {s.rsmooth for s in slabs}
    if len(ngrid_set) != 1 or len(box_set) != 1:
        raise ValueError("Inconsistent ngrid/boxsize across T-Web rank files.")
    if len(thr_set) != 1 or len(rsm_set) != 1:
        raise ValueError("Inconsistent threshold/Rsmooth across T-Web rank files.")

    ngrid = next(iter(ngrid_set))
    boxsize = next(iter(box_set))

    ix_to_slab = np.full(ngrid, -1, dtype=np.int16)
    slab_xstart = np.full(len(slabs), -1, dtype=np.int32)

    expected = 0
    for slab in slabs:
        if slab.x_start != expected:
            raise ValueError(
                f"Slab coverage gap/overlap near x={expected}; got slab starting at {slab.x_start}"
            )
        if slab.x_end <= slab.x_start:
            raise ValueError(f"Invalid slab range [{slab.x_start}, {slab.x_end}) in {slab.path}")
        ix_to_slab[slab.x_start : slab.x_end] = slab.slab_id
        slab_xstart[slab.slab_id] = slab.x_start
        expected = slab.x_end

    if expected != ngrid:
        raise ValueError(f"Slab coverage ends at {expected}, expected ngrid={ngrid}")
    if np.any(ix_to_slab < 0):
        raise ValueError("Some ix cells are not covered by any slab.")

    return ix_to_slab, slab_xstart, ngrid, boxsize


def sky_to_box_coords(ra_deg: np.ndarray, dec_deg: np.ndarray, z_cosmo: np.ndarray, boxsize: float):
    dist = cosmo.comoving_distance(z_cosmo).value * cosmo.h
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)

    x_obs = dist * np.cos(dec) * np.cos(ra)
    y_obs = dist * np.cos(dec) * np.sin(ra)
    z_obs = dist * np.sin(dec)

    x = (x_obs + OBSERVER_ORIGIN[0]) % boxsize
    y = (y_obs + OBSERVER_ORIGIN[1]) % boxsize
    z = (z_obs + OBSERVER_ORIGIN[2]) % boxsize
    return x, y, z


def to_grid_indices(x: np.ndarray, y: np.ndarray, z: np.ndarray, ngrid: int, boxsize: float):
    cell = boxsize / ngrid
    ix = np.floor(x / cell).astype(np.int32)
    iy = np.floor(y / cell).astype(np.int32)
    iz = np.floor(z / cell).astype(np.int32)
    np.clip(ix, 0, ngrid - 1, out=ix)
    np.clip(iy, 0, ngrid - 1, out=iy)
    np.clip(iz, 0, ngrid - 1, out=iz)
    return ix, iy, iz


def make_augmented_chunk(
    chunk: np.ndarray,
    cweb: np.ndarray,
    l1: np.ndarray,
    l2: np.ndarray,
    l3: np.ndarray,
) -> np.ndarray:
    new_dtype = chunk.dtype.descr + [
        ("CWEB", "u1"),
        ("LAMBDA1", "f4"),
        ("LAMBDA2", "f4"),
        ("LAMBDA3", "f4"),
    ]
    out = np.empty(chunk.shape, dtype=new_dtype)
    for name in chunk.dtype.names:
        out[name] = chunk[name]
    out["CWEB"] = cweb
    out["LAMBDA1"] = l1
    out["LAMBDA2"] = l2
    out["LAMBDA3"] = l3
    return out


def main() -> None:
    args = parse_args()
    t0 = time.time()

    os.makedirs(args.output_dir, exist_ok=True)

    in_name = os.path.basename(args.cutsky)
    stem = in_name[:-5] if in_name.endswith(".fits") else in_name
    out_name = args.output_name or f"{stem}_with_tweb.fits"
    out_path = os.path.join(args.output_dir, out_name)

    if os.path.exists(out_path):
        if args.overwrite:
            os.remove(out_path)
        else:
            raise FileExistsError(f"Output exists: {out_path}. Use --overwrite to replace it.")

    temp_dir = args.temp_dir or os.path.join(args.output_dir, f"tmp_tweb_{int(time.time())}")
    os.makedirs(temp_dir, exist_ok=True)

    print("Discovering T-Web rank files...")
    slabs = discover_slabs(args.tweb_dir)
    ix_to_slab, slab_xstart, ngrid, boxsize = validate_and_build_maps(slabs)
    print(
        f"Found {len(slabs)} slabs, ngrid={ngrid}, boxsize={boxsize:.1f}, "
        f"threshold={slabs[0].threshold}, Rsmooth={slabs[0].rsmooth}"
    )

    print("Opening CutSky input...")
    fin = fitsio.FITS(args.cutsky, "r")
    hdu = fin[1]
    nrows = hdu.get_nrows()
    print(f"Input rows: {nrows:,}")

    # Temporary memmaps for per-row lookup indices and outputs.
    slab_id_mm = np.memmap(os.path.join(temp_dir, "slab_id.uint8"), mode="w+", dtype=np.uint8, shape=(nrows,))
    lix_mm = np.memmap(os.path.join(temp_dir, "lix.uint16"), mode="w+", dtype=np.uint16, shape=(nrows,))
    iy_mm = np.memmap(os.path.join(temp_dir, "iy.uint16"), mode="w+", dtype=np.uint16, shape=(nrows,))
    iz_mm = np.memmap(os.path.join(temp_dir, "iz.uint16"), mode="w+", dtype=np.uint16, shape=(nrows,))

    cweb_mm = np.memmap(os.path.join(temp_dir, "cweb.uint8"), mode="w+", dtype=np.uint8, shape=(nrows,))
    lam1_mm = np.memmap(os.path.join(temp_dir, "lambda1.float32"), mode="w+", dtype=np.float32, shape=(nrows,))
    lam2_mm = np.memmap(os.path.join(temp_dir, "lambda2.float32"), mode="w+", dtype=np.float32, shape=(nrows,))
    lam3_mm = np.memmap(os.path.join(temp_dir, "lambda3.float32"), mode="w+", dtype=np.float32, shape=(nrows,))

    # Pass 1: Build per-row slab lookup indices.
    print("Pass 1/3: computing periodic grid indices for all galaxies...")
    for start in range(0, nrows, args.chunk_size):
        stop = min(start + args.chunk_size, nrows)
        chunk = hdu[start:stop]
        ra = np.asarray(chunk["RA"], dtype=np.float64)
        dec = np.asarray(chunk["DEC"], dtype=np.float64)
        zc = np.asarray(chunk["Z_COSMO"], dtype=np.float64)

        x, y, z = sky_to_box_coords(ra, dec, zc, boxsize=boxsize)
        ix, iy, iz = to_grid_indices(x, y, z, ngrid=ngrid, boxsize=boxsize)

        slab_ids = ix_to_slab[ix]
        if np.any(slab_ids < 0):
            raise RuntimeError(f"Found unmapped slab ids in rows {start}:{stop}")

        local_ix = ix - slab_xstart[slab_ids]
        if np.any(local_ix < 0):
            raise RuntimeError(f"Found negative local_ix in rows {start}:{stop}")

        slab_id_mm[start:stop] = slab_ids.astype(np.uint8)
        lix_mm[start:stop] = local_ix.astype(np.uint16)
        iy_mm[start:stop] = iy.astype(np.uint16)
        iz_mm[start:stop] = iz.astype(np.uint16)

        if start == 0 or ((start // args.chunk_size) + 1) % 10 == 0 or stop == nrows:
            print(f"  indexed rows {start:,}-{stop:,} / {nrows:,}")

    slab_id_mm.flush()
    lix_mm.flush()
    iy_mm.flush()
    iz_mm.flush()

    # Pass 2: Load each slab once and gather cweb/eigenvalues for matching rows.
    print("Pass 2/3: assigning cweb/eigenvalues from slab files...")
    for slab in slabs:
        row_idx = np.nonzero(slab_id_mm == slab.slab_id)[0]
        if row_idx.size == 0:
            print(f"  slab {slab.slab_id:02d}: no rows, skipping")
            continue

        print(
            f"  slab {slab.slab_id:02d}: rows={row_idx.size:,}, "
            f"x=[{slab.x_start},{slab.x_end}), loading {os.path.basename(slab.path)}"
        )

        with np.load(slab.path) as d:
            cweb_local = d["cweb"]
            eig_local = d["eig_vals"]

            li = lix_mm[row_idx].astype(np.int64)
            yj = iy_mm[row_idx].astype(np.int64)
            zk = iz_mm[row_idx].astype(np.int64)

            cweb_mm[row_idx] = cweb_local[li, yj, zk]
            lam1_mm[row_idx] = eig_local[0, li, yj, zk]
            lam2_mm[row_idx] = eig_local[1, li, yj, zk]
            lam3_mm[row_idx] = eig_local[2, li, yj, zk]

        cweb_mm.flush()
        lam1_mm.flush()
        lam2_mm.flush()
        lam3_mm.flush()
        del row_idx
        gc.collect()

    # Pass 3: Write final FITS (same original columns + new T-Web columns).
    print(f"Pass 3/3: writing output FITS to {out_path}")
    fout = fitsio.FITS(out_path, "rw", clobber=True)
    first = True
    for start in range(0, nrows, args.chunk_size):
        stop = min(start + args.chunk_size, nrows)
        chunk = hdu[start:stop]
        out_chunk = make_augmented_chunk(
            chunk=chunk,
            cweb=cweb_mm[start:stop],
            l1=lam1_mm[start:stop],
            l2=lam2_mm[start:stop],
            l3=lam3_mm[start:stop],
        )
        if first:
            fout.write(out_chunk)
            first = False
        else:
            fout[-1].append(out_chunk)

        if start == 0 or ((start // args.chunk_size) + 1) % 10 == 0 or stop == nrows:
            print(f"  wrote rows {start:,}-{stop:,} / {nrows:,}")

    fout.close()
    fin.close()

    print("Done.")
    print(f"Output: {out_path}")
    print(f"Elapsed: {(time.time() - t0) / 60.0:.2f} min")

    if args.keep_temp:
        print(f"Temporary files kept at: {temp_dir}")
    else:
        print(f"Removing temporary files: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
