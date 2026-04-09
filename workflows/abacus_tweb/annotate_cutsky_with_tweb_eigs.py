#!/usr/bin/env python3
"""
Annotate Abacus CutSky mock galaxies with T-Web eigenvalues using halo linkage.

Two-step mapping implemented here:
1) Use (FILE_NUM, HALO_INDEX) from the CutSky row to recover host-halo box-frame
   position from Abacus base halo_info files.
2) Convert host-halo (x, y, z) to T-Web voxel indices and assign
   CWEB / LAMBDA1 / LAMBDA2 / LAMBDA3 from slabwise T-Web outputs.

This avoids sky-coordinate inversion/modulo mapping for label assignment.
"""

from __future__ import annotations

import argparse
import glob
import gc
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import fitsio
import numpy as np

# Allow canonical workflow scripts to resolve repo-root modules after reorganization.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config_paths import ABACUS_BASE, ABACUS_MOCKS_WITH_EIGS_DIR, ABACUS_TWEB_OUTPUT_DIR, CUTSKY_Z0200_PATH


DEFAULT_CUTSKY = CUTSKY_Z0200_PATH
DEFAULT_TWEB_DIR = ABACUS_TWEB_OUTPUT_DIR
DEFAULT_OUTPUT_DIR = ABACUS_MOCKS_WITH_EIGS_DIR
DEFAULT_HALO_INFO_DIR = f"{ABACUS_BASE}/halos/z0.200/halo_info"


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


@dataclass(frozen=True)
class TempMemmaps:
    file_num: np.memmap
    halo_index: np.memmap
    slab_id: np.memmap
    lix: np.memmap
    iy: np.memmap
    iz: np.memmap
    cweb: np.memmap
    lam1: np.memmap
    lam2: np.memmap
    lam3: np.memmap


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cutsky", default=DEFAULT_CUTSKY, help="Input CutSky FITS path.")
    p.add_argument("--tweb-dir", default=DEFAULT_TWEB_DIR, help="Directory with abacus_cactus_tweb_rank*.npz.")
    p.add_argument("--halo-info-dir", default=DEFAULT_HALO_INFO_DIR, help="Directory with halo_info_XXX.asdf files.")
    p.add_argument(
        "--halo-pos-field",
        default="x_com",
        choices=("x_com", "x_L2com"),
        help="Halo position field to use from halo_info.",
    )
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to write annotated FITS output.")
    p.add_argument("--output-name", default=None, help="Output FITS filename. Default: <stem>_with_tweb_eigs.fits")
    p.add_argument("--chunk-size", type=int, default=1_000_000, help="Rows per chunk for streaming passes.")
    p.add_argument("--temp-dir", default=None, help="Optional temporary directory.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output.")
    p.add_argument("--keep-temp", action="store_true", help="Keep temporary memmap files.")
    return p.parse_args()


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
    return [
        SlabMeta(
            slab_id=i,
            path=s.path,
            x_start=s.x_start,
            x_end=s.x_end,
            ngrid=s.ngrid,
            boxsize=s.boxsize,
            threshold=s.threshold,
            rsmooth=s.rsmooth,
        )
        for i, s in enumerate(slabs)
    ]


def validate_and_build_maps(slabs: list[SlabMeta]) -> tuple[np.ndarray, np.ndarray, int, float]:
    ngrid_set = {s.ngrid for s in slabs}
    box_set = {s.boxsize for s in slabs}
    thr_set = {s.threshold for s in slabs}
    rsm_set = {s.rsmooth for s in slabs}
    if len(ngrid_set) != 1 or len(box_set) != 1:
        raise ValueError("Inconsistent ngrid/boxsize across T-Web slab files.")
    if len(thr_set) != 1 or len(rsm_set) != 1:
        raise ValueError("Inconsistent threshold/Rsmooth across T-Web slab files.")

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

    if expected != ngrid or np.any(ix_to_slab < 0):
        raise ValueError("Slab coverage is incomplete.")
    return ix_to_slab, slab_xstart, ngrid, boxsize


def to_grid_indices(xyz: np.ndarray, ngrid: int, boxsize: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cell = boxsize / ngrid
    xyz_mod = np.mod(xyz, boxsize)
    ix = np.floor(xyz_mod[:, 0] / cell).astype(np.int32)
    iy = np.floor(xyz_mod[:, 1] / cell).astype(np.int32)
    iz = np.floor(xyz_mod[:, 2] / cell).astype(np.int32)
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


def create_temp_memmaps(temp_dir: str, nrows: int) -> TempMemmaps:
    mm = TempMemmaps(
        file_num=np.memmap(os.path.join(temp_dir, "file_num.int16"), mode="w+", dtype=np.int16, shape=(nrows,)),
        halo_index=np.memmap(os.path.join(temp_dir, "halo_index.int32"), mode="w+", dtype=np.int32, shape=(nrows,)),
        slab_id=np.memmap(os.path.join(temp_dir, "slab_id.int16"), mode="w+", dtype=np.int16, shape=(nrows,)),
        lix=np.memmap(os.path.join(temp_dir, "lix.uint16"), mode="w+", dtype=np.uint16, shape=(nrows,)),
        iy=np.memmap(os.path.join(temp_dir, "iy.uint16"), mode="w+", dtype=np.uint16, shape=(nrows,)),
        iz=np.memmap(os.path.join(temp_dir, "iz.uint16"), mode="w+", dtype=np.uint16, shape=(nrows,)),
        cweb=np.memmap(os.path.join(temp_dir, "cweb.uint8"), mode="w+", dtype=np.uint8, shape=(nrows,)),
        lam1=np.memmap(os.path.join(temp_dir, "lambda1.float32"), mode="w+", dtype=np.float32, shape=(nrows,)),
        lam2=np.memmap(os.path.join(temp_dir, "lambda2.float32"), mode="w+", dtype=np.float32, shape=(nrows,)),
        lam3=np.memmap(os.path.join(temp_dir, "lambda3.float32"), mode="w+", dtype=np.float32, shape=(nrows,)),
    )
    mm.slab_id[:] = -1
    mm.cweb[:] = 255
    mm.lam1[:] = np.nan
    mm.lam2[:] = np.nan
    mm.lam3[:] = np.nan
    mm.slab_id.flush()
    mm.cweb.flush()
    mm.lam1.flush()
    mm.lam2.flush()
    mm.lam3.flush()
    return mm


def pass1_collect_linkage_indices(hdu, nrows: int, chunk_size: int, mm: TempMemmaps) -> None:
    print("Pass 1/4: collecting FILE_NUM and HALO_INDEX from CutSky...")
    for start in range(0, nrows, chunk_size):
        stop = min(start + chunk_size, nrows)
        chunk = hdu[start:stop]
        mm.file_num[start:stop] = np.asarray(chunk["FILE_NUM"], dtype=np.int16)
        mm.halo_index[start:stop] = np.asarray(chunk["HALO_INDEX"], dtype=np.int32)
        if start == 0 or ((start // chunk_size) + 1) % 10 == 0 or stop == nrows:
            print(f"  collected linkage rows {start:,}-{stop:,} / {nrows:,}")
    mm.file_num.flush()
    mm.halo_index.flush()


def pass2_map_halo_to_grid(
    mm: TempMemmaps,
    nrows: int,
    halo_info_dir: str,
    halo_pos_field: str,
    ix_to_slab: np.ndarray,
    slab_xstart: np.ndarray,
    ngrid: int,
    boxsize: float,
) -> None:
    print("Pass 2/4: mapping host halos to T-Web voxel indices...")
    try:
        from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
    except Exception as e:
        raise RuntimeError(
            "abacusnbody (abacusutils) is required for halo linkage mode. "
            "Activate an environment with abacusnbody installed."
        ) from e

    file_nums = np.unique(mm.file_num)
    file_nums = file_nums[file_nums >= 0]
    mapped_rows = 0
    skipped_rows = 0

    for i, fn in enumerate(file_nums):
        row_idx = np.nonzero(mm.file_num == fn)[0]
        if row_idx.size == 0:
            continue

        halo_path = os.path.join(halo_info_dir, f"halo_info_{int(fn):03d}.asdf")
        if not os.path.exists(halo_path):
            print(f"  warning: missing {halo_path}; skipping {row_idx.size:,} rows")
            skipped_rows += int(row_idx.size)
            continue

        print(f"  file_num={int(fn):02d} ({i+1}/{len(file_nums)}), rows={row_idx.size:,}")
        cat = CompaSOHaloCatalog(
            halo_path,
            fields=[halo_pos_field],
            subsamples=False,
            convert_units=True,
            verbose=False,
            cleaned=False
        )
        halo_pos = np.asarray(cat.halos[halo_pos_field], dtype=np.float32)  # [Nhalo, 3]
        n_halos = halo_pos.shape[0]

        hidx = mm.halo_index[row_idx].astype(np.int64)
        valid = (hidx >= 0) & (hidx < n_halos)
        if not np.any(valid):
            skipped_rows += int(row_idx.size)
            continue

        valid_rows = row_idx[valid]
        pos = halo_pos[hidx[valid]]
        ix, iy, iz = to_grid_indices(pos, ngrid=ngrid, boxsize=boxsize)
        slab_ids = ix_to_slab[ix]
        local_ix = ix - slab_xstart[slab_ids]

        mm.slab_id[valid_rows] = slab_ids.astype(np.int16)
        mm.lix[valid_rows] = local_ix.astype(np.uint16)
        mm.iy[valid_rows] = iy.astype(np.uint16)
        mm.iz[valid_rows] = iz.astype(np.uint16)

        mapped_rows += int(valid_rows.size)
        skipped_rows += int(row_idx.size - valid_rows.size)

        del cat, halo_pos, hidx, valid, valid_rows, pos, ix, iy, iz, slab_ids, local_ix, row_idx
        gc.collect()

    mm.slab_id.flush()
    mm.lix.flush()
    mm.iy.flush()
    mm.iz.flush()
    print(f"  mapped rows: {mapped_rows:,} / {nrows:,}; skipped rows: {skipped_rows:,}")


def pass3_assign_tweb_values(slabs: list[SlabMeta], mm: TempMemmaps) -> None:
    print("Pass 3/4: assigning CWEB/eigenvalues from T-Web slabs...")
    for s in slabs:
        row_idx = np.nonzero(mm.slab_id == s.slab_id)[0]
        if row_idx.size == 0:
            continue

        print(
            f"  slab {s.slab_id:02d}: rows={row_idx.size:,}, "
            f"x=[{s.x_start},{s.x_end}), file={os.path.basename(s.path)}"
        )

        with np.load(s.path) as d:
            cweb_local = d["cweb"]
            eig_local = d["eig_vals"]
            li = mm.lix[row_idx].astype(np.int64)
            yj = mm.iy[row_idx].astype(np.int64)
            zk = mm.iz[row_idx].astype(np.int64)

            mm.cweb[row_idx] = cweb_local[li, yj, zk]
            mm.lam1[row_idx] = eig_local[0, li, yj, zk]
            mm.lam2[row_idx] = eig_local[1, li, yj, zk]
            mm.lam3[row_idx] = eig_local[2, li, yj, zk]

        mm.cweb.flush()
        mm.lam1.flush()
        mm.lam2.flush()
        mm.lam3.flush()
        del row_idx
        gc.collect()


def pass4_write_augmented_fits(hdu, nrows: int, chunk_size: int, out_path: str, mm: TempMemmaps) -> None:
    print(f"Pass 4/4: writing output FITS to {out_path}")
    fout = fitsio.FITS(out_path, "rw", clobber=True)
    first = True
    for start in range(0, nrows, chunk_size):
        stop = min(start + chunk_size, nrows)
        chunk = hdu[start:stop]
        out_chunk = make_augmented_chunk(
            chunk=chunk,
            cweb=mm.cweb[start:stop],
            l1=mm.lam1[start:stop],
            l2=mm.lam2[start:stop],
            l3=mm.lam3[start:stop],
        )
        if first:
            fout.write(out_chunk)
            first = False
        else:
            fout[-1].append(out_chunk)

        if start == 0 or ((start // chunk_size) + 1) % 10 == 0 or stop == nrows:
            print(f"  wrote rows {start:,}-{stop:,} / {nrows:,}")
    fout.close()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    os.makedirs(args.output_dir, exist_ok=True)

    in_name = os.path.basename(args.cutsky)
    stem = in_name[:-5] if in_name.endswith(".fits") else in_name
    out_name = args.output_name or f"{stem}_with_tweb_eigs.fits"
    out_path = os.path.join(args.output_dir, out_name)
    if os.path.exists(out_path):
        if args.overwrite:
            os.remove(out_path)
        else:
            raise FileExistsError(f"Output exists: {out_path}. Use --overwrite to replace it.")

    temp_dir = args.temp_dir or os.path.join(args.output_dir, f"tmp_tweb_eigs_{int(time.time())}")
    os.makedirs(temp_dir, exist_ok=True)

    print("Discovering T-Web slabs...")
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
    print(f"Halo linkage source: {args.halo_info_dir}")
    print(f"Halo position field: {args.halo_pos_field}")

    mm = create_temp_memmaps(temp_dir=temp_dir, nrows=nrows)
    pass1_collect_linkage_indices(hdu=hdu, nrows=nrows, chunk_size=args.chunk_size, mm=mm)
    pass2_map_halo_to_grid(
        mm=mm,
        nrows=nrows,
        halo_info_dir=args.halo_info_dir,
        halo_pos_field=args.halo_pos_field,
        ix_to_slab=ix_to_slab,
        slab_xstart=slab_xstart,
        ngrid=ngrid,
        boxsize=boxsize,
    )
    pass3_assign_tweb_values(slabs=slabs, mm=mm)
    pass4_write_augmented_fits(hdu=hdu, nrows=nrows, chunk_size=args.chunk_size, out_path=out_path, mm=mm)
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

