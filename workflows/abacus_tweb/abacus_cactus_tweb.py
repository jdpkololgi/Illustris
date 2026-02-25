import os
import sys
from pathlib import Path
if __name__ == "__main__" and any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    print("usage: abacus_cactus_tweb.py [--help]\n\nRun MPI T-Web on rank-local density slabs.")
    raise SystemExit(0)

import numpy as np

from shift import mpiutils

# Allow canonical workflow scripts to resolve repo-root modules after reorganization.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config_paths import ABACUS_SLAB_DIR, ABACUS_TWEB_OUTPUT_DIR
from shared.resource_requirements import require_cpu_mpi_slurm

# Workflow status: ACTIVE (Abacus slab -> MPI T-Web rank outputs)

# Import the memory-optimized T-Web from our processing script
from abacus_process_particles2 import run_tweb_memory_optimized

SLAB_DIR = ABACUS_SLAB_DIR
OUTPUT_DIR = ABACUS_TWEB_OUTPUT_DIR

THRESHOLD = 0.2
RSMOOTH = 2.0


def load_local_density_from_slab_file(slab_dir, MPI):
    """
    Load rank-local density slab written by abacus_process_particles2.py.
    This avoids huge MPI object sends for multi-GB slabs.
    """
    rank, size = MPI.rank, MPI.size
    slab_path = f"{slab_dir}/abacus_z0200_density_slab_rank{rank:04d}.npz"
    if not os.path.exists(slab_path):
        raise FileNotFoundError(
            f"Rank {rank} missing slab file: {slab_path}. "
            f"Run density slab build with the same MPI size ({size}) first."
        )

    data = np.load(slab_path)
    dens_local = np.ascontiguousarray(data["dens"], dtype=np.float32)
    x_start = int(data["x_start"])
    x_end = int(data["x_end"])
    ngrid = int(data["ngrid"])
    boxsize = float(data["boxsize"])
    data.close()
    return dens_local, ngrid, boxsize, x_start, x_end


def estimate_peak_memory_gb(local_shape):
    """
    Rough per-rank peak memory estimate for run_tweb_memory_optimized.
    The memory-optimized version peaks at ~70 bytes/cell (during eigenvalue
    computation: 6 derivative arrays + 3 eigenvalue arrays = 9 float64 arrays).
    """
    nx, ny, nz = local_shape
    cells = nx * ny * nz
    # Conservative: ~70-90 bytes/cell accounting for temporaries
    min_gb = cells * 70 / (1024**3)
    max_gb = cells * 90 / (1024**3)
    return min_gb, max_gb


def main():
    require_cpu_mpi_slurm("abacus_cactus_tweb.py", min_tasks=2)
    MPI = mpiutils.MPI()
    rank = MPI.rank

    if rank == 0:
        print(f"MPI size: {MPI.size}")
        print(f"Loading density slabs from: {SLAB_DIR}")

    dens_local, ngrid, boxsize, x_start, x_end = load_local_density_from_slab_file(
        SLAB_DIR, MPI
    )

    if rank == 0:
        est_min_gb, est_max_gb = estimate_peak_memory_gb(dens_local.shape)
        print("Running memory-optimized MPI T-Web...")
        print(
            f"Rank-local slab shape: {dens_local.shape}. "
            f"Estimated peak memory per rank: ~{est_min_gb:.1f}-{est_max_gb:.1f} GiB."
        )

    cweb_local, eig_vals_local = run_tweb_memory_optimized(
        dens_local,
        boxsize,
        ngrid,
        THRESHOLD,
        MPI,
        Rsmooth=RSMOOTH,
        verbose=True,
    )

    if rank == 0 and not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    MPI.wait()

    outpath = f"{OUTPUT_DIR}/abacus_cactus_tweb_rank{rank:04d}.npz"
    np.savez(
        outpath,
        cweb=cweb_local.astype(np.uint8),
        eig_vals=eig_vals_local.astype(np.float32),
        x_start=x_start,
        x_end=x_end,
        ngrid=ngrid,
        boxsize=boxsize,
        threshold=THRESHOLD,
        Rsmooth=RSMOOTH,
    )

    print(f"[rank {rank}] Saved {outpath}")
    MPI.end()


if __name__ == "__main__":
    main()