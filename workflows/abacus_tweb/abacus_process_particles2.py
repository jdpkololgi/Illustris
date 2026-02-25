'''
AbacusSummit Particle Processing for T-Web Analysis

This script loads AbacusSummit particle data to build density fields for T-Web,
and matches mock galaxy catalogs to T-Web classifications.

KEY INSIGHT: The CutSky mock spans z=0-0.8 (built from light cone), but the
CubicBox mock is from the z=0.200 snapshot with direct (x,y,z) coordinates.
For T-Web, use the PERIODIC BOX particles at z=0.200 with the CubicBox mock.

Data Sources:
-------------
1. PERIODIC BOX PARTICLES (z=0.200 snapshot) - USE THIS FOR T-WEB:
   /global/cfs/cdirs/desi/public/cosmosim/AbacusSummit/AbacusSummit_base_c000_ph000/halos/z0.200/
   - field_rv_A/: 3% subsample of field particles (not in halos)
   - halo_rv_A/:  3% subsample of halo particles
   - NOTE: Only subsample A is available for snapshots (3% = ~10 billion particles)
   - The light cone has 10% (A+B), but snapshots only store A to save space
   - 3% is sufficient for density field construction at typical resolutions

2. CUBICBOX MOCK (same z=0.200, RECOMMENDED for T-Web matching):
   /global/cfs/cdirs/desi/cosmosim/SecondGenMocks/AbacusSummit/CubicBox/BGS/v0.1/z0.200/
   - Has direct (x, y, z) box coordinates in Mpc/h
   - Same HOD as CutSky, just in periodic box geometry

3. CUTSKY MOCK (DESI footprint, spans z=0-0.8):
   /global/cfs/cdirs/desi/cosmosim/SecondGenMocks/AbacusSummit/CutSky/BGS/v0.1/z0.200/
   - Has (RA, DEC, Z) sky coordinates
   - Built from light cone, needs conversion to box coords

4. LIGHT CONE PARTICLES (for reference, z >= 0.1 only):
   /global/cfs/cdirs/desi/public/cosmosim/AbacusSummit/AbacusSummit_base_c000_ph000/lightcones/

Workflow for T-Web:
-------------------
1. Load particles from z=0.200 snapshot (periodic box)
2. Build density field on 3D grid (CIC mass assignment)
3. Run T-Web (via cactus) → eigenvalues per cell
4. Load CubicBox mock galaxies → look up T-Web cell → assign eigenvalues

Box Geometry:
-------------
- BoxSize = 2000 Mpc/h
- Coordinates range: [0, 2000) Mpc/h in snapshot files
- Periodic boundary conditions
'''

import numpy as np
import glob
import asdf
import argparse
import sys
from pathlib import Path
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
import fitsio
from astropy.table import Table
from numba import njit

import cactus
from cactus.ext import fiesta

# abacusutils read_asdf automatically decompresses rvint data
from abacusnbody.data.read_abacus import read_asdf

# Allow canonical workflow scripts to resolve repo-root modules after reorganization.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config_paths import ABACUS_BASE as CFG_ABACUS_BASE
from config_paths import ABACUS_SLAB_DIR, MOCKS_BASE as CFG_MOCKS_BASE

# Workflow status: ACTIVE (canonical Abacus particle processing path)

# =============================================================================
# PATHS
# =============================================================================
ABACUS_BASE = CFG_ABACUS_BASE
MOCKS_BASE = CFG_MOCKS_BASE

# Light cone path (for reference)
LC_PATH = f'{ABACUS_BASE}/lightcones/'

# Snapshot paths (PERIODIC BOX - use these for T-Web)
SNAPSHOT_Z0200 = f'{ABACUS_BASE}/halos/z0.200'
SNAPSHOT_Z0100 = f'{ABACUS_BASE}/halos/z0.100'

# Mock paths
CUBICBOX_BGS = f'{MOCKS_BASE}/CubicBox/BGS/v0.1/z0.200/AbacusSummit_base_c000_ph000/BGS_box_ph000.fits'
CUTSKY_BGS = f'{MOCKS_BASE}/CutSky/BGS/v0.1/z0.200/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000.fits'

# Simulation parameters
BOXSIZE = 2000.0  # Mpc/h
H0 = 67.36
OMEGA_M = 0.315192
OBSERVER_ORIGIN = np.array([-990., -990., -990.])  # Mpc/h (for light cone)


# =============================================================================
# PERIODIC BOX PARTICLE LOADING (FOR T-WEB)
# =============================================================================

def load_snapshot_particles(snapshot_path=SNAPSHOT_Z0200, subsample='A',
                           particle_type='both', max_files=None, verbose=True):
    """
    Load particles from a periodic box snapshot for T-Web density field construction.

    Args:
        snapshot_path: Path to snapshot, e.g., '.../halos/z0.200'
        subsample: 'A' (3%) - only A is available for snapshots
        particle_type: 'field' (not in halos), 'halo', or 'both'
        max_files: Limit number of files PER DIRECTORY (for testing)
        verbose: Print progress

    Returns:
        pos: (N, 3) positions in Mpc/h, range [0, 2000)
        vel: (N, 3) velocities in km/s
        header: dict with metadata

    Example:
        # Load 3% subsample (field + halo particles) - ~10 billion particles
        pos, vel, header = load_snapshot_particles(SNAPSHOT_Z0200)

        # Load just field particles for testing
        pos, vel, header = load_snapshot_particles(SNAPSHOT_Z0200, particle_type='field', max_files=5)

    Note:
        - Only subsample A (3%) is stored for periodic box snapshots
        - Light cone has 10% (A+B), but snapshots only have A
        - 3% is sufficient for density fields at typical resolutions (256³-512³)
        - Full 330 billion particles are NOT stored to save disk space
    """
    all_pos = []
    all_vel = []

    # Determine which directories to load
    dirs_to_load = []
    if particle_type in ['field', 'both']:
        dirs_to_load.append(f'field_rv_{subsample}')
    if particle_type in ['halo', 'both']:
        dirs_to_load.append(f'halo_rv_{subsample}')

    header = None
    total_files = 0

    for dirname in dirs_to_load:
        dirpath = f'{snapshot_path}/{dirname}'
        files = sorted(glob.glob(f'{dirpath}/*.asdf'))

        if max_files:
            files = files[:max_files]

        if verbose:
            print(f"Loading {len(files)} files from {dirname}/...")

        for i, filepath in enumerate(files):
            # Use read_asdf which auto-decompresses
            data = read_asdf(filepath, verbose=False)

            if header is None:
                with asdf.open(filepath) as af:
                    header = dict(af.tree['header'])

            pos = np.array(data['pos'])
            vel = np.array(data['vel'])

            all_pos.append(pos)
            all_vel.append(vel)
            total_files += 1

            if verbose and (i + 1) % 10 == 0:
                print(f"  Loaded {i+1}/{len(files)} files ({len(pos):,} particles each)...")

    all_pos = np.vstack(all_pos)
    all_vel = np.vstack(all_vel)

    if verbose:
        print(f"\n{'='*50}")
        print(f"Loaded {total_files} files, {len(all_pos):,} particles total")
        print(f"Position range: x=[{all_pos[:,0].min():.1f}, {all_pos[:,0].max():.1f}] Mpc/h")
        print(f"Redshift: z = {header.get('Redshift', 'N/A')}")
        print(f"{'='*50}")

    return all_pos, all_vel, header


# =============================================================================
# MOCK CATALOG LOADING
# =============================================================================

def load_cubicbox_mock(filepath=CUBICBOX_BGS, columns=None, verbose=True):
    """
    Load CubicBox BGS mock with (x, y, z) box coordinates.
    USE THIS FOR T-WEB MATCHING - coordinates match the periodic box directly.

    Args:
        filepath: Path to FITS file
        columns: List of columns to load, or None for default set

    Returns:
        Table with galaxy properties including x, y, z positions in Mpc/h
    """
    if columns is None:
        columns = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'HALO_MASS', 'cen', 'R_MAG_ABS']

    if verbose:
        print(f"Loading CubicBox mock: {filepath.split('/')[-1]}")

    data = Table(fitsio.read(filepath, columns=columns))

    if verbose:
        print(f"  Loaded {len(data):,} galaxies")
        print(f"  Position range: x=[{data['x'].min():.1f}, {data['x'].max():.1f}] Mpc/h")

    return data


# =============================================================================
# DENSITY FIELD CONSTRUCTION
# =============================================================================

@njit
def _deposit_ngp(pos, density, cell_size, boxsize, ngrid):
    """Deposit particles onto grid using NGP (Numba-accelerated).

    Loops over particles in compiled code, replacing slow np.add.at.
    """
    for i in range(len(pos)):
        ix = int((pos[i, 0] % boxsize) / cell_size) % ngrid
        iy = int((pos[i, 1] % boxsize) / cell_size) % ngrid
        iz = int((pos[i, 2] % boxsize) / cell_size) % ngrid
        density[ix, iy, iz] += 1.0


@njit
def _deposit_cic(pos, density, cell_size, boxsize, ngrid):
    """Deposit particles onto grid using CIC (Numba-accelerated).

    Loops over particles in compiled code with trilinear weight distribution.
    """
    for i in range(len(pos)):
        px = (pos[i, 0] % boxsize) / cell_size
        py = (pos[i, 1] % boxsize) / cell_size
        pz = (pos[i, 2] % boxsize) / cell_size

        ix = int(px)
        iy = int(py)
        iz = int(pz)

        dx = px - ix
        dy = py - iy
        dz = pz - iz

        for ox in range(2):
            for oy in range(2):
                for oz in range(2):
                    wx = (1.0 - dx) if ox == 0 else dx
                    wy = (1.0 - dy) if oy == 0 else dy
                    wz = (1.0 - dz) if oz == 0 else dz
                    jx = (ix + ox) % ngrid
                    jy = (iy + oy) % ngrid
                    jz = (iz + oz) % ngrid
                    density[jx, jy, jz] += wx * wy * wz


def build_density_field_streaming(snapshot_path=SNAPSHOT_Z0200, ngrid=512,
                                   boxsize=BOXSIZE, method='NGP', verbose=True):
    """
    Build density field by streaming through particle files one at a time.
    Uses Numba JIT-compiled deposit functions for fast grid assignment.
    This avoids loading all ~10 billion particles into memory at once.

    Memory usage: ~ngrid^3 * 8 bytes for the grid (e.g., 512^3 = 1 GB)
    vs ~240 GB if loading all particles.

    Args:
        snapshot_path: Path to snapshot
        ngrid: Grid resolution (256, 512, etc.)
        boxsize: Box size in Mpc/h
        method: 'NGP' or 'CIC'
        verbose: Print progress

    Returns:
        density: (ngrid, ngrid, ngrid) raw particle count density field
                 (pass directly to cactus run_tweb, which normalizes internally)

    Example:
        # Build 512^3 density field from all particles
        density = build_density_field_streaming(SNAPSHOT_Z0200, ngrid=512)

        # Save for later use (like Illustris_cactus.py)
        np.savez('abacus_z0200_density_512.npz', dens=density)
    """
    density = np.zeros((ngrid, ngrid, ngrid), dtype=np.float64)
    cell_size = boxsize / ngrid
    total_particles = 0

    # Select the deposit function
    if method == 'NGP':
        deposit_func = _deposit_ngp
    elif method == 'CIC':
        deposit_func = _deposit_cic
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'NGP' or 'CIC'.")

    # Process both field and halo particles
    for particle_type in ['field', 'halo']:
        dirpath = f'{snapshot_path}/{particle_type}_rv_A'
        files = sorted(glob.glob(f'{dirpath}/*.asdf'))

        if verbose:
            print(f"\nProcessing {len(files)} {particle_type} files...")

        for i, filepath in enumerate(files):
            # Load particles from this file
            data = read_asdf(filepath, verbose=False)
            pos = np.ascontiguousarray(np.array(data['pos'], dtype=np.float64))
            n_particles = len(pos)
            total_particles += n_particles

            # Deposit onto grid (Numba-compiled, ~5-20x faster than np.add.at)
            deposit_func(pos, density, cell_size, boxsize, ngrid)

            # Free memory
            del pos, data

            if verbose:
                print(f"  [{i+1}/{len(files)}] {filepath.split('/')[-1]}: {n_particles:,} particles")

    if verbose:
        print(f"\n{'='*60}")
        print(f"DENSITY FIELD COMPLETE")
        print(f"  Total particles processed: {total_particles:,}")
        print(f"  Grid: {ngrid}^3 = {ngrid**3:,} cells")
        print(f"  Particles per cell: {total_particles / ngrid**3:.1f}")
        print(f"  Density range: [{density.min():.2f}, {density.max():.2f}]")
        print(f"{'='*60}")

    return density.astype(np.float32)

def build_density_field_fiesta(snapshot_path=SNAPSHOT_Z0200, ngrid=512,
                                boxsize=BOXSIZE, method='NGP', verbose=True):
    """
    Build density field by streaming through particle files using FIESTA part2grid3D.
    Uses cactus/FIESTA's Numba-JIT compiled grid assignment (same code cactus uses
    internally) for fast, accurate deposition supporting NGP, CIC, TSC, and PCS.

    Each ASDF file is loaded one at a time and its contribution is accumulated
    into the running density grid via +=, avoiding loading all particles at once.

    Memory usage: ~2 * ngrid^3 * 8 bytes (one persistent grid + one temporary per call).
    E.g., ngrid=512 → ~2 GB, ngrid=1000 → ~16 GB.

    Args:
        snapshot_path: Path to snapshot, e.g., '.../halos/z0.200'
        ngrid: Grid resolution (256, 512, 1000, etc.)
        boxsize: Box size in Mpc/h
        method: 'NGP', 'CIC', 'TSC', or 'PCS' (FIESTA supports all four)
        verbose: Print progress

    Returns:
        density: (ngrid, ngrid, ngrid) density field (float32)
                 Non-negative, suitable for passing directly to cactus run_tweb
                 (which normalizes internally via norm_dens).

    Example:
        # Build density field using FIESTA CIC
        density = build_density_field_fiesta(SNAPSHOT_Z0200, ngrid=512, method='CIC')

        # Run T-Web
        cweb, eig_vals = cactus.src.tweb.run_tweb(
            density, boxsize=2000., ngrid=512,
            threshold=0.2, Rsmooth=2., boundary='periodic'
        )
    """
    density = np.zeros((ngrid, ngrid, ngrid), dtype=np.float64)
    total_particles = 0
    particle_mass = None

    # Read particle mass from the header of the first available file
    for particle_type in ['field', 'halo']:
        dirpath = f'{snapshot_path}/{particle_type}_rv_A'
        files = sorted(glob.glob(f'{dirpath}/*.asdf'))
        if files and particle_mass is None:
            with asdf.open(files[0]) as af:
                header = dict(af.tree['header'])
                particle_mass = float(header['ParticleMassHMsun'])
                if verbose:
                    print(f"Particle mass from header: {particle_mass:.4e} Msun/h")
                    print(f"Box size: {boxsize} Mpc/h, ngrid: {ngrid}, method: {method}")
            break

    if particle_mass is None:
        raise FileNotFoundError(f"No ASDF files found in {snapshot_path}")

    # Process both field and halo particles
    for particle_type in ['field', 'halo']:
        dirpath = f'{snapshot_path}/{particle_type}_rv_A'
        files = sorted(glob.glob(f'{dirpath}/*.asdf'))

        if verbose:
            print(f"\nProcessing {len(files)} {particle_type} files...")

        for i, filepath in enumerate(files):
            data = read_asdf(filepath, verbose=False)
            pos = np.array(data['pos'])
            n_particles = len(pos)
            total_particles += n_particles

            # Wrap coordinates to [0, boxsize) for FIESTA with origin=0
            # (AbacusSummit may store positions in [-BoxSize/2, BoxSize/2))
            x = pos[:, 0] % boxsize
            y = pos[:, 1] % boxsize
            z = pos[:, 2] % boxsize
            f = np.full(n_particles, particle_mass)

            # FIESTA part2grid3D: Numba-JIT compiled, handles periodic BC
            # All files use the same origin=0 since coordinates are global box coords
            rho_file = fiesta.p2g.part2grid3D(
                x, y, z, f,
                boxsize=boxsize, ngrid=ngrid,
                method=method, periodic=True, origin=0.
            )

            density += rho_file

            del pos, data, x, y, z, f, rho_file

            if verbose:
                print(f"  [{i+1}/{len(files)}] {filepath.split('/')[-1]}: "
                      f"{n_particles:,} particles")

    if verbose:
        print(f"\n{'='*60}")
        print(f"DENSITY FIELD COMPLETE (FIESTA {method})")
        print(f"  Total particles processed: {total_particles:,}")
        print(f"  Grid: {ngrid}^3 = {ngrid**3:,} cells")
        print(f"  Particles per cell: {total_particles / ngrid**3:.1f}")
        print(f"  Density range: [{density.min():.4e}, {density.max():.4e}]")
        print(f"{'='*60}")

    return density.astype(np.float32)


def plot_density_field_field_stream(density, slice_indices=None):
    """
    Plot 2x2 panel of slices from the 3D density field.

    Parameters
    ----------
    density : np.ndarray
        3D array of shape (ngrid, ngrid, ngrid).
    slice_indices : list or None
        List of four integers indicating the z-slices to display in the 2x2 grid.
        If None, chooses [0, n//3, 2n//3, n-1].
    """
    import matplotlib.pyplot as plt
    ngrid = density.shape[2]
    if slice_indices is None:
        slice_indices = [0, ngrid // 3, 2 * ngrid // 3, ngrid - 1]
    assert len(slice_indices) == 4, "Must provide four slice indices."

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for idx, ax in enumerate(axes.flat):
        iz = slice_indices[idx]
        im = ax.imshow(np.log10(density[:, :, iz]).T, origin='lower', cmap='viridis')
        ax.set_title(f'z = {iz}')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)

    plt.tight_layout()
    plt.show()

def build_density_field_mpi_slabs(snapshot_path=SNAPSHOT_Z0200, ngrid=512,
                                  boxsize=BOXSIZE, method='NGP',
                                  save_dir=None, MPI=None, verbose=True):
    """
    Build density field with MPI by splitting the grid into x-slabs per rank.
    Uses FIESTA's Numba-JIT compiled grid assignment for fast, accurate deposition.
    Supports NGP, CIC, TSC, and PCS methods.

    For CIC/TSC/PCS, uses a padded x-slab approach: each rank deposits its
    center particles onto a slab that extends by `pad` cells on each side in x,
    then communicates the padding overlap with neighboring ranks via MPI.

    The x-axis decomposition matches shift's MPI.split(ngrid), so the output
    slabs are directly compatible with shift's MPI FFT functions and
    run_tweb_memory_optimized.

    Args:
        snapshot_path: Path to snapshot, e.g., '.../halos/z0.200'
        ngrid: Full grid resolution along each axis
        boxsize: Box size in Mpc/h
        method: 'NGP', 'CIC', 'TSC', or 'PCS'
        save_dir: If not None, save slab to this directory as .npz
        MPI: MPIutils MPI object (required)
        verbose: Print progress

    Returns:
        density_local: (nx_local, ngrid, ngrid) density slab (float32)
    """
    if MPI is None:
        raise ValueError("MPI object is required for MPI slab build.")

    rank, size = MPI.rank, MPI.size
    cell_size = boxsize / ngrid

    # Compute slab bounds (same decomposition as shift's MPI.split)
    nx_base = ngrid // size
    remainder = ngrid % size
    if rank < remainder:
        x_start = rank * (nx_base + 1)
        x_end = x_start + (nx_base + 1)
    else:
        x_start = rank * nx_base + remainder
        x_end = x_start + nx_base
    nx_local = x_end - x_start

    # Padding width for higher-order methods (CIC/TSC spread ±1, PCS ±2)
    pad_widths = {'NGP': 0, 'CIC': 1, 'TSC': 1, 'PCS': 2}
    if method not in pad_widths:
        raise ValueError(f"Unknown method '{method}'. Use 'NGP', 'CIC', 'TSC', or 'PCS'.")
    pad = pad_widths[method]
    nx_padded = nx_local + 2 * pad

    # NGP can safely accumulate in float32 (no fractional weights).
    # CIC/TSC/PCS use float64 for precision with trilinear weights.
    accum_dtype = np.float32 if pad == 0 else np.float64
    density_padded = np.zeros((nx_padded, ngrid, ngrid), dtype=accum_dtype)
    total_particles = 0
    particle_mass = None

    # Read particle mass from the header of the first available file
    for particle_type in ['field', 'halo']:
        dirpath = f'{snapshot_path}/{particle_type}_rv_A'
        files = sorted(glob.glob(f'{dirpath}/*.asdf'))
        if files and particle_mass is None:
            with asdf.open(files[0]) as af:
                header = dict(af.tree['header'])
                particle_mass = float(header['ParticleMassHMsun'])
                if verbose and rank == 0:
                    print(f"Particle mass from header: {particle_mass:.4e} Msun/h")
            break

    if particle_mass is None:
        raise FileNotFoundError(f"No ASDF files found in {snapshot_path}")

    if verbose:
        print(f"[rank {rank}] Slab x=[{x_start}, {x_end}), "
              f"nx_local={nx_local}, pad={pad}, method={method}")

    for particle_type in ['field', 'halo']:
        dirpath = f'{snapshot_path}/{particle_type}_rv_A'
        files = sorted(glob.glob(f'{dirpath}/*.asdf'))

        if verbose:
            print(f"[rank {rank}] Processing {len(files)} {particle_type} files...")

        for i, filepath in enumerate(files):
            data = read_asdf(filepath, verbose=False)
            pos = np.array(data['pos'])
            total_particles += len(pos)
            del data  # free ASDF container immediately

            if pad == 0:
                # ----------------------------------------------------------
                # NGP: direct deposition via np.add.at (no temporary grid).
                # This is far more memory-efficient than FIESTA for NGP
                # because it avoids allocating a full (nx_local, ngrid, ngrid)
                # temporary array for every file.
                # ----------------------------------------------------------
                ix = ((pos[:, 0] % boxsize) / cell_size).astype(np.int64) % ngrid
                iy = ((pos[:, 1] % boxsize) / cell_size).astype(np.int64) % ngrid
                iz = ((pos[:, 2] % boxsize) / cell_size).astype(np.int64) % ngrid
                del pos

                mask = (ix >= x_start) & (ix < x_end)
                if np.any(mask):
                    lx = ix[mask] - x_start
                    ly = iy[mask]
                    lz = iz[mask]
                    np.add.at(density_padded, (lx, ly, lz), particle_mass)
                    del lx, ly, lz
                del ix, iy, iz, mask
            else:
                # ----------------------------------------------------------
                # CIC/TSC/PCS: use FIESTA on padded slab, with aggressive
                # memory cleanup of intermediates before the FIESTA call.
                # ----------------------------------------------------------
                x = pos[:, 0] % boxsize
                y = pos[:, 1] % boxsize
                z = pos[:, 2] % boxsize
                del pos

                ix_global = (x / cell_size).astype(np.int64) % ngrid
                mask = (ix_global >= x_start) & (ix_global < x_end)
                del ix_global

                if np.any(mask):
                    x_filt = x[mask]
                    y_filt = y[mask]
                    z_filt = z[mask]
                    del x, y, z, mask  # free full-size arrays before FIESTA

                    n_filt = len(x_filt)
                    f = np.full(n_filt, particle_mass)

                    # Shift x so center particles land at local indices
                    # [pad, pad + nx_local), leaving room for spread into pads.
                    x_local = (x_filt - x_start * cell_size
                               + pad * cell_size) % boxsize
                    del x_filt
                    rho = fiesta.p2g.part2grid3D(
                        x_local, y_filt, z_filt, f,
                        boxsize=[nx_padded * cell_size, boxsize, boxsize],
                        ngrid=[nx_padded, ngrid, ngrid],
                        method=method,
                        periodic=[False, True, True],
                        origin=[0., 0., 0.],
                    )
                    del x_local, y_filt, z_filt, f
                    density_padded += rho
                    del rho
                else:
                    del x, y, z, mask

            if verbose and (i + 1) % 10 == 0:
                print(f"[rank {rank}]  {i+1}/{len(files)} files done")

    # Communicate boundary overlaps for padded methods (CIC/TSC/PCS).
    # Each rank's pad zones contain the spread from ITS center particles into
    # neighboring slabs. Send these to the appropriate neighbor and add to
    # their edge cells.
    if pad > 0:
        for p in range(pad):
            # Right pad → send to rank+1, receive rank-1's right pad
            from_below = MPI.send_up(density_padded[pad + nx_local + p])
            density_padded[pad + p] += from_below

            # Left pad → send to rank-1, receive rank+1's left pad
            from_above = MPI.send_down(density_padded[pad - 1 - p])
            density_padded[pad + nx_local - 1 - p] += from_above

    # Extract center (non-padded) region
    density_local = density_padded[pad:pad + nx_local].astype(np.float32)

    if save_dir is not None:
        np.savez(
            f"{save_dir}/abacus_z0200_density_slab_rank{rank:04d}.npz",
            dens=density_local,
            x_start=x_start,
            x_end=x_end,
            ngrid=ngrid,
            boxsize=boxsize,
        )

    if verbose:
        print(f"[rank {rank}] Done. Particles processed: {total_particles:,}")

    return density_local


def stitch_density_slabs(slab_dir, output_path, expected_slabs=None, verbose=True):
    """
    Stitch MPI density slabs into a single full grid.
    WARNING: For large ngrid this can require >160 GB RAM (float32).
    """
    slab_files = sorted(glob.glob(f"{slab_dir}/abacus_z0200_density_slab_rank*.npz"))
    if not slab_files:
        raise FileNotFoundError(f"No slab files found in {slab_dir}")
    if expected_slabs is not None and len(slab_files) != expected_slabs:
        raise RuntimeError(
            f"Expected {expected_slabs} slab files, found {len(slab_files)}. "
            "A rank may still be writing or a prior run left stale files."
        )

    sample = np.load(slab_files[0])
    ngrid = int(sample["ngrid"])
    boxsize = float(sample["boxsize"])
    sample.close()

    full = np.zeros((ngrid, ngrid, ngrid), dtype=np.float32)

    for i, slab_path in enumerate(slab_files):
        data = np.load(slab_path)
        dens = data["dens"]
        x_start = int(data["x_start"])
        x_end = int(data["x_end"])
        full[x_start:x_end, :, :] = dens
        data.close()
        if verbose and (i + 1) % 10 == 0:
            print(f"Stitched {i+1}/{len(slab_files)} slabs...")

    np.savez(output_path, dens=full, ngrid=ngrid, boxsize=boxsize)
    if verbose:
        print(f"Saved full density grid to {output_path}")


def particles_to_density_grid(pos, ngrid=256, boxsize=BOXSIZE, method='CIC', verbose=True):
    """
    Deposit particles onto a 3D grid using mass assignment.

    Args:
        pos: (N, 3) particle positions in Mpc/h
        ngrid: Grid resolution (e.g., 256, 512)
        boxsize: Box size in Mpc/h
        method: 'NGP' (nearest grid point) or 'CIC' (cloud-in-cell)

    Returns:
        density: (ngrid, ngrid, ngrid) overdensity field (delta = rho/rho_mean - 1)
    """
    if verbose:
        print(f"Building density field: {ngrid}^3 grid, {method} assignment...")

    density = np.zeros((ngrid, ngrid, ngrid), dtype=np.float64)
    cell_size = boxsize / ngrid

    # Normalize positions to grid units [0, ngrid)
    pos_grid = (pos % boxsize) / cell_size

    if method == 'NGP':
        ix = pos_grid[:, 0].astype(int) % ngrid
        iy = pos_grid[:, 1].astype(int) % ngrid
        iz = pos_grid[:, 2].astype(int) % ngrid
        np.add.at(density, (ix, iy, iz), 1.0)

    elif method == 'CIC':
        ix = pos_grid[:, 0].astype(int)
        iy = pos_grid[:, 1].astype(int)
        iz = pos_grid[:, 2].astype(int)

        dx = pos_grid[:, 0] - ix
        dy = pos_grid[:, 1] - iy
        dz = pos_grid[:, 2] - iz

        # 8 neighboring cells with trilinear weights
        for ox in [0, 1]:
            for oy in [0, 1]:
                for oz in [0, 1]:
                    wx = (1 - dx) if ox == 0 else dx
                    wy = (1 - dy) if oy == 0 else dy
                    wz = (1 - dz) if oz == 0 else dz

                    jx = (ix + ox) % ngrid
                    jy = (iy + oy) % ngrid
                    jz = (iz + oz) % ngrid

                    np.add.at(density, (jx, jy, jz), wx * wy * wz)

    # Convert to overdensity: delta = rho / rho_mean - 1
    rho_mean = density.mean()
    if rho_mean > 0:
        density = density / rho_mean - 1

    if verbose:
        print(f"  Density field: min={density.min():.2f}, max={density.max():.2f}, mean={density.mean():.6f}")

    return density.astype(np.float32)


# =============================================================================
# MEMORY-OPTIMIZED MPI T-WEB
# =============================================================================

def run_tweb_memory_optimized(dens, boxsize, ngrid, threshold, MPI,
                               Rsmooth=None, verbose=True):
    """
    Memory-optimized MPI T-Web classification for large grids (e.g., ngrid=3414).

    Replicates the same algorithm as cactus.src.tweb.mpi_run_tweb but
    aggressively frees intermediate arrays at each stage, reducing peak
    per-rank memory from ~200+ GB to ~90 GB (for 32 ranks, ngrid=3414).

    Key optimizations vs mpi_run_tweb:
    - Skips unused x3d, y3d, z3d 3D grid arrays (saves ~30 GB/rank)
    - Frees kx3d, ky3d, kz3d immediately after computing kmag (saves ~30 GB)
    - Frees delta after FFT, deltak after potential, kmag after potential
    - Uses usereal=True (real-space derivatives) so k-grids aren't needed
      during the derivative stage
    - Calls gc.collect() after each major deallocation

    Uses the same shift/fiesta/cactus functions as mpi_run_tweb:
    - shift.cart.mpi_fft3D / mpi_ifft3D for distributed FFTs
    - shift.cart.mpi_kgrid3D for k-space grids
    - fiesta.maths.mpi_dfdx/dfdy/dfdz for real-space derivatives
    - cactus.src.maths.get_eig_3by3 for eigenvalue computation

    Assumes periodic boundary conditions.

    Parameters
    ----------
    dens : 3darray
        Local density slab, shape (nx_local, ngrid, ngrid).
    boxsize : float
        Box size in Mpc/h.
    ngrid : int
        Full grid size along each axis.
    threshold : float
        Threshold for T-Web eigenvalue classification.
    MPI : object
        MPIutils MPI object (from shift.mpiutils).
    Rsmooth : float, optional
        Gaussian smoothing radius in Mpc/h.
    verbose : bool, optional
        Print progress from rank 0.

    Returns
    -------
    cweb : 3darray
        Cosmic web classification (0=void, 1=wall, 2=filament, 3=cluster),
        shape (nx_local, ngrid, ngrid).
    eig_vals : 4darray
        Eigenvalues array of shape (3, nx_local, ngrid, ngrid).
    """
    import gc
    import shift
    from cactus.src import density as cactus_density, maths as cactus_maths

    dshape = np.shape(dens)
    rank = MPI.rank

    # -------------------------------------------------------------------------
    # Stage 1: Density contrast  (~10 GB/rank)
    # -------------------------------------------------------------------------
    if verbose:
        MPI.mpi_print_zero('  [1/8] Convert density to density contrast.')
    delta = cactus_density.mpi_norm_dens(dens, MPI) - 1.
    del dens
    gc.collect()

    # -------------------------------------------------------------------------
    # Stage 2: Build k-grids and kmag, then free 3D k-grids  (~20 GB/rank)
    # Skip x3d, y3d, z3d entirely -- they are never used in the algorithm.
    # -------------------------------------------------------------------------
    if verbose:
        MPI.mpi_print_zero('  [2/8] Constructing k-space grids.')
    kx3d, ky3d, kz3d = shift.cart.mpi_kgrid3D(boxsize, ngrid, MPI)
    kmag = np.sqrt(kx3d**2. + ky3d**2. + kz3d**2.)
    del kx3d, ky3d, kz3d  # saves ~30 GB/rank (not needed for usereal=True)
    gc.collect()

    # -------------------------------------------------------------------------
    # Stage 3: Forward FFT  (~30 GB/rank: kmag + deltak, delta freed)
    # -------------------------------------------------------------------------
    if verbose:
        MPI.mpi_print_zero('  [3/8] Forward FFT of density contrast.')
    deltak = shift.cart.mpi_fft3D(delta, boxsize, ngrid, MPI)
    del delta
    gc.collect()

    # -------------------------------------------------------------------------
    # Stage 4: Smooth + compute gravitational potential in Fourier space
    #          (~20 GB/rank: just phik, after freeing deltak and kmag)
    # -------------------------------------------------------------------------
    if Rsmooth is not None:
        if verbose:
            MPI.mpi_print_zero('  [4/8] Smoothing + computing potential field.')
        deltak *= shift.cart.convolve_gaussian(kmag, Rsmooth)
    else:
        if verbose:
            MPI.mpi_print_zero('  [4/8] Computing potential field in Fourier space.')

    cond = np.where(kmag != 0)
    phik = np.zeros(np.shape(deltak)) + 1j * np.zeros(np.shape(deltak))
    phik[cond] = -deltak[cond] / kmag[cond]**2.
    del deltak, kmag, cond
    gc.collect()

    # -------------------------------------------------------------------------
    # Stage 5: Inverse FFT to get real-space potential  (~10 GB/rank)
    # -------------------------------------------------------------------------
    if verbose:
        MPI.mpi_print_zero('  [5/8] Inverse FFT to get gravitational potential.')
    phi = shift.cart.mpi_ifft3D(phik, boxsize, ngrid, MPI)
    del phik
    gc.collect()

    # -------------------------------------------------------------------------
    # Stage 6-7: Real-space derivatives of the potential  (peak ~70 GB/rank)
    # Uses fiesta finite differences with MPI boundary communication.
    # -------------------------------------------------------------------------
    if verbose:
        MPI.mpi_print_zero('  [6/8] Differentiating gravitational potential (real-space).')

    _, xgrid = shift.cart.mpi_grid1D(boxsize, ngrid, MPI)
    _, ygrid = shift.cart.grid1D(boxsize, ngrid)
    _, zgrid = shift.cart.grid1D(boxsize, ngrid)

    # First derivatives
    phi_x = fiesta.maths.mpi_dfdx(xgrid, phi, MPI, periodic=True)
    phi_y = fiesta.maths.mpi_dfdy(ygrid, phi, MPI, periodic=True)
    phi_z = fiesta.maths.mpi_dfdz(zgrid, phi, MPI, periodic=True)

    # Second derivatives from phi_x
    if verbose:
        MPI.mpi_print_zero('  [7/8] Computing second derivatives (Hessian).')
    phi_xx = fiesta.maths.mpi_dfdx(xgrid, phi_x, MPI, periodic=True)
    phi_xy = fiesta.maths.mpi_dfdy(ygrid, phi_x, MPI, periodic=True)
    phi_xz = fiesta.maths.mpi_dfdz(zgrid, phi_x, MPI, periodic=True)
    del phi_x
    gc.collect()

    # Second derivatives from phi_y
    phi_yy = fiesta.maths.mpi_dfdy(ygrid, phi_y, MPI, periodic=True)
    phi_yz = fiesta.maths.mpi_dfdz(zgrid, phi_y, MPI, periodic=True)
    del phi_y
    gc.collect()

    # Second derivative from phi_z
    phi_zz = fiesta.maths.mpi_dfdz(zgrid, phi_z, MPI, periodic=True)
    del phi_z, phi
    gc.collect()

    # -------------------------------------------------------------------------
    # Stage 8: Eigenvalues and classification  (peak ~90 GB/rank)
    # -------------------------------------------------------------------------
    if verbose:
        MPI.mpi_print_zero('  [8/8] Computing eigenvalues and classifying cosmic web.')

    phi_xx = phi_xx.flatten()
    phi_xy = phi_xy.flatten()
    phi_xz = phi_xz.flatten()
    phi_yy = phi_yy.flatten()
    phi_yz = phi_yz.flatten()
    phi_zz = phi_zz.flatten()

    eig1, eig2, eig3 = cactus_maths.get_eig_3by3(
        phi_xx, phi_xy, phi_xz, phi_yy, phi_yz, phi_zz
    )
    del phi_xx, phi_xy, phi_xz, phi_yy, phi_yz, phi_zz
    gc.collect()

    # Classification: 0=void, 1=wall, 2=filament, 3=cluster
    cweb = np.zeros(len(eig1))
    cond = np.where(
        (eig3 >= threshold) & (eig2 < threshold) & (eig1 < threshold)
    )[0]
    cweb[cond] = 1.
    cond = np.where(
        (eig3 >= threshold) & (eig2 >= threshold) & (eig1 < threshold)
    )[0]
    cweb[cond] = 2.
    cond = np.where(
        (eig3 >= threshold) & (eig2 >= threshold) & (eig1 >= threshold)
    )[0]
    cweb[cond] = 3.

    cweb = cweb.reshape(dshape)
    eig_vals = np.array([
        eig1.reshape(dshape),
        eig2.reshape(dshape),
        eig3.reshape(dshape),
    ])

    if verbose:
        MPI.mpi_print_zero('  T-Web classification complete.')

    return cweb, eig_vals


# =============================================================================
# T-WEB LOOKUP
# =============================================================================

def assign_tweb_to_galaxies(gal_x, gal_y, gal_z, tweb_grid, boxsize=BOXSIZE):
    """
    Look up T-Web classification/eigenvalues for each galaxy based on position.

    Args:
        gal_x, gal_y, gal_z: Galaxy positions in Mpc/h (from CubicBox mock)
        tweb_grid: 3D array from T-Web calculation
                   Shape: (ngrid, ngrid, ngrid) for classification
                   Shape: (ngrid, ngrid, ngrid, 3) for eigenvalues
        boxsize: Box size in Mpc/h

    Returns:
        tweb_values: T-Web values for each galaxy
                     Shape: (N_gal,) for classification
                     Shape: (N_gal, 3) for eigenvalues
    """
    ngrid = tweb_grid.shape[0]
    cell_size = boxsize / ngrid

    # Convert positions to grid indices (with periodic wrapping)
    ix = ((np.asarray(gal_x) % boxsize) / cell_size).astype(int)
    iy = ((np.asarray(gal_y) % boxsize) / cell_size).astype(int)
    iz = ((np.asarray(gal_z) % boxsize) / cell_size).astype(int)

    # Ensure indices are within bounds
    ix = np.clip(ix, 0, ngrid - 1)
    iy = np.clip(iy, 0, ngrid - 1)
    iz = np.clip(iz, 0, ngrid - 1)

    # Look up values
    if tweb_grid.ndim == 3:
        return tweb_grid[ix, iy, iz]
    else:
        return tweb_grid[ix, iy, iz, :]


# =============================================================================
# COORDINATE TRANSFORMATIONS (for CutSky mock)
# =============================================================================

def cutsky_to_box_coords(ra, dec, z_cosmo):
    """
    Convert CutSky (RA, DEC, Z_COSMO) to approximate box (x, y, z) coordinates.

    WARNING: The CutSky mock is from a light cone spanning z=0-0.8, constructed
    from multiple box replicas. This conversion gives approximate box coords
    but may not perfectly match the z=0.2 snapshot used for T-Web.

    For accurate T-Web matching, USE THE CUBICBOX MOCK instead.

    Args:
        ra, dec: Sky coordinates in degrees
        z_cosmo: Cosmological redshift (no RSD)

    Returns:
        x, y, z: Approximate box coordinates in Mpc/h, range [0, BOXSIZE)
    """
    # Convert redshift to comoving distance
    dist = cosmo.comoving_distance(z_cosmo).to(u.Mpc).value * cosmo.h

    # Convert to Cartesian (observer at origin)
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)

    x_obs = dist * np.cos(dec_rad) * np.cos(ra_rad)
    y_obs = dist * np.cos(dec_rad) * np.sin(ra_rad)
    z_obs = dist * np.sin(dec_rad)

    # Shift by observer origin and apply periodic BC
    x = (x_obs + OBSERVER_ORIGIN[0]) % BOXSIZE
    y = (y_obs + OBSERVER_ORIGIN[1]) % BOXSIZE
    z = (z_obs + OBSERVER_ORIGIN[2]) % BOXSIZE

    return x, y, z


def print_recommended_workflow() -> None:
    """Print the recommended non-MPI workflow reference for operators."""
    print("""
PARTICLE DATA SUMMARY:
======================
- Field files: 34 × ~170M particles = ~5.8 billion
- Halo files:  34 × ~122M particles = ~4.2 billion
- TOTAL: ~10 billion particles (3% subsample)
- Memory if loaded: ~240 GB (too large!)

RECOMMENDED WORKFLOW (like Illustris_cactus.py):
================================================

# STEP 1: Build density field by streaming (only ~1 GB memory)
density = build_density_field_streaming(SNAPSHOT_Z0200, ngrid=512)

# STEP 2: Save density field for reuse
np.savez(f'{ABACUS_SLAB_DIR}/abacus_z0200_density_512.npz', dens=density)

# STEP 3: Run T-Web (like Illustris_cactus.py)
import cactus
cweb, eig_vals = cactus.src.tweb.run_tweb(
    density, boxsize=2000., ngrid=512,
    threshold=0.2, Rsmooth=2., boundary='periodic', verbose=True
)

# STEP 4: Save eigenvalues
np.savez(f'{ABACUS_SLAB_DIR}/abacus_z0200_eigenvalues_512.npz', eig_vals=eig_vals)

# STEP 5: Assign to CutSky mock galaxies
# (Use your catalog loader, then map RA/DEC/Z_COSMO with cutsky_to_box_coords)
x, y, z = cutsky_to_box_coords(cutsky['RA'], cutsky['DEC'], cutsky['Z_COSMO'])
cutsky['eig_vals'] = assign_tweb_to_galaxies(x, y, z, eig_vals)
""")


def run_mpi_slab_workflow(ngrid: int = 3414, stitch_after_mpi: bool = True) -> None:
    """Run MPI slab build and optional slab stitching."""
    from shift import mpiutils

    MPI = mpiutils.MPI()
    print("MPI size:", MPI.size)

    save_dir = ABACUS_SLAB_DIR
    build_density_field_mpi_slabs(
        snapshot_path=SNAPSHOT_Z0200,
        ngrid=ngrid,
        boxsize=BOXSIZE,
        method='NGP',
        save_dir=save_dir,
        MPI=MPI,
        verbose=True,
    )

    # Synchronize all ranks before rank 0 starts reading slab files.
    MPI.wait()

    if stitch_after_mpi and MPI.rank == 0:
        output_path = f"{save_dir}/AbacusSummit_base_c000_ph000_z0200_ngrid{ngrid}_NGP_full.npz"
        stitch_density_slabs(
            save_dir,
            output_path,
            expected_slabs=MPI.size,
            verbose=True,
        )
    MPI.end()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for safe entrypoint behavior."""
    parser = argparse.ArgumentParser(
        description="AbacusSummit particle processing and MPI slab workflow for T-Web."
    )
    parser.add_argument(
        "--show-workflow",
        action="store_true",
        help="Print the recommended workflow summary and exit (unless --run-mpi is also set).",
    )
    parser.add_argument(
        "--run-mpi",
        action="store_true",
        help="Run the MPI slab density build + optional stitching.",
    )
    parser.add_argument(
        "--ngrid",
        type=int,
        default=3414,
        help="Grid size for MPI slab workflow (default: 3414).",
    )
    parser.add_argument(
        "--no-stitch",
        action="store_true",
        help="Disable post-MPI slab stitching on rank 0.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print("=" * 70)
    print("ABACUSSUMMIT T-WEB PROCESSING")
    print("=" * 70)
    # Preserve legacy default behavior when no explicit mode is requested.
    should_show = args.show_workflow or not args.run_mpi
    should_run_mpi = args.run_mpi or (not args.show_workflow and not args.run_mpi)

    if should_show:
        print_recommended_workflow()
    if should_run_mpi:
        run_mpi_slab_workflow(ngrid=args.ngrid, stitch_after_mpi=not args.no_stitch)





# =============================================================================
# MAIN - DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    main()