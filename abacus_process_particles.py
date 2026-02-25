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

# Workflow status: LEGACY
# NOTE: This script is retained for historical reference.
# The canonical active implementation is `abacus_process_particles2.py`.

import numpy as np
import glob
import asdf
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
import fitsio
from astropy.table import Table

import cactus
from cactus.ext import fiesta

# abacusutils read_asdf automatically decompresses rvint data
from abacusnbody.data.read_abacus import read_asdf

# =============================================================================
# PATHS
# =============================================================================
ABACUS_BASE = '/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit/AbacusSummit_base_c000_ph000'
MOCKS_BASE = '/global/cfs/cdirs/desi/cosmosim/SecondGenMocks/AbacusSummit'

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


def load_cutsky_mock(filepath=CUTSKY_BGS, columns=None, in_y5_only=True, verbose=True):
    """
    Load CutSky BGS mock with (RA, DEC, Z) sky coordinates.
    Note: This mock spans z=0-0.8 and requires coordinate conversion for T-Web.

    Args:
        filepath: Path to FITS file
        columns: List of columns to load
        in_y5_only: If True, filter to IN_Y5==1 galaxies (DESI Y5 footprint)

    Returns:
        Table with galaxy properties
    """
    if columns is None:
        columns = ['RA', 'DEC', 'Z', 'Z_COSMO', 'HALO_MASS', 'CEN', 'R_MAG_ABS', 'IN_Y5']

    if verbose:
        print(f"Loading CutSky mock: {filepath.split('/')[-1]}")

    data = Table(fitsio.read(filepath, columns=columns))

    if in_y5_only:
        mask = data['IN_Y5'] == 1
        data = data[mask]

    if verbose:
        print(f"  Loaded {len(data):,} galaxies (IN_Y5={in_y5_only})")
        print(f"  Redshift range: z=[{data['Z'].min():.4f}, {data['Z'].max():.4f}]")

    return data


# =============================================================================
# DENSITY FIELD CONSTRUCTION
# =============================================================================

def cactus_density_field():
    dens = cactus.ext.fiesta.p2g.part2grid3D(pos[0], pos[1], pos[2], f = np.ones(len(pos[0])), boxsize=BOXSIZE, ngrid=512, method='NGP', periodic=True, origin=OBSERVER_ORIGIN)
    '''
    Code above works but does not work for ngrid=4096...interesting...
    '''
    return 'code not implemented'

def build_density_field_streaming(snapshot_path=SNAPSHOT_Z0200, ngrid=512,
                                   boxsize=BOXSIZE, method='NGP', verbose=True):
    """
    Build density field by streaming through particle files one at a time.
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
        density: (ngrid, ngrid, ngrid) overdensity field

    Example:
        # Build 512^3 density field from all particles
        density = build_density_field_streaming(SNAPSHOT_Z0200, ngrid=512)

        # Save for later use (like Illustris_cactus.py)
        np.savez('abacus_z0200_density_512.npz', dens=density)
    """
    density = np.zeros((ngrid, ngrid, ngrid), dtype=np.float64)
    cell_size = boxsize / ngrid
    total_particles = 0

    # Process both field and halo particles
    for particle_type in ['field', 'halo']:
        dirpath = f'{snapshot_path}/{particle_type}_rv_A'
        files = sorted(glob.glob(f'{dirpath}/*.asdf'))

        if verbose:
            print(f"\nProcessing {len(files)} {particle_type} files...")

        for i, filepath in enumerate(files):
            # Load particles from this file
            data = read_asdf(filepath, verbose=False)
            pos = np.array(data['pos'])
            n_particles = len(pos)
            total_particles += n_particles

            # Deposit onto grid
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

            # Free memory
            del pos, data

            if verbose:
                print(f"  [{i+1}/{len(files)}] {filepath.split('/')[-1]}: {n_particles:,} particles")

    # Convert to overdensity
    rho_mean = density.mean()
    density = density / rho_mean - 1

    if verbose:
        print(f"\n{'='*60}")
        print(f"DENSITY FIELD COMPLETE")
        print(f"  Total particles processed: {total_particles:,}")
        print(f"  Grid: {ngrid}^3 = {ngrid**3:,} cells")
        print(f"  Particles per cell: {total_particles / ngrid**3:.1f}")
        print(f"  Overdensity range: [{density.min():.2f}, {density.max():.2f}]")
        print(f"{'='*60}")

    return density.astype(np.float32)


def build_density_field_mpi_slabs(snapshot_path=SNAPSHOT_Z0200, ngrid=512,
                                  boxsize=BOXSIZE, method='NGP',
                                  save_dir=None, MPI=None, verbose=True):
    """
    Build density field with MPI by splitting the grid into x-slabs per rank.
    Each rank loads only particles in its slab and saves its local grid.
    """
    if MPI is None:
        raise ValueError("MPI object is required for MPI slab build.")

    rank, size = MPI.rank, MPI.size
    cell_size = boxsize / ngrid

    # Compute slab bounds for this rank (load-balanced by remainder)
    nx_base = ngrid // size
    remainder = ngrid % size
    if rank < remainder:
        x_start = rank * (nx_base + 1)
        x_end = x_start + (nx_base + 1)
    else:
        x_start = rank * nx_base + remainder
        x_end = x_start + nx_base
    nx_local = x_end - x_start

    density_local = np.zeros((nx_local, ngrid, ngrid), dtype=np.float32)
    total_particles = 0

    for particle_type in ['field', 'halo']:
        dirpath = f'{snapshot_path}/{particle_type}_rv_A'
        files = sorted(glob.glob(f'{dirpath}/*.asdf'))

        if verbose:
            print(f"[rank {rank}] Processing {len(files)} {particle_type} files...")

        for i, filepath in enumerate(files):
            data = read_asdf(filepath, verbose=False)
            pos = np.array(data['pos'])
            total_particles += len(pos)

            # Grid indices (periodic)
            ix = ((pos[:, 0] % boxsize) / cell_size).astype(np.int64)
            iy = ((pos[:, 1] % boxsize) / cell_size).astype(np.int64)
            iz = ((pos[:, 2] % boxsize) / cell_size).astype(np.int64)

            # Keep only particles in this rank's x-slab
            mask = (ix >= x_start) & (ix < x_end)
            if np.any(mask):
                lx = ix[mask] - x_start
                ly = iy[mask] % ngrid
                lz = iz[mask] % ngrid
                np.add.at(density_local, (lx, ly, lz), 1.0)

            del pos, data

            if verbose and (i + 1) % 10 == 0:
                print(f"[rank {rank}]  {i+1}/{len(files)} files done")

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





# =============================================================================
# MAIN - DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ABACUSSUMMIT T-WEB PROCESSING")
    print("=" * 70)

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
np.savez('/pscratch/sd/d/dkololgi/abacus_z0200_density_512.npz', dens=density)

# STEP 3: Run T-Web (like Illustris_cactus.py)
import cactus
cweb, eig_vals = cactus.src.tweb.run_tweb(
    density, boxsize=2000., ngrid=512,
    threshold=0.2, Rsmooth=2., boundary='periodic', verbose=True
)

# STEP 4: Save eigenvalues
np.savez('/pscratch/sd/d/dkololgi/abacus_z0200_eigenvalues_512.npz', eig_vals=eig_vals)

# STEP 5: Assign to CutSky mock galaxies
cutsky = load_cutsky_mock(in_y5_only=True)
x, y, z = cutsky_to_box_coords(cutsky['RA'], cutsky['DEC'], cutsky['Z_COSMO'])
cutsky['eig_vals'] = assign_tweb_to_galaxies(x, y, z, eig_vals)
""")

    # -------------------------------------------------------------------------
    # MPI slab density build (requires srun/mpirun)
    # -------------------------------------------------------------------------
    from shift import mpiutils

    MPI = mpiutils.MPI()
    print("MPI size:", MPI.size)

    ngrid = 3414
    save_dir = '/pscratch/sd/d/dkololgi/AbscusSummit_densities'
    stitch_after_mpi = True  # Set True on a high-memory node

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
        output_path = (
            f"{save_dir}/AbacusSummit_base_c000_ph000_z0200_ngrid{ngrid}_NGP_full.npz"
        )
        stitch_density_slabs(
            save_dir,
            output_path,
            expected_slabs=MPI.size,
            verbose=True,
        )
    MPI.end()