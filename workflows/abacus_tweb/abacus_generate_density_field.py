'''
Generate the density field for the Abacus T-Web mock.

Relevant Paths:
Abacus Base Simulation:
(A type or 3% only): /global/cfs/cdirs/desi/public/cosmosim/AbacusSummit/AbacusSummit_base_c000_ph000/halos/z0.200/

A and B type for 10% (3+7) particle subsample - transferred from nersc hpss):
/pscratch/sd/d/dkololgi/AbscusSummit_densities/AbacusSummit_base_c000_ph000/halos/z0.200/halos/z0.200/

Alex Smith's Mock:
/global/cfs/cdirs/desi/cosmosim/SecondGenMocks/AbacusSummit/CutSky/BGS/v0.1/z0.200/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000.fits
'''
import os
import sys
from pathlib import Path
import glob
import numpy as np

from abacusnbody.data.read_abacus import read_asdf
from abacusnbody.analysis.tsc import tsc_parallel

# Allow imports from the repo root (so `shared` resolves when run directly).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.config_paths import ABACUS_BASE_LOCAL, ABACUS_SLAB_DIR

OUTPUT_PATH = (
    f"{ABACUS_SLAB_DIR}/density_fields/"
    "AbacusSummit_base_c000_ph000_z0.200_ngrid_{NGRID}_10pc_density_field.npy"
)


NGRID = 1024

def _detect_snapshot_dir():
    """
    Resolve z=0.200 snapshot directory across common layouts:
      1) <base>/halos/z0.200
      2) <base>/halos/z0.200/halos/z0.200   (from htar extract in z0.200 cwd)
    This supports both direct CFS layout and nested HPSS extract layout.
    """
    c1 = Path(ABACUS_BASE_LOCAL) / "halos" / "z0.200"
    c2 = c1 / "halos" / "z0.200"
    for cand in (c1, c2):
        if (cand / "header").exists():
            return cand
    raise FileNotFoundError(
        "Could not find z0.200 snapshot header. Checked "
        f"{c1} and {c2}."
    )


SNAPSHOT_DIR = _detect_snapshot_dir()
HALO_PATH_A = str(SNAPSHOT_DIR / 'halo_rv_A')
HALO_PATH_B = str(SNAPSHOT_DIR / 'halo_rv_B')
FIELD_PATH_A = str(SNAPSHOT_DIR / 'field_rv_A')
FIELD_PATH_B = str(SNAPSHOT_DIR / 'field_rv_B')
HEADER_PATH = str(SNAPSHOT_DIR)

# Detect BOXSIZE from header text file
with open(Path(HEADER_PATH) / 'header', 'r') as f:
    for line in f:
        if 'BoxSize' in line:
            BOXSIZE = float(line.split('=')[1].strip()) # Mpc/h

# Detect number of files in each path
num_files_halo_a = len(glob.glob(f'{HALO_PATH_A}/*.asdf'))
num_files_halo_b = len(glob.glob(f'{HALO_PATH_B}/*.asdf'))
num_files_field_a = len(glob.glob(f'{FIELD_PATH_A}/*.asdf'))
num_files_field_b = len(glob.glob(f'{FIELD_PATH_B}/*.asdf'))

print(f'Number of halo files for A type: {num_files_halo_a}')
print(f'Number of halo files for B type: {num_files_halo_b}')
print(f'Number of field files for A type: {num_files_field_a}')
print(f'Number of field files for B type: {num_files_field_b}')
print(f'Using snapshot dir: {SNAPSHOT_DIR}')
print(f'Box size: {BOXSIZE}')
if min(num_files_halo_a, num_files_halo_b, num_files_field_a, num_files_field_b) == 0:
    print('Warning: one or more expected rv directories is empty.')

# Detect available threads
num_threads = os.cpu_count() or 1
print(f'Available threads: {num_threads}')

def _files_in(folder):
    return sorted(glob.glob(f'{folder}/*.asdf'))


def generate_density_field(
    NGRID,
    BOXSIZE,
    HALO_PATH_A,
    HALO_PATH_B,
    FIELD_PATH_A,
    FIELD_PATH_B,
):
    '''
    Generate the density field for the Abacus T-Web mock.
    Args:
        NGRID: Number of grid points along each axis
        BOXSIZE: Box size in Mpc/h
        HALO_PATH_A: Path to halo catalog for A type
        HALO_PATH_B: Path to halo catalog for B type
        FIELD_PATH_A: Path to field catalog for A type
        FIELD_PATH_B: Path to field catalog for B type
    '''
    # Create grid
    grid = np.zeros((NGRID, NGRID, NGRID), dtype=np.float32)

    # Loop over halo and field files
    folders = [HALO_PATH_A, HALO_PATH_B, FIELD_PATH_A, FIELD_PATH_B]
    for folder in folders:
        files = _files_in(folder)
        if not files:
            print(f'No files found in: {folder}')
            continue

        for i, fname in enumerate(files, start=1):
            data = read_asdf(fname, verbose=False)
            pos = np.asarray(data['pos'], dtype=np.float32)
            # Robust to either [-L/2, L/2) or [0, L) conventions.
            pos = np.mod(pos, BOXSIZE)

            tsc_parallel(pos, grid, BOXSIZE, nthread=num_threads)
            print(f'Processed file {i}/{len(files)}: {Path(fname).name}')
            del data, pos

    out_path = OUTPUT_PATH.format(NGRID=NGRID)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, grid)
    print(f'Saved density field to {out_path}')

def plot_density_field(DENSITY_FIELD_PATH):
    density_field = np.load(DENSITY_FIELD_PATH)
    NGRID = density_field.shape[0]

    
    import matplotlib.pyplot as plt
    # Plot 2D slices of the density field at the midplane along each axis
    mid = NGRID // 2
    thickness = 25  # Half-thickness for a 50-cell thick slab
    slices = {
        'X': density_field[mid-thickness:mid+thickness, :, :],
        'Y': density_field[:, mid-thickness:mid+thickness, :],
        'Z': density_field[:, :, mid-thickness:mid+thickness]
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    vmax = None
    vmin = None
    for ax, (axis, cube) in zip(axes, slices.items()):
        # Collapse 50 thick slab along that axis (mean of log10 to reduce dynamic range spikiness)
        dense_slice = np.mean(np.log10(np.clip(cube, a_min=1e-3, a_max=None)), axis=0)
        im = ax.imshow(
            dense_slice,
            origin='lower',
            aspect='equal',
            cmap='magma'
        )
        ax.set_title(f'Log$_{{10}}$ Density: {axis} slice')
        ax.set_xlabel('Cell Index')
        ax.set_ylabel('Cell Index')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='log$_{10}$ Density')
        # Track global min/max for color scale
        min_val_this = np.nanmin(dense_slice)
        max_val_this = np.nanmax(dense_slice)
        vmin = min_val_this if vmin is None else min(vmin, min_val_this)
        vmax = max_val_this if vmax is None else max(vmax, max_val_this)

    # Set common color limits
    for ax in axes:
        for im in ax.get_images():
            im.set_clim(vmin, vmax)

    fig.suptitle('2D Slices of Log$_{10}$ Density Field (50-cell thick slabs at grid center)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def main(mode: str, DENSITY_FIELD_PATH: str):
    if mode == 'plot':
        plot_density_field(DENSITY_FIELD_PATH)
    elif mode == 'generate':
        generate_density_field(
            NGRID,
            BOXSIZE,
            HALO_PATH_A,
            HALO_PATH_B,
            FIELD_PATH_A,
            FIELD_PATH_B,
        )

if __name__ == '__main__':
    main(mode='plot', DENSITY_FIELD_PATH='/pscratch/sd/d/dkololgi/AbacusSummit_densities/density_fields/AbacusSummit_base_c000_ph000_z0.200_ngrid_2048_10pc_density_field.npy')