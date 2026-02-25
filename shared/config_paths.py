"""Centralized path configuration for TNG/Illustris workflows.

Phase 1.5 goal:
- Keep legacy defaults stable.
- Add a canonical pscratch layout that can be enabled via env vars.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


PROJECT_DIR = _env("TNG_ILLUSTRIS_PROJECT_DIR", str(Path(__file__).resolve().parent))
DK_SCRATCH_ROOT = _env("DK_SCRATCH_ROOT", "/pscratch/sd/d/dkololgi")
TNG_SCRATCH_ROOT = _env("TNG_SCRATCH_ROOT", f"{DK_SCRATCH_ROOT}/tng_illustris")

ABACUS_BASE = _env(
    "TNG_ABACUS_BASE",
    "/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit/AbacusSummit_base_c000_ph000",
)
MOCKS_BASE = _env(
    "TNG_MOCKS_BASE",
    "/global/cfs/cdirs/desi/cosmosim/SecondGenMocks/AbacusSummit",
)

ABACUS_SLAB_DIR = _env("TNG_ABACUS_SLAB_DIR", "/pscratch/sd/d/dkololgi/AbscusSummit_densities")
ABACUS_TWEB_OUTPUT_DIR = _env(
    "TNG_ABACUS_TWEB_OUTPUT_DIR",
    f"{ABACUS_SLAB_DIR}/tweb_rank_outputs",
)
ABACUS_MOCKS_WITH_EIGS_DIR = _env(
    "TNG_ABACUS_MOCKS_WITH_EIGS_DIR",
    "/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs",
)

CUTSKY_Z0200_PATH = _env(
    "TNG_CUTSKY_Z0200_PATH",
    "/global/cfs/cdirs/desi/cosmosim/SecondGenMocks/AbacusSummit/"
    "CutSky/BGS/v0.1/z0.200/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000.fits",
)

ABACUS_CARTESIAN_OUTPUT = _env(
    "TNG_ABACUS_CARTESIAN_OUTPUT",
    f"{PROJECT_DIR}/abacus_cartesian_coords.npy",
)

TNG_LOG_DIR = _env("TNG_LOG_DIR", "/pscratch/sd/d/dkololgi/logs")

# Canonical pscratch layout (opt-in via env vars in current migration stage)
CANONICAL_CACHE_ROOT = _env("TNG_CANONICAL_CACHE_ROOT", f"{TNG_SCRATCH_ROOT}/cache")
CANONICAL_OUTPUT_ROOT = _env("TNG_CANONICAL_OUTPUT_ROOT", f"{TNG_SCRATCH_ROOT}/outputs")
CANONICAL_FIGURE_ROOT = _env("TNG_CANONICAL_FIGURE_ROOT", f"{TNG_SCRATCH_ROOT}/figures")
