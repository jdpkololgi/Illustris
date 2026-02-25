"""Centralized path configuration for TNG/Illustris workflows.

Phase 1 goal: keep existing defaults unchanged while allowing env overrides.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


PROJECT_DIR = _env("TNG_ILLUSTRIS_PROJECT_DIR", str(Path(__file__).resolve().parent))

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
