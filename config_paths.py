"""Back-compat import shim for legacy workflow scripts.

Historically, some scripts imported `config_paths` from the TNG/Illustris
package root. The canonical configuration now lives in `shared.config_paths`.
This module re-exports the shared constants to avoid breaking older entrypoints.
"""

from __future__ import annotations

# NOTE: This file must work in two execution styles:
# 1) As part of a package import (e.g. `import TNG.Illustris.config_paths`)
# 2) As a top-level module when running scripts that add `TNG/Illustris` to sys.path
#    (common in HPC workflows). In that case, relative imports (".shared") fail.
try:
    # Typical workflow case: `TNG/Illustris` is on sys.path, so `shared` is importable.
    from shared.config_paths import (  # type: ignore  # noqa: F401
        ABACUS_BASE,
        ABACUS_BASE_LOCAL,
        ABACUS_CARTESIAN_OUTPUT,
        ABACUS_MOCKS_WITH_EIGS_DIR,
        ABACUS_SLAB_DIR,
        ABACUS_TWEB_OUTPUT_DIR,
        CANONICAL_CACHE_ROOT,
        CANONICAL_FIGURE_ROOT,
        CANONICAL_OUTPUT_ROOT,
        CUTSKY_Z0200_PATH,
        DK_SCRATCH_ROOT,
        MOCKS_BASE,
        PROJECT_DIR,
        TNG_LOG_DIR,
        TNG_SCRATCH_ROOT,
    )
except ModuleNotFoundError:  # pragma: no cover
    # Package-style fallback.
    from .shared.config_paths import (  # noqa: F401
    ABACUS_BASE,
    ABACUS_BASE_LOCAL,
    ABACUS_CARTESIAN_OUTPUT,
    ABACUS_MOCKS_WITH_EIGS_DIR,
    ABACUS_SLAB_DIR,
    ABACUS_TWEB_OUTPUT_DIR,
    CANONICAL_CACHE_ROOT,
    CANONICAL_FIGURE_ROOT,
    CANONICAL_OUTPUT_ROOT,
    CUTSKY_Z0200_PATH,
    DK_SCRATCH_ROOT,
    MOCKS_BASE,
    PROJECT_DIR,
    TNG_LOG_DIR,
    TNG_SCRATCH_ROOT,
    )

__all__ = [
    "ABACUS_BASE",
    "ABACUS_BASE_LOCAL",
    "ABACUS_CARTESIAN_OUTPUT",
    "ABACUS_MOCKS_WITH_EIGS_DIR",
    "ABACUS_SLAB_DIR",
    "ABACUS_TWEB_OUTPUT_DIR",
    "CANONICAL_CACHE_ROOT",
    "CANONICAL_FIGURE_ROOT",
    "CANONICAL_OUTPUT_ROOT",
    "CUTSKY_Z0200_PATH",
    "DK_SCRATCH_ROOT",
    "MOCKS_BASE",
    "PROJECT_DIR",
    "TNG_LOG_DIR",
    "TNG_SCRATCH_ROOT",
]

