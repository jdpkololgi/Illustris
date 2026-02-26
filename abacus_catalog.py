"""Compatibility wrapper for moved Abacus workflow module.

Canonical location:
`workflows/abacus_tweb/abacus_catalog.py`
"""

import warnings

# DEPRECATION SHIM WARNING
warnings.warn(
    "abacus_catalog.py is a deprecated compatibility shim. Use `workflows.abacus_tweb.abacus_catalog` directly.",
    FutureWarning,
    stacklevel=2,
)


from workflows.abacus_tweb.abacus_catalog import *  # noqa: F401,F403


if __name__ == "__main__":
    path = CUTSKY_Z0200_PATH
    AbacusCatalog(path).save_cartesian_coords(ABACUS_CARTESIAN_OUTPUT)
