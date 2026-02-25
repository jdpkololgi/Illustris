"""Compatibility wrapper for moved Abacus workflow module.

Canonical location:
`workflows/abacus_tweb/abacus_catalog.py`
"""

from workflows.abacus_tweb.abacus_catalog import *  # noqa: F401,F403


if __name__ == "__main__":
    path = CUTSKY_Z0200_PATH
    AbacusCatalog(path).save_cartesian_coords(ABACUS_CARTESIAN_OUTPUT)
