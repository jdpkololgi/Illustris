"""Compatibility wrapper for migrated module."""

import warnings

# DEPRECATION SHIM WARNING
warnings.warn(
    "hdf5_helper.py is a deprecated compatibility shim. Use `shared.hdf5_helper` directly.",
    FutureWarning,
    stacklevel=2,
)


from shared.hdf5_helper import *  # noqa: F401,F403