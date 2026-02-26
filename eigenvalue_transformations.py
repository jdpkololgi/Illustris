"""Compatibility wrapper for migrated module."""

import warnings

# DEPRECATION SHIM WARNING
warnings.warn(
    "eigenvalue_transformations.py is a deprecated compatibility shim. Use `shared.eigenvalue_transformations` directly.",
    FutureWarning,
    stacklevel=2,
)


from shared.eigenvalue_transformations import *  # noqa: F401,F403
