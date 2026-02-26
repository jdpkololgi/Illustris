"""Compatibility wrapper for migrated module."""

import warnings

# DEPRECATION SHIM WARNING
warnings.warn(
    "utils.py is a deprecated compatibility shim. Use `shared.utils` directly.",
    FutureWarning,
    stacklevel=2,
)


from shared.utils import *  # noqa: F401,F403
