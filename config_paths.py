"""Compatibility wrapper for migrated module."""

import warnings

# DEPRECATION SHIM WARNING
warnings.warn(
    "config_paths.py is a deprecated compatibility shim. Use `shared.config_paths` directly.",
    FutureWarning,
    stacklevel=2,
)


from shared.config_paths import *  # noqa: F401,F403
