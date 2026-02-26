"""Compatibility wrapper for migrated module."""

import warnings

# DEPRECATION SHIM WARNING
warnings.warn(
    "tng_pipeline_paths.py is a deprecated compatibility shim. Use `shared.tng_pipeline_paths` directly.",
    FutureWarning,
    stacklevel=2,
)


from shared.tng_pipeline_paths import *  # noqa: F401,F403
