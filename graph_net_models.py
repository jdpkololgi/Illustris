"""Compatibility wrapper for migrated module."""

import warnings

# DEPRECATION SHIM WARNING
warnings.warn(
    "graph_net_models.py is a deprecated compatibility shim. Use `shared.graph_net_models` directly.",
    FutureWarning,
    stacklevel=2,
)


from shared.graph_net_models import *  # noqa: F401,F403
