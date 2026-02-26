"""Compatibility wrapper for migrated GCN paper module."""

import warnings

# DEPRECATION SHIM WARNING
warnings.warn(
    "Network_stats.py is a deprecated compatibility shim. Use `workflows.gcn_paper.Network_stats` directly.",
    FutureWarning,
    stacklevel=2,
)


from workflows.gcn_paper.Network_stats import *  # noqa: F401,F403
