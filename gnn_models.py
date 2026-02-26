"""Compatibility wrapper for migrated GCN paper module."""

import warnings

# DEPRECATION SHIM WARNING
warnings.warn(
    "gnn_models.py is a deprecated compatibility shim. Use `workflows.gcn_paper.gnn_models` directly.",
    FutureWarning,
    stacklevel=2,
)


from workflows.gcn_paper.gnn_models import *  # noqa: F401,F403
