"""Compatibility wrapper for migrated GCN paper module."""

import warnings

# DEPRECATION SHIM WARNING
warnings.warn(
    "Model.py is a deprecated compatibility shim. Use `workflows.gcn_paper.Model` directly.",
    FutureWarning,
    stacklevel=2,
)


from workflows.gcn_paper.Model import *  # noqa: F401,F403
