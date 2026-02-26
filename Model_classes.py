"""Compatibility wrapper for migrated GCN paper module."""

import warnings

# DEPRECATION SHIM WARNING
warnings.warn(
    "Model_classes.py is a deprecated compatibility shim. Use `workflows.gcn_paper.Model_classes` directly.",
    FutureWarning,
    stacklevel=2,
)


from workflows.gcn_paper.Model_classes import *  # noqa: F401,F403
