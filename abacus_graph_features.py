"""Compatibility wrapper for migrated module."""

import warnings

# DEPRECATION SHIM WARNING
warnings.warn(
    "abacus_graph_features.py is a deprecated compatibility shim. Use `workflows.abacus_tweb.abacus_graph_features` directly.",
    FutureWarning,
    stacklevel=2,
)


if __name__ == "__main__":
    import runpy

    runpy.run_module("workflows.abacus_tweb.abacus_graph_features", run_name="__main__")
else:
    from workflows.abacus_tweb.abacus_graph_features import *  # noqa: F401,F403