"""Compatibility wrapper for migrated module."""

import warnings

# DEPRECATION SHIM WARNING
warnings.warn(
    "jraph_pipeline.py is a deprecated compatibility shim. Use `workflows.jraph.jraph_pipeline` directly.",
    FutureWarning,
    stacklevel=2,
)


if __name__ == "__main__":
    import runpy

    runpy.run_module("workflows.jraph.jraph_pipeline", run_name="__main__")
else:
    from workflows.jraph.jraph_pipeline import *  # noqa: F401,F403
