"""Compatibility wrapper for migrated GCN paper entrypoint."""

import warnings

# DEPRECATION SHIM WARNING
warnings.warn(
    "gcn_pipeline.py is a deprecated compatibility shim. Use `workflows.gcn_paper.gcn_pipeline` directly.",
    FutureWarning,
    stacklevel=2,
)


from workflows.gcn_paper.gcn_pipeline import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy

    runpy.run_module("workflows.gcn_paper.gcn_pipeline", run_name="__main__")
