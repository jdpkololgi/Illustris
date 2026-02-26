"""Compatibility wrapper for migrated GCN paper entrypoint."""

import warnings

# DEPRECATION SHIM WARNING
warnings.warn(
    "gcn_pipeline_postprocess.py is a deprecated compatibility shim. Use `workflows.gcn_paper.gcn_pipeline_postprocess` directly.",
    FutureWarning,
    stacklevel=2,
)


from workflows.gcn_paper.gcn_pipeline_postprocess import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy

    runpy.run_module("workflows.gcn_paper.gcn_pipeline_postprocess", run_name="__main__")
