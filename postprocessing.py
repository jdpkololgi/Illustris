"""Compatibility wrapper for migrated GCN paper entrypoint."""

import warnings

# DEPRECATION SHIM WARNING
warnings.warn(
    "postprocessing.py is a deprecated compatibility shim. Use `workflows.gcn_paper.postprocessing` directly.",
    FutureWarning,
    stacklevel=2,
)


from workflows.gcn_paper.postprocessing import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy

    runpy.run_module("workflows.gcn_paper.postprocessing", run_name="__main__")
