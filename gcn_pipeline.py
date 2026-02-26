"""Compatibility wrapper for migrated GCN paper entrypoint."""

from workflows.gcn_paper.gcn_pipeline import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy

    runpy.run_module("workflows.gcn_paper.gcn_pipeline", run_name="__main__")
