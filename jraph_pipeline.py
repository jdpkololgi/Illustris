"""Compatibility wrapper for migrated module."""

if __name__ == "__main__":
    import runpy

    runpy.run_module("workflows.jraph.jraph_pipeline", run_name="__main__")
else:
    from workflows.jraph.jraph_pipeline import *  # noqa: F401,F403
