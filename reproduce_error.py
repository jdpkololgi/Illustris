"""Compatibility wrapper for migrated module."""

if __name__ == "__main__":
    import runpy

    runpy.run_module("workflows.jraph.experimental.reproduce_error", run_name="__main__")
else:
    from workflows.jraph.experimental.reproduce_error import *  # noqa: F401,F403
