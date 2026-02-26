"""Compatibility wrapper for migrated module."""

if __name__ == "__main__":
    import runpy

    runpy.run_module("legacy.gcn_paper.getting_started", run_name="__main__")
else:
    from legacy.gcn_paper.getting_started import *  # noqa: F401,F403