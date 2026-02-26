"""Compatibility wrapper for migrated module."""

if __name__ == "__main__":
    import runpy

    runpy.run_module("workflows.gcn_paper.experimental.Illustris_cactus", run_name="__main__")
else:
    from workflows.gcn_paper.experimental.Illustris_cactus import *  # noqa: F401,F403

