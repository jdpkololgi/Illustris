"""Compatibility wrapper for migrated module."""

if __name__ == "__main__":
    import runpy

    runpy.run_module("workflows.jraph.debug_eig_order", run_name="__main__")
else:
    from workflows.jraph.debug_eig_order import *  # noqa: F401,F403
