"""Compatibility wrapper for migrated module."""

if __name__ == "__main__":
    import runpy

    runpy.run_module("legacy.sbi.jraph_sbi_pipeline", run_name="__main__")
else:
    from legacy.sbi.jraph_sbi_pipeline import *  # noqa: F401,F403
