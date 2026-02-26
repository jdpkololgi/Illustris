"""Compatibility wrapper for migrated module."""

import warnings

# DEPRECATION SHIM WARNING
warnings.warn(
    "jraph_sbi_pipeline.py is a deprecated compatibility shim. Use `legacy.sbi.jraph_sbi_pipeline` directly.",
    FutureWarning,
    stacklevel=2,
)


if __name__ == "__main__":
    import runpy

    runpy.run_module("legacy.sbi.jraph_sbi_pipeline", run_name="__main__")
else:
    from legacy.sbi.jraph_sbi_pipeline import *  # noqa: F401,F403
