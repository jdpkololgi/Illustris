"""Compatibility wrapper for migrated module."""

import warnings

# DEPRECATION SHIM WARNING
warnings.warn(
    "jraph_sbi_flowjax_two_stage.py is a deprecated compatibility shim. Use `legacy.sbi.jraph_sbi_flowjax_two_stage` directly.",
    FutureWarning,
    stacklevel=2,
)


if __name__ == "__main__":
    import runpy

    runpy.run_module("legacy.sbi.jraph_sbi_flowjax_two_stage", run_name="__main__")
else:
    from legacy.sbi.jraph_sbi_flowjax_two_stage import *  # noqa: F401,F403
