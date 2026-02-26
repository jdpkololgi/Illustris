"""Compatibility wrapper for migrated module."""

import warnings

# DEPRECATION SHIM WARNING
warnings.warn(
    "getting_started.py is a deprecated compatibility shim. Use `legacy.gcn_paper.getting_started` directly.",
    FutureWarning,
    stacklevel=2,
)


if __name__ == "__main__":
    import runpy

    runpy.run_module("legacy.gcn_paper.getting_started", run_name="__main__")
else:
    from legacy.gcn_paper.getting_started import *  # noqa: F401,F403