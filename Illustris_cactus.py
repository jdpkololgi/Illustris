"""Compatibility wrapper for migrated module."""

import warnings

# DEPRECATION SHIM WARNING
warnings.warn(
    "Illustris_cactus.py is a deprecated compatibility shim. Use `workflows.gcn_paper.experimental.Illustris_cactus` directly.",
    FutureWarning,
    stacklevel=2,
)


if __name__ == "__main__":
    import runpy

    runpy.run_module("workflows.gcn_paper.experimental.Illustris_cactus", run_name="__main__")
else:
    from workflows.gcn_paper.experimental.Illustris_cactus import *  # noqa: F401,F403

