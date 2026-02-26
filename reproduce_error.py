"""Compatibility wrapper for migrated module."""

import warnings

# DEPRECATION SHIM WARNING
warnings.warn(
    "reproduce_error.py is a deprecated compatibility shim. Use `workflows.jraph.experimental.reproduce_error` directly.",
    FutureWarning,
    stacklevel=2,
)


if __name__ == "__main__":
    import runpy

    runpy.run_module("workflows.jraph.experimental.reproduce_error", run_name="__main__")
else:
    from workflows.jraph.experimental.reproduce_error import *  # noqa: F401,F403
