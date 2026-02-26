"""Compatibility wrapper for migrated module."""

import warnings

# DEPRECATION SHIM WARNING
warnings.warn(
    "debug_eig_order.py is a deprecated compatibility shim. Use `workflows.jraph.debug_eig_order` directly.",
    FutureWarning,
    stacklevel=2,
)


if __name__ == "__main__":
    import runpy

    runpy.run_module("workflows.jraph.debug_eig_order", run_name="__main__")
else:
    from workflows.jraph.debug_eig_order import *  # noqa: F401,F403
