"""Compatibility wrapper for migrated module."""

import warnings

# DEPRECATION SHIM WARNING
warnings.warn(
    "hyperparameter_tuning.py is a deprecated compatibility shim. Use `workflows.jraph.hyperparameter_tuning` directly.",
    FutureWarning,
    stacklevel=2,
)


from workflows.jraph.hyperparameter_tuning import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy

    runpy.run_module("workflows.jraph.hyperparameter_tuning", run_name="__main__")
