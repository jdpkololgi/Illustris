"""Compatibility wrapper for migrated module."""

from workflows.jraph.hyperparameter_tuning import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy

    runpy.run_module("workflows.jraph.hyperparameter_tuning", run_name="__main__")
