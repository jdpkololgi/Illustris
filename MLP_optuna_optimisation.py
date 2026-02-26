"""Compatibility wrapper for migrated GCN paper entrypoint."""

from workflows.gcn_paper.MLP_optuna_optimisation import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy

    runpy.run_module("workflows.gcn_paper.MLP_optuna_optimisation", run_name="__main__")
