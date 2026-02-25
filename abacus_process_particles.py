"""Compatibility wrapper for migrated legacy script."""

from legacy.abacus_process_particles import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy

    runpy.run_module("legacy.abacus_process_particles", run_name="__main__")
