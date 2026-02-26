"""Compatibility wrapper for migrated legacy script."""

import warnings

# DEPRECATION SHIM WARNING
warnings.warn(
    "abacus_process_particles.py is a deprecated compatibility shim. Use `legacy.abacus_process_particles` directly.",
    FutureWarning,
    stacklevel=2,
)


from legacy.abacus_process_particles import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy

    runpy.run_module("legacy.abacus_process_particles", run_name="__main__")
