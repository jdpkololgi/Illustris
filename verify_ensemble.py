"""Compatibility wrapper for migrated module."""

import warnings

# DEPRECATION SHIM WARNING
warnings.warn(
    "verify_ensemble.py is a deprecated compatibility shim. Use `workflows.jraph.verify_ensemble` directly.",
    FutureWarning,
    stacklevel=2,
)


from workflows.jraph.verify_ensemble import *  # noqa: F401,F403

if __name__ == "__main__":
    main()
