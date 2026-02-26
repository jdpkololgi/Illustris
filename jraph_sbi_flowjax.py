"""Compatibility wrapper for migrated module."""

import warnings

# DEPRECATION SHIM WARNING
warnings.warn(
    "jraph_sbi_flowjax.py is a deprecated compatibility shim. Use `workflows.sbi.jraph_sbi_flowjax` directly.",
    FutureWarning,
    stacklevel=2,
)


from workflows.sbi.jraph_sbi_flowjax import *  # noqa: F401,F403

if __name__ == "__main__":
    main()
