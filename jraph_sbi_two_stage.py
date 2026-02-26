"""Compatibility wrapper for migrated module."""

import warnings

# DEPRECATION SHIM WARNING
warnings.warn(
    "jraph_sbi_two_stage.py is a deprecated compatibility shim. Use `workflows.sbi.experimental.jraph_sbi_two_stage` directly.",
    FutureWarning,
    stacklevel=2,
)


from workflows.sbi.experimental.jraph_sbi_two_stage import *  # noqa: F401,F403

if __name__ == "__main__":
    main()
