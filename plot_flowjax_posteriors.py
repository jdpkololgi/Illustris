"""Compatibility wrapper for migrated module."""

import warnings

# DEPRECATION SHIM WARNING
warnings.warn(
    "plot_flowjax_posteriors.py is a deprecated compatibility shim. Use `workflows.sbi.plot_flowjax_posteriors` directly.",
    FutureWarning,
    stacklevel=2,
)


from workflows.sbi.plot_flowjax_posteriors import *  # noqa: F401,F403

if __name__ == "__main__":
    main()
