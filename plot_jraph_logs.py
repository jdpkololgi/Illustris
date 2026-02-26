"""Compatibility wrapper for migrated module."""

import warnings

# DEPRECATION SHIM WARNING
warnings.warn(
    "plot_jraph_logs.py is a deprecated compatibility shim. Use `workflows.jraph.plot_jraph_logs` directly.",
    FutureWarning,
    stacklevel=2,
)


from workflows.jraph.plot_jraph_logs import *  # noqa: F401,F403

if __name__ == "__main__":
    main()
