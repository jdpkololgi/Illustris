"""Compatibility wrapper for moved Abacus workflow module.

Canonical location:
`workflows/abacus_tweb/abacus_cactus_tweb.py`
"""

import warnings

# DEPRECATION SHIM WARNING
warnings.warn(
    "abacus_cactus_tweb.py is a deprecated compatibility shim. Use `workflows.abacus_tweb.abacus_cactus_tweb` directly.",
    FutureWarning,
    stacklevel=2,
)


from workflows.abacus_tweb.abacus_cactus_tweb import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
