"""Compatibility wrapper for moved Abacus workflow module.

Canonical location:
`workflows/abacus_tweb/annotate_cutsky_with_tweb.py`
"""

import warnings

# DEPRECATION SHIM WARNING
warnings.warn(
    "annotate_cutsky_with_tweb.py is a deprecated compatibility shim. Use `workflows.abacus_tweb.annotate_cutsky_with_tweb` directly.",
    FutureWarning,
    stacklevel=2,
)


from workflows.abacus_tweb.annotate_cutsky_with_tweb import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
