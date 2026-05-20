"""DESI-aligned row selection for Abacus CutSky BGS mock catalogs.

Used anywhere the graph / SBI / validation stack applies the standard CutSky
survey footprint flags ``IN_Y1`` / ``IN_Y5`` together with the BGS bright
apparent magnitude limit on ``R_MAG_APP``.
"""

from __future__ import annotations

import numpy as np

# BGS bright: observed apparent r magnitude (instrument / targeting spec).
R_MAG_APP_BRIGHT_LT = 19.5


def cutsky_desi_bgs_mock_mask(table: np.ndarray) -> np.ndarray:
    """Return a per-row boolean mask for the DESI BGS bright mock selection.

    Selection is ``(IN_Y1 == 1) | (IN_Y5 == 1)`` **and** ``R_MAG_APP < R_MAG_APP_BRIGHT_LT``.

    Column names are resolved case-insensitively against ``table.dtype.names``.
    ``IN_Y1``, ``IN_Y5``, and ``R_MAG_APP`` must all be present.
    """
    names_upper = {name.upper(): name for name in table.dtype.names}
    in_y1 = names_upper.get("IN_Y1")
    in_y5 = names_upper.get("IN_Y5")
    r_col = names_upper.get("R_MAG_APP")
    if in_y1 is None or in_y5 is None:
        raise KeyError(
            "IN_Y1 and IN_Y5 are required for DESI-aligned CutSky mock selection "
            f"(found IN_Y1={in_y1 is not None}, IN_Y5={in_y5 is not None})."
        )
    if r_col is None:
        raise KeyError(
            "R_MAG_APP is required when applying DESI-aligned (Y1|Y5 & magnitude) mock selection."
        )
    y1y5 = (np.asarray(table[in_y1]) == 1) | (np.asarray(table[in_y5]) == 1)
    rmag = np.asarray(table[r_col], dtype=np.float64)
    return y1y5 & (rmag < float(R_MAG_APP_BRIGHT_LT))
