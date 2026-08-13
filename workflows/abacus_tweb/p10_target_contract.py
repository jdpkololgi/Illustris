"""Shared precision-aware P10 tidal-target consistency helpers."""
from __future__ import annotations

import numpy as np


def stored_class_consistency(
    eigenvalues: np.ndarray,
    cweb: np.ndarray,
    *,
    threshold: float = 0.2,
) -> dict[str, np.ndarray]:
    """Compare stored eigenvalues with the higher-precision CACTUS class.

    CACTUS classifies its native eigenvalues before the catalogue writer casts
    them to float32.  A value infinitesimally above or below the threshold can
    consequently round to exactly ``float32(threshold)``.  Such a row has lost
    enough precision that the original class cannot be reconstructed from the
    catalogue eigenvalues alone.  It is an allowed, explicitly counted boundary
    ambiguity; a disagreement away from that boundary remains a hard error.
    """
    stored = np.asarray(eigenvalues)
    labels = np.asarray(cweb, dtype=np.int8)
    if stored.ndim != 2 or stored.shape[1] != 3:
        raise ValueError(f"eigenvalues must have shape (N,3), got {stored.shape}")
    if labels.shape != (len(stored),):
        raise ValueError("cweb must have one value per eigenvalue row")
    stored_threshold = np.asarray(threshold, dtype=stored.dtype)
    reconstructed = np.sum(stored > stored_threshold, axis=1).astype(np.int8)
    mismatch = reconstructed != labels
    boundary_ambiguous = np.any(stored == stored_threshold, axis=1)
    return {
        "reconstructed": reconstructed,
        "mismatch": mismatch,
        "boundary_ambiguous": mismatch & boundary_ambiguous,
        "nonboundary_mismatch": mismatch & ~boundary_ambiguous,
    }
