"""Schema helpers for SBI cache payload validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

BASE_REQUIRED_KEYS = ("graph", "regression_targets", "masks")
SCALER_KEYS = ("target_scaler", "eigenvalue_scaler")


def get_scaler_key(payload: Mapping[str, object]) -> str:
    """Return the scaler key present in payload or raise ValueError."""
    for key in SCALER_KEYS:
        if key in payload:
            return key
    raise ValueError(
        "SBI cache schema missing scaler key: expected one of "
        f"{', '.join(SCALER_KEYS)}."
    )


def validate_sbi_cache_payload(payload: Mapping[str, object]) -> None:
    """Validate minimal cache schema required by SBI workflows."""
    missing = [key for key in BASE_REQUIRED_KEYS if key not in payload]
    if missing:
        raise ValueError(f"SBI cache schema missing keys: {', '.join(missing)}")

    get_scaler_key(payload)

    masks = payload["masks"]
    if not isinstance(masks, Sequence) or len(masks) != 3:
        raise ValueError("SBI cache schema invalid `masks`: expected 3-item sequence.")
