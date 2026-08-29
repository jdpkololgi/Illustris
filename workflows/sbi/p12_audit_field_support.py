#!/usr/bin/env python3
"""Audit field-support distance semantics in P12 phase assignments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase")
PHASES = ("ph000", "ph002", "ph003", "ph004", "ph005", "ph006")


def contract_root(phase: str) -> Path:
    if phase == "ph006":
        return ROOT / "training_contract"
    return ROOT / "p12_crossfit_contracts" / f"omit_{phase}"


def main() -> None:
    for phase in PHASES:
        root = contract_root(phase)
        contract = json.loads(
            (root / "phases" / phase / "phase_contract.json").read_text()
        )
        assignment = np.load(contract["inputs"]["assignment"], mmap_mode="r")
        distance = np.asarray(assignment["field_support_distance_mpc"])
        finite = np.isfinite(distance)
        finite_values = distance[finite]
        print(
            phase,
            "rows", len(distance),
            "finite", int(finite.sum()),
            "nan", int(np.isnan(distance).sum()),
            "posinf", int(np.isposinf(distance).sum()),
            "negative", int((distance < 0).sum()),
            "min_finite", (
                float(np.min(finite_values)) if len(finite_values) else None
            ),
            "max_finite", (
                float(np.max(finite_values)) if len(finite_values) else None
            ),
        )
        for cap in (0, 1):
            for shell in (0, 1, 2, 3):
                mask = (assignment["cap"] == cap) & (
                    assignment["shell"] == shell
                )
                print(
                    " ", cap, shell,
                    "rows", int(mask.sum()),
                    "invalid", int((mask & ~finite).sum()),
                )


if __name__ == "__main__":
    main()
