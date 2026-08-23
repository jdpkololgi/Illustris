#!/usr/bin/env python3
"""Validate representative P10 R3-RF patches before the GPU throughput gate."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p10_r3_random_field_contract import (
    P10RawRandomFieldLoader,
    R3_RF_MODEL_CHANNELS,
)
from workflows.abacus_tweb.p10_training_contract import atomic_json, sha256


ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase")


def main() -> None:
    contract = ROOT / "training_contract_r3_random_field"
    loader = P10RawRandomFieldLoader(contract, include_blind=False)
    phases = loader.training_phases + (loader.validation_phase,)
    rows = {}
    for phase in phases:
        if phase == loader.validation_phase:
            core_id = loader.validation_refs()[0].core_id
        else:
            core_id = next(
                ref.core_id
                for ref in loader.training_epoch(seed=42, epoch=1)
                if ref.phase == phase
            )
        patch = loader.field_adapter(phase).extract(
            core_id,
            24,
            R3_RF_MODEL_CHANNELS,
            alignment_voxels=8,
        )
        at = {name: index for index, name in enumerate(patch.channel_names)}
        values = patch.values
        support = values[at["support_random"]]
        intensity = values[at["expected_counts_random"]]
        response = values[at["angular_response"]]
        unsupported = support < 0.5
        gates = {
            "channel_order_exact": tuple(patch.channel_names) == tuple(R3_RF_MODEL_CHANNELS),
            "six_channel_width": int(values.shape[0]) == 6,
            "values_finite": bool(np.isfinite(values).all()),
            "intensity_nonnegative": bool(np.all(intensity >= 0.0)),
            "support_binary": bool(np.all((support == 0.0) | (support == 1.0))),
            "response_zero_outside_support": bool(
                np.all(intensity[unsupported] == 0.0)
                and np.all(response[unsupported] == 0.0)
            ),
            "authoritative_rows_nonzero": len(patch.authoritative_parent_id) > 0,
        }
        rows[phase] = {
            "core_id": int(core_id),
            "shape": list(values.shape),
            "authoritative_rows": int(len(patch.authoritative_parent_id)),
            "supported_voxels": int(np.count_nonzero(~unsupported)),
            "gates": gates,
            "pass": bool(all(gates.values())),
        }
        if not rows[phase]["pass"]:
            raise RuntimeError(f"{phase}: R3-RF loader smoke failed: {gates}")
    report = {
        "schema_version": "p10-r3-rf-loader-smoke-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract": str(contract / "TRAINING_LOADER_READY.json"),
        "contract_sha256": sha256(contract / "TRAINING_LOADER_READY.json"),
        "phases": rows,
        "ph001_opened": False,
        "pass": bool(all(row["pass"] for row in rows.values())),
    }
    output = contract / "R3_RF_LOADER_SMOKE.json"
    atomic_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

