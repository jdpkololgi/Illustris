#!/usr/bin/env python3
"""Validate representative P10 R2 patches before the GPU throughput gate."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p10_build_r2_assignment_overlays import R2_MODEL_CHANNELS
from workflows.abacus_tweb.p10_r2_training_contract import P10AssignmentResponseLoader
from workflows.abacus_tweb.p10_training_contract import atomic_json, sha256


ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p10_multiphase")


def main() -> None:
    contract = ROOT / "training_contract_r2_assignment"
    loader = P10AssignmentResponseLoader(contract, include_blind=False)
    phases = loader.training_phases + (loader.validation_phase,)
    rows = {}
    for phase in phases:
        if phase == loader.validation_phase:
            core_id = loader.validation_refs()[0].core_id
        else:
            core_id = next(
                ref.core_id for ref in loader.training_epoch(seed=42, epoch=1) if ref.phase == phase
            )
        patch = loader.field_adapter(phase).extract(
            core_id,
            24,
            R2_MODEL_CHANNELS + ("exposure_binary",),
            alignment_voxels=8,
        )
        at = {name: index for index, name in enumerate(patch.channel_names)}
        values = patch.values
        support = values[at["exposure_binary"]] > 0.5
        defined = values[at["c_fibre_defined"]] > 0.5
        tileloc = values[at["c_fibre_tileloc"]]
        tiles = values[at["c_fibre_tiles"]]
        gates = {
            "channel_order_exact": tuple(patch.channel_names[:-1]) == tuple(R2_MODEL_CHANNELS),
            "values_finite": bool(np.isfinite(values).all()),
            "response_range": bool(
                np.all((tileloc >= 0.0) & (tileloc <= 1.0))
                and np.all((tiles >= 0.0) & (tiles <= 1.0))
            ),
            "undefined_response_neutral_on_support": bool(
                np.all(tileloc[support & ~defined] == 1.0)
                and np.all(tiles[support & ~defined] == 1.0)
            ),
            "authoritative_rows_nonzero": len(patch.authoritative_parent_id) > 0,
        }
        rows[phase] = {
            "core_id": int(core_id),
            "shape": list(values.shape),
            "authoritative_rows": int(len(patch.authoritative_parent_id)),
            "supported_voxels": int(support.sum()),
            "defined_response_voxels": int(np.count_nonzero(support & defined)),
            "gates": gates,
            "pass": bool(all(gates.values())),
        }
        if not rows[phase]["pass"]:
            raise RuntimeError(f"{phase}: R2 loader smoke failed: {gates}")
    report = {
        "schema_version": "p10-r2-loader-smoke-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract": str(contract / "TRAINING_LOADER_READY.json"),
        "contract_sha256": sha256(contract / "TRAINING_LOADER_READY.json"),
        "phases": rows,
        "ph001_opened": False,
        "pass": bool(all(row["pass"] for row in rows.values())),
    }
    output = contract / "R2_LOADER_SMOKE.json"
    atomic_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
