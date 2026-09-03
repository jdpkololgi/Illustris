#!/usr/bin/env python3
"""Freeze the complete truth-free P12 prediction set before blind opening."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from workflows.abacus_tweb.p8_deterministic_common import atomic_json
from workflows.sbi.p12_production_contract import freeze_blind_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--method-selection", type=Path, required=True)
    parser.add_argument("--prediction-manifest", type=Path, action="append", required=True)
    parser.add_argument("--deterministic-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen blind contract: {args.output}")
    marker = freeze_blind_predictions(
        candidate_marker=args.candidate,
        method_selection_marker=args.method_selection,
        prediction_manifests=args.prediction_manifest,
        deterministic_contract=args.deterministic_contract,
    )
    atomic_json(args.output, marker)
    print(json.dumps(marker, indent=2), flush=True)


if __name__ == "__main__":
    main()
