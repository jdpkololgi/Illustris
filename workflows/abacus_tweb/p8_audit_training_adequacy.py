#!/usr/bin/env python3
"""Audit whether the P8 smoke runs had enough exposure to support a model gate.

The current G-PATCH and U-PATCH trainers use a dedicated NumPy generator only
for selecting a core at each optimization step. This makes the historical
sampling sequence exactly reproducible from the frozen seed and core weights.
The audit deliberately assesses optimization exposure, not prediction quality.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p8_deterministic_common import atomic_json


P8_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p8_deterministic_v1")
P4_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest")


def audit_run(
    p8_root: Path,
    model: str,
    rotation: int,
    seed: int,
    dominant_shell_by_core: np.ndarray,
) -> dict:
    slug = model[0].lower() + "_patch"
    summary_path = (
        p8_root / slug / f"rotation_{rotation}" / f"seed_{seed}" / "screen_summary.json"
    )
    rotation_dir = p8_root / f"rotation_{rotation}"
    training_core = np.load(rotation_dir / "training_core_id.npy")
    training_weight = np.load(rotation_dir / "training_core_weight.npy").astype(np.float64)
    summary = json.loads(summary_path.read_text())
    steps = int(summary["steps_run"])

    # Exact replay of the sampler used by both historical P8 trainers. Neither
    # trainer uses this Generator for augmentation or any other operation.
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        training_core,
        size=steps,
        replace=True,
        p=training_weight / training_weight.sum(),
    )
    unique, counts = np.unique(draws, return_counts=True)
    unique_selector = np.isin(training_core, unique)
    eligible_shell = dominant_shell_by_core[training_core]
    drawn_shell = dominant_shell_by_core[draws]
    unique_shell = dominant_shell_by_core[unique]

    return {
        "model": model,
        "rotation": int(rotation),
        "seed": int(seed),
        "steps": steps,
        "validation_checkpoints": int(len(summary.get("history", []))),
        "eligible_training_cores": int(len(training_core)),
        "unique_training_cores_seen": int(len(unique)),
        "unique_core_fraction": float(len(unique) / len(training_core)),
        "sampled_weight_mass_fraction": float(
            training_weight[unique_selector].sum() / training_weight.sum()
        ),
        "maximum_core_repeats": int(counts.max()),
        "draws_by_dominant_shell": np.bincount(drawn_shell, minlength=4).tolist(),
        "unique_cores_seen_by_dominant_shell": np.bincount(
            unique_shell, minlength=4
        ).tolist(),
        "eligible_cores_by_dominant_shell": np.bincount(
            eligible_shell, minlength=4
        ).tolist(),
        "screen_summary": str(summary_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p8-root", type=Path, default=P8_ROOT)
    parser.add_argument("--p4-root", type=Path, default=P4_ROOT)
    parser.add_argument("--rotations", type=int, nargs="+", default=(0, 2))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cores = np.load(args.p4_root / "cores.npz")
    dominant_shell = np.argmax(cores["active_count_by_shell"], axis=1).astype(np.int8)
    runs = [
        audit_run(args.p8_root, model, rotation, args.seed, dominant_shell)
        for model in ("G-PATCH", "U-PATCH")
        for rotation in args.rotations
    ]
    minimum_unique = min(row["unique_core_fraction"] for row in runs)
    minimum_checkpoints = min(row["validation_checkpoints"] for row in runs)
    payload = {
        "schema_version": 1,
        "stage": "P8 short-screen optimization adequacy audit",
        "status": "INADEQUATE_FOR_SCIENTIFIC_MODEL_GATE",
        "reason": (
            "No run completed one exposure of all eligible training cores and each "
            "run evaluated validation only once, so convergence and early stopping "
            "were not measured."
        ),
        "minimum_unique_core_fraction": float(minimum_unique),
        "minimum_validation_checkpoints": int(minimum_checkpoints),
        "runs": runs,
        "required_recovery": {
            "sampling": "complete exposure-aware patch epochs",
            "validation": "complete blocked validation at every epoch decision",
            "convergence": "at least three validation points and a registered plateau rule",
        },
    }
    output = args.p8_root / "training_adequacy.json"
    atomic_json(output, payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
