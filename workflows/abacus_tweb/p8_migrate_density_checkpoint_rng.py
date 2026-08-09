#!/usr/bin/env python3
"""Migrate the frozen D0 checkpoint from all-device to single-device CUDA RNG.

This deliberately narrow migration preserves every scientific and optimizer
state tensor while making the interrupted four-visible-GPU checkpoint resumable
inside the production one-GPU srun contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.abacus_tweb.p8_deterministic_common import atomic_json
from workflows.abacus_tweb.p8_train_density_patch import CHECKPOINT_SCHEMA
from workflows.abacus_tweb.p8_train_patch_recovery import atomic_torch_save, torch_load


DEFAULT_CHECKPOINT = Path(
    "/pscratch/sd/d/dkololgi/abacus/p8_density_phys_v1/d0_runs/"
    "rotation_0/seed_42/scientific_v1/training_checkpoint.pt"
)
SOURCE_SCHEMA = "p8-density-training-checkpoint-v1"
SOURCE_REVISION = "0a95141f90766064378d22cf53b3e7ddb6e85408"
SOURCE_SHA256 = "2c213ca710c8bd2138f949955cc14de90904ab1b40497878acc27c8d9b43439b"
CONFIG_SHA256 = "4b78dc6b4b6ee0979c1236b222478ada6b826eb4560cc24383e2cd63013e6dab"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_digest(value) -> str:
    """Hash nested optimizer/model state independently of torch serialization."""
    digest = hashlib.sha256()

    def update(item) -> None:
        if torch.is_tensor(item):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(b"tensor")
            digest.update(str(array.dtype).encode())
            digest.update(json.dumps(array.shape).encode())
            digest.update(array.tobytes())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"ndarray")
            digest.update(str(array.dtype).encode())
            digest.update(json.dumps(array.shape).encode())
            digest.update(array.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda candidate: str(candidate)):
                update(key)
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                update(child)
        else:
            digest.update(type(item).__name__.encode())
            digest.update(repr(item).encode())

    update(value)
    return digest.hexdigest()


def current_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def require_equal(name: str, actual, expected) -> None:
    if actual != expected:
        raise RuntimeError(f"{name} mismatch: {actual!r} != {expected!r}")


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    report_path = (args.report or checkpoint.with_name("rng_migration_report.json")).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    source_sha = file_sha256(checkpoint)
    require_equal("source checkpoint sha256", source_sha, SOURCE_SHA256)

    state = torch_load(checkpoint, "cpu")
    require_equal("schema", state.get("schema_version"), SOURCE_SCHEMA)
    require_equal("source revision", state.get("git_revision"), SOURCE_REVISION)
    require_equal("model", state.get("model"), "U-DENSITY-PHYS-v1")
    require_equal("rotation", state.get("rotation"), 0)
    require_equal("seed", state.get("seed"), 42)
    require_equal("epoch", state.get("epoch"), 3)
    require_equal("cursor", state.get("cursor"), 5922)
    require_equal("global step", state.get("global_step"), 33250)
    require_equal("config sha256", state.get("config_sha256"), CONFIG_SHA256)
    legacy = state.get("cuda_rng_state_all")
    if not isinstance(legacy, (list, tuple)) or len(legacy) != 4:
        raise RuntimeError("expected the four-state legacy CUDA RNG payload")
    if not all(torch.is_tensor(item) for item in legacy):
        raise RuntimeError("legacy CUDA RNG payload contains a non-tensor value")

    model_digest_before = state_digest(state["model_state"])
    optimizer_digest_before = state_digest(state["optimizer_state"])
    scheduler_digest_before = state_digest(state["scheduler_state"])
    revision = current_revision()
    backup = checkpoint.with_name(
        f"training_checkpoint.pre_rng_schema_v2.{source_sha[:12]}.pt"
    )
    if backup.exists():
        require_equal("existing backup sha256", file_sha256(backup), source_sha)
    else:
        shutil.copy2(checkpoint, backup)
        require_equal("new backup sha256", file_sha256(backup), source_sha)

    state["schema_version"] = CHECKPOINT_SCHEMA
    state["cuda_rng_state"] = legacy[0].clone()
    state["cuda_device_count_at_save"] = len(legacy)
    del state["cuda_rng_state_all"]
    state["git_revision"] = revision
    state["arguments"] = dict(state["arguments"])
    state["arguments"]["git_revision"] = revision
    state["migration"] = {
        "name": "single-device-cuda-rng-v2",
        "source_checkpoint_sha256": source_sha,
        "source_git_revision": SOURCE_REVISION,
        "target_git_revision": revision,
        "scientific_state_changed": False,
        "selected_legacy_cuda_device_index": 0,
    }

    require_equal("model digest", state_digest(state["model_state"]), model_digest_before)
    require_equal(
        "optimizer digest", state_digest(state["optimizer_state"]), optimizer_digest_before
    )
    require_equal(
        "scheduler digest", state_digest(state["scheduler_state"]), scheduler_digest_before
    )
    atomic_torch_save(state, checkpoint)
    migrated_sha = file_sha256(checkpoint)
    restored = torch_load(checkpoint, "cpu")
    require_equal("migrated schema", restored.get("schema_version"), CHECKPOINT_SCHEMA)
    require_equal("migrated revision", restored.get("git_revision"), revision)
    require_equal("restored model digest", state_digest(restored["model_state"]), model_digest_before)
    require_equal(
        "restored optimizer digest",
        state_digest(restored["optimizer_state"]),
        optimizer_digest_before,
    )
    if "cuda_rng_state_all" in restored or not torch.equal(
        restored["cuda_rng_state"], legacy[0]
    ):
        raise RuntimeError("migrated CUDA RNG state failed round-trip validation")

    report = {
        "schema_version": "p8-density-checkpoint-rng-migration-report-v1",
        "status": "PASS",
        "checkpoint": str(checkpoint),
        "backup": str(backup),
        "source_checkpoint_sha256": source_sha,
        "backup_sha256": file_sha256(backup),
        "migrated_checkpoint_sha256": migrated_sha,
        "source_schema": SOURCE_SCHEMA,
        "target_schema": CHECKPOINT_SCHEMA,
        "source_git_revision": SOURCE_REVISION,
        "target_git_revision": revision,
        "epoch": int(restored["epoch"]),
        "cursor": int(restored["cursor"]),
        "global_step": int(restored["global_step"]),
        "legacy_cuda_states": len(legacy),
        "selected_legacy_cuda_device_index": 0,
        "model_state_sha256": model_digest_before,
        "optimizer_state_sha256": optimizer_digest_before,
        "scheduler_state_sha256": scheduler_digest_before,
        "scientific_state_changed": False,
    }
    atomic_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
