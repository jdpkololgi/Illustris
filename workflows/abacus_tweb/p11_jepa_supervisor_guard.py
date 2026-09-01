#!/usr/bin/env python3
"""Fail-closed provenance guard for the persistent P11 JEPA supervisor.

The cheap ``marker``, ``preallocation`` and ``terminal`` modes are safe to run
before requesting an allocation.  The ``checkpoint`` and ``complete`` modes are
intended for the compute node: they rehash the registered source and data
contracts and reload the Torch checkpoint on CPU before another interactive
allocation may be requested.

This guard never constructs or opens ph001.  It accepts only the frozen
ph002--ph005 training, ph006 selection, ph001-sealed role contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = REPO_ROOT / "configs/p11_paired_degrade_jepa_v1.json"
DEFAULT_RUN_DIR = Path(
    "/pscratch/sd/d/dkololgi/abacus/p10_multiphase/"
    "p11_factorial_views_v1/training/paired_degrade_jepa_v1/"
    "paired_degrade_jepa_v1/jepa/seed_42"
)
TRAINING_PHASES = ("ph002", "ph003", "ph004", "ph005")
VALIDATION_PHASE = "ph006"
SEALED_PHASE = "ph001"
CANARY_MINIMUM_STEP = 500
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _load_json(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"required supervisor artifact is absent: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"supervisor artifact is not a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _require_phase_contract(contract: dict) -> None:
    split = contract.get("phase_split", {})
    if tuple(split.get("training", ())) != TRAINING_PHASES:
        raise RuntimeError("P11 supervisor training phases are not ph002--ph005")
    if split.get("validation_and_selection") != VALIDATION_PHASE:
        raise RuntimeError("P11 supervisor selection phase is not ph006")
    if split.get("sealed_blind_test") != SEALED_PHASE:
        raise RuntimeError("P11 supervisor blind phase is not sealed ph001")
    if contract.get("scientific_guards", {}).get("ph001_may_be_opened"):
        raise RuntimeError("P11 contract permits sealed-phase access")


def validate_canary_marker(run_dir: Path) -> dict:
    """Validate only small JSON; this gate runs before every salloc request."""
    run_dir = Path(run_dir)
    marker = _load_json(run_dir / "TECHNICAL_CANARY_COMPLETE.json")
    if marker.get("schema_version") != "p11-jepa-technical-canary-v1":
        raise RuntimeError("unsupported P11 technical-canary marker")
    if marker.get("arm") != "jepa" or not marker.get("pass"):
        raise RuntimeError("P11 JEPA technical canary is absent or did not pass")
    if marker.get("ph001_opened") or not marker.get("teacher_gradient_free"):
        raise RuntimeError("P11 canary violates a sealed-phase or teacher-gradient guard")
    if int(marker.get("global_step", -1)) < CANARY_MINIMUM_STEP:
        raise RuntimeError("P11 canary did not complete 500 optimizer updates")
    gates = marker.get("gates")
    if not isinstance(gates, dict) or not gates or not all(value is True for value in gates.values()):
        raise RuntimeError("one or more P11 technical-canary gates did not pass")
    digest = str(marker.get("data_contract_aggregate_sha256", ""))
    if not SHA256_PATTERN.fullmatch(digest):
        raise RuntimeError("P11 canary data-contract digest is malformed")
    expected_checkpoint = (run_dir / "p11_jepa_checkpoint.pt").resolve()
    if Path(marker.get("checkpoint", "")).resolve() != expected_checkpoint:
        raise RuntimeError("P11 canary marker names an unexpected checkpoint")
    if not expected_checkpoint.is_file():
        raise RuntimeError("P11 canary checkpoint is absent")
    return marker


def _paths_from_manifest(manifest: dict) -> SimpleNamespace:
    arguments = manifest.get("arguments", {})
    required = ("contract_root", "factorial_root", "adapter_contract")
    missing = [name for name in required if not arguments.get(name)]
    if missing:
        raise RuntimeError(f"P11 run manifest is missing path arguments: {missing}")
    return SimpleNamespace(**{name: Path(arguments[name]) for name in required})


def _assert_no_blind_artifact(data_contract: dict) -> None:
    files = data_contract.get("files", {})
    if not isinstance(files, dict) or not files:
        raise RuntimeError("P11 frozen data contract has no artifact inventory")
    for name, record in files.items():
        path = Path(str(record.get("path", "")))
        if name.startswith(f"{SEALED_PHASE}_") or SEALED_PHASE in path.parts:
            raise RuntimeError(f"sealed ph001 artifact entered the P11 data contract: {name}")


def _validate_stored_contract_digest(data_contract: dict) -> None:
    """Validate the signed inventory without reading its pscratch artifacts.

    ``preallocation`` runs on a login node and must remain lightweight.  Some
    registered arrays are tens of MB per phase, so their live bytes are rehashed
    only by :func:`validate_frozen_run` inside the allocated compute worker.
    Here we validate the canonical inventory digest and its record schema, then
    cross-check that digest against the canary marker and run manifest.
    """
    files = data_contract.get("files", {})
    if not isinstance(files, dict) or not files:
        raise RuntimeError("P11 frozen data contract has no artifact inventory")
    for name, record in sorted(files.items()):
        if not isinstance(record, dict):
            raise RuntimeError(f"P11 data-contract record is malformed: {name}")
        if not str(record.get("path", "")):
            raise RuntimeError(f"P11 data-contract record has no path: {name}")
        if not isinstance(record.get("bytes"), int) or int(record["bytes"]) < 0:
            raise RuntimeError(f"P11 data-contract byte count is malformed: {name}")
        if not SHA256_PATTERN.fullmatch(str(record.get("sha256", ""))):
            raise RuntimeError(f"P11 data-contract SHA256 is malformed: {name}")
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    aggregate = hashlib.sha256(canonical).hexdigest()
    if aggregate != data_contract.get("aggregate_sha256"):
        raise RuntimeError("P11 frozen data aggregate digest is invalid")


def validate_preallocation(run_dir: Path, contract_path: Path) -> dict:
    """Cheap exact revision/source/stored-digest guard before requesting a GPU."""
    run_dir = Path(run_dir)
    contract_path = Path(contract_path)
    marker = validate_canary_marker(run_dir)
    contract = _load_json(contract_path)
    _require_phase_contract(contract)
    manifest = _load_json(run_dir / "run_manifest.json")
    if manifest.get("schema_version") != "p11-paired-degrade-jepa-run-v1":
        raise RuntimeError("unsupported P11 run manifest")
    if manifest.get("arm") != "jepa" or int(manifest.get("seed", -1)) != 42:
        raise RuntimeError("P11 supervisor accepts only the frozen JEPA seed-42 arm")
    if tuple(manifest.get("training_phases", ())) != TRAINING_PHASES:
        raise RuntimeError("P11 run manifest training phases changed")
    if manifest.get("validation_and_selection_phase") != VALIDATION_PHASE:
        raise RuntimeError("P11 run manifest selection phase changed")
    if manifest.get("sealed_blind_phase") != SEALED_PHASE:
        raise RuntimeError("P11 run manifest blind phase changed")
    if manifest.get("blind_truth_accessed"):
        raise RuntimeError("P11 run manifest reports sealed-phase truth access")
    current_revision = _git_head(REPO_ROOT)
    if manifest.get("git_revision_at_launch") != current_revision:
        raise RuntimeError("repository revision differs from the canary launch")
    if manifest.get("contract_sha256") != _sha256(contract_path):
        raise RuntimeError("P11 JEPA contract changed since canary launch")
    sources = manifest.get("source_contract", {})
    if not isinstance(sources, dict) or not sources:
        raise RuntimeError("P11 run manifest has no frozen source inventory")
    current_sources = {
        name: _sha256(REPO_ROOT / name)
        for name in sources
    }
    if current_sources != sources:
        raise RuntimeError("P11 source hashes changed since canary launch")
    stored_data = _load_json(run_dir / "FROZEN_DATA_CONTRACT.json")
    if stored_data.get("aggregate_sha256") != marker["data_contract_aggregate_sha256"]:
        raise RuntimeError("stored P11 data digest differs from the passing canary")
    if manifest.get("data_contract") != stored_data:
        raise RuntimeError("run manifest and stored P11 data contract disagree")
    _assert_no_blind_artifact(stored_data)
    _validate_stored_contract_digest(stored_data)
    return {
        "mode": "preallocation",
        "pass": True,
        "run_dir": str(run_dir),
        "git_revision": current_revision,
        "contract_sha256": manifest["contract_sha256"],
        "data_contract_aggregate_sha256": stored_data["aggregate_sha256"],
        "global_step": int(marker["global_step"]),
        "sealed_phase": SEALED_PHASE,
        "sealed_phase_opened": False,
    }


def validate_frozen_run(run_dir: Path, contract_path: Path) -> tuple[dict, dict, dict, object]:
    """Recompute all current source/data digests and compare with the canary run."""
    # Heavy project imports are deliberately lazy so marker-only mode remains a
    # lightweight pre-allocation check.
    from workflows.abacus_tweb import p11_jepa_canary as implementation

    run_dir = Path(run_dir)
    contract_path = Path(contract_path)
    marker = validate_canary_marker(run_dir)
    contract = implementation.load_contract(contract_path)
    _require_phase_contract(contract)
    manifest = _load_json(run_dir / "run_manifest.json")
    if manifest.get("schema_version") != "p11-paired-degrade-jepa-run-v1":
        raise RuntimeError("unsupported P11 run manifest")
    if manifest.get("arm") != "jepa" or int(manifest.get("seed", -1)) != 42:
        raise RuntimeError("P11 supervisor accepts only the frozen JEPA seed-42 arm")
    if tuple(manifest.get("training_phases", ())) != TRAINING_PHASES:
        raise RuntimeError("P11 run manifest training phases changed")
    if manifest.get("validation_and_selection_phase") != VALIDATION_PHASE:
        raise RuntimeError("P11 run manifest selection phase changed")
    if manifest.get("sealed_blind_phase") != SEALED_PHASE:
        raise RuntimeError("P11 run manifest blind phase changed")
    if manifest.get("blind_truth_accessed") or not manifest.get("student_only_deployment"):
        raise RuntimeError("P11 run manifest violates deployment or blind-phase guards")

    current_revision = _git_head(REPO_ROOT)
    if manifest.get("git_revision_at_launch") != current_revision:
        raise RuntimeError(
            "repository revision differs from the canary launch; refuse mixed-revision resume"
        )
    contract_hash = _sha256(contract_path)
    if manifest.get("contract_sha256") != contract_hash:
        raise RuntimeError("P11 JEPA contract changed since canary launch")
    current_sources = implementation.source_contract(contract_path)
    if manifest.get("source_contract") != current_sources:
        raise RuntimeError("P11 source hashes changed since canary launch")

    paths = _paths_from_manifest(manifest)
    current_data = implementation.frozen_data_contract(paths, contract)
    stored_data = _load_json(run_dir / "FROZEN_DATA_CONTRACT.json")
    if stored_data != current_data or manifest.get("data_contract") != current_data:
        raise RuntimeError("P11 data products or transformations changed since canary launch")
    if marker["data_contract_aggregate_sha256"] != current_data.get("aggregate_sha256"):
        raise RuntimeError("P11 canary/data-contract digests disagree")
    _assert_no_blind_artifact(current_data)
    return marker, manifest, current_data, paths


def validate_checkpoint(run_dir: Path, contract_path: Path) -> dict:
    from workflows.abacus_tweb import p11_factorial_training as p11_impl
    from workflows.abacus_tweb import p11_jepa_canary as implementation
    from workflows.abacus_tweb.p10_training_contract import validate_resume_state

    run_dir = Path(run_dir)
    marker, manifest, current_data, paths = validate_frozen_run(run_dir, contract_path)
    checkpoint_path = run_dir / "p11_jepa_checkpoint.pt"
    state = implementation.torch_load(checkpoint_path, "cpu")
    if state.get("schema_version") != "p11-paired-degrade-jepa-checkpoint-v1":
        raise RuntimeError("unsupported P11 JEPA checkpoint")
    if state.get("source_contract") != manifest["source_contract"]:
        raise RuntimeError("P11 checkpoint source hashes changed")
    if state.get("frozen_execution") != manifest["frozen_execution"]:
        raise RuntimeError("P11 checkpoint frozen execution changed")
    if state.get("data_contract") != current_data:
        raise RuntimeError("P11 checkpoint data contract changed")
    if int(state.get("global_step", -1)) < int(marker["global_step"]):
        raise RuntimeError("P11 checkpoint predates the passing canary")
    for name in ("model_state", "optimizer_state", "scheduler_state", "resume"):
        if name not in state:
            raise RuntimeError(f"P11 checkpoint is missing {name}")
    resume = state["resume"]
    if int(resume.get("seed", -1)) != 42:
        raise RuntimeError("P11 checkpoint seed changed")

    loader = p11_impl.P11DensePhaseBalancedLoader(
        paths.contract_root,
        factorial_root=paths.factorial_root,
        adapter_contract=paths.adapter_contract,
    )
    if tuple(loader.training_phases) != TRAINING_PHASES:
        raise RuntimeError("runtime P11 training phases changed")
    if loader.validation_phase != VALIDATION_PHASE or loader.blind_phase != SEALED_PHASE:
        raise RuntimeError("runtime P11 visible/sealed roles changed")
    refs = loader.training_epoch(seed=42, epoch=int(resume["epoch"]))
    validate_resume_state(resume, refs)
    return {
        "mode": "checkpoint",
        "pass": True,
        "run_dir": str(run_dir),
        "git_revision": manifest["git_revision_at_launch"],
        "contract_sha256": manifest["contract_sha256"],
        "data_contract_aggregate_sha256": current_data["aggregate_sha256"],
        "global_step": int(state["global_step"]),
        "resume_epoch": int(resume["epoch"]),
        "resume_cursor": int(resume["cursor"]),
        "sealed_phase": SEALED_PHASE,
        "sealed_phase_opened": False,
    }


def _validate_completion_payload(
    *, run_dir: Path, marker: dict, manifest: dict, data_contract: dict
) -> dict:
    complete = _load_json(Path(run_dir) / "P11_MATCHED_ARM_COMPLETE.json")
    if complete.get("status") not in {"TRAINING_COMPLETE", "CONVERGED_EARLY_STOP"}:
        raise RuntimeError("P11 JEPA completion marker has no accepted terminal status")
    if complete.get("arm") != "jepa" or int(complete.get("seed", -1)) != 42:
        raise RuntimeError("P11 completion marker belongs to another arm or seed")
    if complete.get("ph001_opened") or complete.get("blind_truth_accessed"):
        raise RuntimeError("P11 completion marker reports sealed-phase access")
    if complete.get("source_contract") != manifest["source_contract"]:
        raise RuntimeError("P11 completion source hashes changed")
    if complete.get("data_contract") != data_contract:
        raise RuntimeError("P11 completion data contract changed")
    if complete.get("frozen_execution") != manifest["frozen_execution"]:
        raise RuntimeError("P11 completion execution contract changed")
    if int(complete.get("global_steps", -1)) < int(marker["global_step"]):
        raise RuntimeError("P11 completion predates the passing canary")
    if not complete.get("history"):
        raise RuntimeError("P11 completion has no full-epoch scientific history")
    return {
        "mode": "complete",
        "pass": True,
        "run_dir": str(run_dir),
        "git_revision": manifest["git_revision_at_launch"],
        "contract_sha256": manifest["contract_sha256"],
        "data_contract_aggregate_sha256": data_contract["aggregate_sha256"],
        "global_steps": int(complete["global_steps"]),
        "epochs_completed": int(complete["epochs_completed"]),
        "status": complete["status"],
        "sealed_phase": SEALED_PHASE,
        "sealed_phase_opened": False,
    }


def validate_stored_complete(run_dir: Path, contract_path: Path) -> dict:
    """Lightweight terminal check for the persistent login-node supervisor.

    The compute worker has already run :func:`validate_complete`, including the
    live artifact rehash.  This check only ties the terminal marker back to the
    exact canary, revision, source, contract and signed data inventory before a
    subsequent supervisor treats the arm as done.
    """
    run_dir = Path(run_dir)
    validate_preallocation(run_dir, contract_path)
    marker = validate_canary_marker(run_dir)
    manifest = _load_json(run_dir / "run_manifest.json")
    stored_data = _load_json(run_dir / "FROZEN_DATA_CONTRACT.json")
    report = _validate_completion_payload(
        run_dir=run_dir,
        marker=marker,
        manifest=manifest,
        data_contract=stored_data,
    )
    report["mode"] = "terminal"
    report["live_data_rehashed"] = False
    return report


def validate_complete(run_dir: Path, contract_path: Path) -> dict:
    marker, manifest, current_data, _ = validate_frozen_run(run_dir, contract_path)
    report = _validate_completion_payload(
        run_dir=Path(run_dir),
        marker=marker,
        manifest=manifest,
        data_contract=current_data,
    )
    report["live_data_rehashed"] = True
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("marker", "preallocation", "terminal", "checkpoint", "complete"),
        required=True,
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    if args.mode == "marker":
        marker = validate_canary_marker(args.run_dir)
        report = {
            "mode": "marker",
            "pass": True,
            "run_dir": str(args.run_dir),
            "global_step": int(marker["global_step"]),
            "data_contract_aggregate_sha256": marker[
                "data_contract_aggregate_sha256"
            ],
            "sealed_phase": SEALED_PHASE,
            "sealed_phase_opened": False,
        }
    elif args.mode == "preallocation":
        report = validate_preallocation(args.run_dir, args.contract)
    elif args.mode == "terminal":
        report = validate_stored_complete(args.run_dir, args.contract)
    elif args.mode == "checkpoint":
        report = validate_checkpoint(args.run_dir, args.contract)
    else:
        report = validate_complete(args.run_dir, args.contract)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
