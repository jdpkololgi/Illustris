#!/usr/bin/env python3
"""Re-score all frozen D2 references with one identical ph006 score contract.

The historical G1/F3-L2b/F3-L2d reports used different evaluator seeds.  That
changes the fixed voxel/pair subsets used by ``core_joint_scores`` and makes a
nominally paired comparison non-paired.  This one-shot stage rebuilds only the
reports (never samples) from the already-frozen archives with the D2 seed 42.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from workflows.abacus_tweb.p8_deterministic_common import (
    acquire_run_lock,
    atomic_json,
    sha256,
)
from workflows.sbi.p12f3_d2_contract import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    digest,
    utc_now,
    validate_frozen_contract,
    validate_output_root,
)
from workflows.sbi.p12f3_d2_evaluate import label_authoritative_core_bootstrap
from workflows.sbi.p12f3_d2_models import configure_d2_determinism
from workflows.sbi.p12f_common_evaluator import evaluate_records, load_core_record
from workflows.sbi.plot_p12f3_hierarchical_comparison import analyze_archive
from workflows.sbi.p12f3l2_shear_audit import audit_archive


SCHEMA = "p12f3-d2-matched-reference-reports-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _records(archive_path: Path, panel: dict, expected_ids: list[int]) -> tuple[dict, list]:
    archive = json.loads(archive_path.read_text())
    entries = archive.get("entries", ())
    found = [int(row["core_id"]) for row in entries]
    if (
        archive.get("schema_version") != "p12f-sample-archive-v1"
        or not archive.get("pass", True)
        or archive.get("phase") != "ph006"
        or int(archive.get("draws", -1)) != 64
        or archive.get("ph001_opened")
        or found != expected_ids
    ):
        raise RuntimeError(f"unsafe D2 reference archive: {archive_path}")
    metadata = {
        int(row["core_id"]): row for row in panel["selected_core_metadata"]
    }
    records = []
    for entry in entries:
        artifact = Path(entry["path"])
        if "ph001" in str(artifact).lower() or sha256(artifact) != entry["sha256"]:
            raise RuntimeError(f"D2 reference core changed: {artifact}")
        records.append((metadata[int(entry["core_id"])], load_core_record(entry, 64)))
    return archive, records


def main() -> None:
    args = parse_args()
    contract_path = args.contract or args.output_root / "D2_CONTRACT_FROZEN.json"
    contract, config = validate_frozen_contract(contract_path, args.config)
    validate_output_root(contract, args.output_root, contract_path)
    deterministic_runtime = configure_d2_determinism(
        config["reproducibility"], args.device
    )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("D2 matched-reference evaluation requires a compute GPU")
    seed = int(config["evaluation"]["common_evaluator_seed"])
    if seed != 42:
        raise RuntimeError("D2 common evaluator seed changed")
    panel_path = Path(contract["frozen"]["source_paths"]["ph006_panel"])
    panel = json.loads(panel_path.read_text())
    expected_ids = [int(value) for value in panel.get("selected_core_id", ())]
    if (
        panel.get("phase") != "ph006"
        or panel.get("selection_uses_truth")
        or panel.get("ph001_opened")
        or len(expected_ids) != 256
    ):
        raise RuntimeError("unsafe D2 matched-reference panel")

    frozen_references = {}
    for key in config["evaluation"]["matched_reference_methods"]:
        row = contract["frozen"]["reference_contract"][key]
        archive_path = Path(row["archive"])
        if sha256(archive_path) != row["archive_sha256"]:
            raise RuntimeError(f"D2 {key} reference archive changed")
        frozen_references[key] = {
            "archive": str(archive_path.resolve()),
            "archive_sha256": row["archive_sha256"],
            "method": row["method"],
            "core_ids": expected_ids,
        }
    frozen = {
        "contract": str(contract_path.resolve()),
        "contract_sha256": sha256(contract_path),
        "contract_digest": contract["frozen_digest"],
        "deterministic_runtime": deterministic_runtime,
        "panel": str(panel_path.resolve()),
        "panel_sha256": sha256(panel_path),
        "common_evaluator_seed": seed,
        "core_joint_score_subset_seed_rule": "seed_plus_core_id",
        "references": frozen_references,
        "ph001_opened": False,
    }
    frozen_digest = digest(frozen)
    marker_path = args.output_root / "D2_MATCHED_REFERENCE_REPORTS.json"
    lock = acquire_run_lock(
        args.output_root / ".matched_references.lock",
        purpose="P12-F3-D2 matched reference evaluation",
    )
    try:
        if marker_path.exists():
            marker = json.loads(marker_path.read_text())
            if (
                marker.get("schema_version") != SCHEMA
                or not marker.get("pass")
                or marker.get("frozen_digest") != frozen_digest
                or marker.get("ph001_opened")
            ):
                raise RuntimeError("existing D2 matched-reference freeze changed")
            for key, row in marker.get("reports", {}).items():
                if (
                    key not in frozen_references
                    or sha256(Path(row["path"])) != row["sha256"]
                    or sha256(Path(row["visual_path"])) != row["visual_sha256"]
                    or sha256(Path(row["shear_path"])) != row["shear_sha256"]
                ):
                    raise RuntimeError("existing D2 matched-reference report changed")
            if set(marker.get("reports", {})) != set(frozen_references):
                raise RuntimeError("existing D2 matched-reference set is incomplete")
            print(json.dumps(marker, indent=2, sort_keys=True))
            return

        report_root = args.output_root / "matched_reference_reports"
        if report_root.exists() and any(report_root.iterdir()):
            raise RuntimeError("partial D2 matched-reference reports lack a terminal freeze")
        report_root.mkdir(parents=True, exist_ok=True)
        reports = {}
        for key in config["evaluation"]["matched_reference_methods"]:
            source = frozen_references[key]
            archive, records = _records(Path(source["archive"]), panel, expected_ids)
            report = evaluate_records(
                records,
                method=str(archive["method"]),
                seed=seed,
                device=args.device,
            )
            label_authoritative_core_bootstrap(
                report["tarp"]["ordered_eigenvalues"],
                report["tarp"]["eigengaps"],
            )
            report.update(
                {
                    "schema_version": "p12f3-d2-matched-reference-report-v1",
                    "reference_key": key,
                    "archive": source["archive"],
                    "archive_sha256": source["archive_sha256"],
                    "common_evaluator_seed": seed,
                    "core_ids": expected_ids,
                    "truth_files_read": ["ph006"],
                    "ph001_opened": False,
                }
            )
            report_path = report_root / f"{key}.json"
            atomic_json(report_path, report)
            visual, _ = analyze_archive(Path(source["archive"]), device=args.device)
            visual.update(
                {
                    "schema_version": "p12f3-d2-matched-reference-visual-v1",
                    "reference_key": key,
                    "archive": source["archive"],
                    "archive_sha256": source["archive_sha256"],
                    "truth_files_read": ["ph006"],
                    "ph001_opened": False,
                }
            )
            visual_path = report_root / f"{key}_visual.json"
            atomic_json(visual_path, visual)
            shear = audit_archive(
                Path(source["archive"]),
                device=args.device,
                draw_batch=8,
                maximum_k=0.1813799364234218,
            )
            label_authoritative_core_bootstrap(shear["joint_tarp_blocked"])
            shear["resampling_note"] = (
                "joint_tarp is pooled visualization; joint_tarp_blocked resamples "
                "authoritative ph006 patch cores"
            )
            shear.update(
                {
                    "schema_version": "p12f3-d2-matched-reference-shear-v1",
                    "reference_key": key,
                    "archive": source["archive"],
                    "archive_sha256": source["archive_sha256"],
                    "truth_files_read": ["ph006"],
                    "ph001_opened": False,
                }
            )
            shear_path = report_root / f"{key}_shear.json"
            atomic_json(shear_path, shear)
            reports[key] = {
                "path": str(report_path.resolve()),
                "sha256": sha256(report_path),
                "visual_path": str(visual_path.resolve()),
                "visual_sha256": sha256(visual_path),
                "shear_path": str(shear_path.resolve()),
                "shear_sha256": sha256(shear_path),
                "archive": source["archive"],
                "archive_sha256": source["archive_sha256"],
                "method": source["method"],
                "core_ids": expected_ids,
            }
        marker = {
            "schema_version": SCHEMA,
            "created_utc": utc_now(),
            "pass": True,
            "frozen": frozen,
            "frozen_digest": frozen_digest,
            "reports": reports,
            "truth_files_read": ["ph006"],
            "ph001_opened": False,
        }
        atomic_json(marker_path, marker)
        print(json.dumps(marker, indent=2, sort_keys=True))
    finally:
        lock.close()


if __name__ == "__main__":
    main()
