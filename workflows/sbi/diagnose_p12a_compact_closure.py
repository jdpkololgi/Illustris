#!/usr/bin/env python3
"""Read-only diagnosis of a failed compact-truth physical-closure check.

No posterior prediction, score, fit or calibration is evaluated. Frozen source
and authorization guards run before truth access. The diagnostic writes only
an exclusive aggregate receipt; it cannot create a compact truth stage marker.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def closure_diagnostic(eigenvalues, cweb):
    original = np.asarray(eigenvalues)
    eigen32 = original.astype(np.float32)
    eigen64 = eigen32.astype(np.float64)
    expected32 = np.sum(eigen32 > 0.2, axis=1).astype(np.uint8)
    expected64 = np.sum(eigen64 > 0.2, axis=1).astype(np.uint8)
    mismatch32 = expected32 != cweb
    mismatch64 = expected64 != cweb
    rounded_threshold = np.float32(0.2)
    affected = eigen32 == rounded_threshold
    return {
        "rows": int(len(original)), "source_dtype": str(original.dtype),
        "finite": bool(np.isfinite(eigen32).all()),
        "ordered": bool(np.all(np.diff(eigen32, axis=1) >= 0)),
        "source_values_preserved_by_float32": bool(np.array_equal(original, eigen32)),
        "float32_threshold_as_float64": float(rounded_threshold),
        "float32_comparison_class_mismatches": int(mismatch32.sum()),
        "float64_comparison_class_mismatches": int(mismatch64.sum()),
        "rows_at_rounded_threshold": int(affected.any(axis=1).sum()),
        "all_float32_mismatches_explained_by_threshold_rounding": bool(
            np.array_equal(mismatch32, (expected32 != expected64))
            and not mismatch64.any()
            and np.all(affected[mismatch32].any(axis=1))
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    from workflows.sbi import p12a_authorized_truth as frozen
    import fitsio

    frozen.authorization_context()
    annotation = frozen.validate_stage_marker(stage="annotation", truth_root=frozen.TRUTH_ROOT)
    context_marker, _, context_path = frozen._frozen_context()
    with np.load(context_path) as context, np.load(frozen.P1_INDEX) as canonical:
        parent = np.asarray(context["parent_node_id"], dtype=np.int64)
        canonical_parent = np.asarray(canonical["parent_node_id"], dtype=np.int64)
        targetid = np.asarray(canonical["targetid"], dtype=np.int64)[parent]
    if len(parent) != frozen.EXPECTED_CONTEXT_ROWS or len(parent) != len(np.unique(parent)):
        raise RuntimeError("Frozen context identity is invalid")
    if not np.array_equal(canonical_parent, np.arange(len(canonical_parent))):
        raise RuntimeError("Canonical parent identity is invalid")
    path = Path(annotation["artifacts"]["annotated_parent"]["path"])
    table = fitsio.read(str(path), columns=["TARGETID", "CWEB", "LAMBDA1", "LAMBDA2", "LAMBDA3"])
    row = targetid - 1
    if np.any(row < 0) or np.any(row >= len(table)) or not np.array_equal(table["TARGETID"][row], targetid):
        raise RuntimeError("Exact target identity join failed")
    eigen = np.column_stack([table[f"LAMBDA{i}"][row] for i in (1, 2, 3)])
    report = {
        "schema_version": "p12a-compact-closure-diagnostic-v1",
        "created_utc": frozen.utc_now(),
        "slurm_job_id": __import__("os").environ.get("SLURM_JOB_ID"),
        "authorization": frozen.record(frozen.AUTHORIZATION),
        "annotation_stage": frozen.record(frozen.stage_marker_path(frozen.TRUTH_ROOT, "annotation")),
        "diagnostic_implementation": frozen.record(Path(__file__)),
        "truth_files_read": [str(path)],
        "posterior_scores_computed": False,
        "predictions_modified": False,
        "truth_outputs_modified": False,
        "open_count": 1,
        "identity_join_exact": True,
        "closure": closure_diagnostic(eigen, table["CWEB"][row]),
    }
    frozen.write_or_validate_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
