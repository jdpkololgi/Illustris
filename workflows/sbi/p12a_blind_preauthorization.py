#!/usr/bin/env python3
"""Deep, truth-free validation of the frozen P12-A ph001 posterior export.

This module is intentionally outside the live export/freeze path.  It is called
only while constructing the one-open authorization marker, after the export is
complete.  That separation lets a queued export continue under its already
frozen output contract while authorization still verifies every production
summary row and every retained 512-draw audit row.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from workflows.abacus_tweb.p8_deterministic_common import sha256
from workflows.sbi.p12_production_contract import (
    QUALITY_BITS,
    assert_truth_free_payload,
    posterior_summaries,
    quality_bitmask,
)


CONTEXT_SCHEMA = "p12a-blind-base-context-v1"
COMPLETE_SCHEMA = "p12a-blind-export-complete-v1"
SHARD_SCHEMA = "p12a-blind-posterior-shard-v1"
PLAN_SCHEMA = "p12a-blind-core-shard-plan-v1"
PRODUCTION_SHARD_SEED_BASE = 42
PRODUCTION_POSTERIOR_DRAWS = 512
PRODUCTION_AUDIT_ROWS = 50_000
P1_CANONICAL_INDEX = Path(
    "/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph001/"
    "p1_canonical/canonical_index.npz"
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _verified_path(record: Mapping[str, Any], label: str) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    if (
        not path.is_file()
        or record.get("sha256") != sha256(path)
        or ("bytes" in record and int(record["bytes"]) != path.stat().st_size)
    ):
        raise RuntimeError(f"stale or incomplete {label}: {path}")
    return path


def _manifest_by_schema(frozen: Mapping[str, Any]) -> dict[str, tuple[Path, dict]]:
    result: dict[str, tuple[Path, dict]] = {}
    for record in frozen.get("prediction_manifests", []):
        path = _verified_path(record, "frozen prediction manifest")
        payload = _load(path)
        assert_truth_free_payload(payload)
        schema = str(payload.get("schema_version", ""))
        if schema in result:
            raise RuntimeError(f"duplicate frozen prediction schema: {schema}")
        result[schema] = (path, payload)
    return result


def _validate_summary(
    path: Path,
    *,
    rows: int,
    expected_parent: np.ndarray,
    expected_core: np.ndarray,
    expected_context: Mapping[str, np.ndarray],
    expected_response_covariate: np.ndarray,
    quality_contract: Mapping[str, Any],
    audit_mask: np.ndarray,
    audit_draws: np.ndarray,
) -> None:
    with np.load(path, mmap_mode="r") as summary:
        required = {
            "parent_node_id",
            "core_id",
            "base_prediction_eigenvalues",
            "eigenvalue_mean",
            "eigenvalue_std",
            "eigenvalue_q05",
            "eigenvalue_q16",
            "eigenvalue_q50",
            "eigenvalue_q84",
            "eigenvalue_q95",
            "probability_eigenvalue_gt_0p2",
            "web_class_probability",
            "web_class_entropy_nats",
            "trace_mean",
            "trace_std",
            "trace_q05",
            "trace_q16",
            "trace_q50",
            "trace_q84",
            "trace_q95",
            "redshift",
            "ntilde_mpc3",
            "cap",
            "shell",
            "support_random",
            "distance_to_support_boundary_mpc",
            "quality_bitmask",
        }
        if not required.issubset(summary.files):
            raise RuntimeError(
                f"posterior summary lacks arrays: {sorted(required - set(summary.files))}"
            )
        if not np.array_equal(
            np.asarray(summary["parent_node_id"], dtype=np.int64), expected_parent
        ) or not np.array_equal(
            np.asarray(summary["core_id"], dtype=np.int64), expected_core
        ):
            raise RuntimeError("posterior summary does not preserve exact context ordering")
        for name, expected in expected_context.items():
            if not np.array_equal(np.asarray(summary[name]), np.asarray(expected)):
                raise RuntimeError(
                    f"posterior summary does not preserve frozen context field: {name}"
                )
        eigen_arrays = (
            "base_prediction_eigenvalues",
            "eigenvalue_mean",
            "eigenvalue_q05",
            "eigenvalue_q16",
            "eigenvalue_q50",
            "eigenvalue_q84",
            "eigenvalue_q95",
        )
        values: dict[str, np.ndarray] = {}
        for name in eigen_arrays:
            value = np.asarray(summary[name], dtype=np.float64)
            if (
                value.shape != (rows, 3)
                or not np.all(np.isfinite(value))
                or np.any(np.diff(value, axis=1) < -64.0 * np.finfo(np.float32).eps)
            ):
                raise RuntimeError(f"posterior summary array is invalid or unordered: {name}")
            values[name] = value
        standard_deviation = np.asarray(summary["eigenvalue_std"], dtype=np.float64)
        if (
            standard_deviation.shape != (rows, 3)
            or not np.all(np.isfinite(standard_deviation))
            or np.any(standard_deviation < 0)
        ):
            raise RuntimeError("posterior standard deviations are invalid")
        above = np.asarray(
            summary["probability_eigenvalue_gt_0p2"], dtype=np.float64
        )
        if (
            above.shape != (rows, 3)
            or not np.all(np.isfinite(above))
            or np.any((above < 0.0) | (above > 1.0))
        ):
            raise RuntimeError("posterior threshold probabilities are invalid")
        quantiles = np.stack(
            [
                values["eigenvalue_q05"],
                values["eigenvalue_q16"],
                values["eigenvalue_q50"],
                values["eigenvalue_q84"],
                values["eigenvalue_q95"],
            ],
            axis=1,
        )
        if np.any(np.diff(quantiles, axis=1) < -64.0 * np.finfo(np.float32).eps):
            raise RuntimeError("posterior summary quantiles are not monotone")
        probability = np.asarray(summary["web_class_probability"], dtype=np.float64)
        if (
            probability.shape != (rows, 4)
            or not np.all(np.isfinite(probability))
            or np.any(probability < 0)
            or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-6, rtol=0.0)
        ):
            raise RuntimeError("posterior web-class probabilities are invalid")
        entropy = np.asarray(summary["web_class_entropy_nats"], dtype=np.float64)
        if (
            entropy.shape != (rows,)
            or not np.all(np.isfinite(entropy))
            or np.any(entropy < 0.0)
            or np.any(entropy > np.log(4.0) + 1e-6)
        ):
            raise RuntimeError("posterior web-class entropy is invalid")
        trace_values: dict[str, np.ndarray] = {}
        for name in (
            "trace_mean",
            "trace_q05",
            "trace_q16",
            "trace_q50",
            "trace_q84",
            "trace_q95",
        ):
            value = np.asarray(summary[name], dtype=np.float64)
            if value.shape != (rows,) or not np.all(np.isfinite(value)):
                raise RuntimeError(f"posterior trace summary is invalid: {name}")
            trace_values[name] = value
        trace_standard_deviation = np.asarray(summary["trace_std"], dtype=np.float64)
        if (
            trace_standard_deviation.shape != (rows,)
            or not np.all(np.isfinite(trace_standard_deviation))
            or np.any(trace_standard_deviation < 0.0)
        ):
            raise RuntimeError("posterior trace standard deviation is invalid")
        trace_quantiles = np.column_stack(
            [
                trace_values["trace_q05"],
                trace_values["trace_q16"],
                trace_values["trace_q50"],
                trace_values["trace_q84"],
                trace_values["trace_q95"],
            ]
        )
        if np.any(np.diff(trace_quantiles, axis=1) < -64.0 * np.finfo(np.float32).eps):
            raise RuntimeError("posterior trace quantiles are not monotone")
        for name in ("redshift", "ntilde_mpc3", "distance_to_support_boundary_mpc"):
            value = np.asarray(summary[name], dtype=np.float64)
            if value.shape != (rows,) or not np.all(np.isfinite(value)):
                raise RuntimeError(f"posterior covariate is invalid: {name}")
        cap = np.asarray(summary["cap"])
        shell = np.asarray(summary["shell"])
        support = np.asarray(summary["support_random"], dtype=bool)
        if (
            cap.shape != (rows,)
            or shell.shape != (rows,)
            or support.shape != (rows,)
            or not np.all(np.isin(cap, (0, 1)))
            or not np.all(np.isin(shell, (0, 1, 2, 3)))
            or not np.all(support)
        ):
            raise RuntimeError("posterior cap/shell/support metadata are invalid")
        quality = np.asarray(summary["quality_bitmask"])
        if quality.shape != (rows,) or not np.issubdtype(quality.dtype, np.integer):
            raise RuntimeError("posterior quality bitmask is invalid")
        response = quality_contract.get("response_covariate", {})
        prior = quality_contract.get("prior_dominated_width", {})
        boundary = quality_contract.get("boundary_distance", {})
        if (
            response.get("name") != "log_ntilde_mpc3"
            or int(response.get("context_index", -1)) != 4
            or quality_contract.get("schema_version")
            != "p12a-production-quality-thresholds-v1"
        ):
            raise RuntimeError("frozen posterior quality-threshold contract changed")
        expected_quality = quality_bitmask(
            redshift=np.asarray(summary["redshift"], dtype=np.float64),
            boundary_distance_mpc_h=np.asarray(
                summary["distance_to_support_boundary_mpc"], dtype=np.float64
            ),
            response_covariate=np.asarray(
                expected_response_covariate, dtype=np.float64
            ),
            posterior_width=(
                np.asarray(summary["eigenvalue_q84"], dtype=np.float64)
                - np.asarray(summary["eigenvalue_q16"], dtype=np.float64)
            ),
            response_training_range=(
                float(response["training_minimum"]),
                float(response["training_maximum"]),
            ),
            prior_width_threshold=np.asarray(
                prior["threshold_by_ordered_eigenvalue"], dtype=np.float64
            ),
            boundary_r_mpc=float(boundary["threshold_r_mpc"]),
            boundary_2r_mpc=float(boundary["threshold_2r_mpc"]),
        )
        if not np.array_equal(quality.astype(np.uint16), expected_quality):
            raise RuntimeError("posterior quality bitmask fails exact frozen recomputation")
        known_bits = np.uint16(0)
        for bit in QUALITY_BITS.values():
            known_bits |= np.uint16(bit)
        if np.any(quality.astype(np.uint16) & np.uint16(~known_bits)):
            raise RuntimeError("posterior quality bitmask contains an unknown bit")

        # The retained production draws are the only rows on which every summary
        # can be recomputed without re-running FMPE.  Exact equality here catches
        # exporter mistakes (wrong axis, quantile, threshold, class ordering, or
        # trace) that ordinary shape/finiteness checks would silently accept.
        audit_mask = np.asarray(audit_mask, dtype=bool)
        if audit_mask.shape != (rows,) or audit_draws.shape != (
            int(audit_mask.sum()),
            PRODUCTION_POSTERIOR_DRAWS,
            3,
        ):
            raise RuntimeError("posterior audit mask/draw shape is invalid")
        recomputed = posterior_summaries(audit_draws)
        for name, expected in recomputed.items():
            observed = np.asarray(summary[name])[audit_mask]
            if not np.array_equal(observed, expected):
                raise RuntimeError(
                    f"posterior summary differs from retained production draws: {name}"
                )


def validate_frozen_audit_export(
    frozen: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, int]:
    """Validate all summaries and the exact frozen audit subset without truth."""

    manifests = _manifest_by_schema(frozen)
    if CONTEXT_SCHEMA not in manifests or COMPLETE_SCHEMA not in manifests:
        raise RuntimeError("frozen prediction inventory lacks context/export completion")
    _, context_marker = manifests[CONTEXT_SCHEMA]
    _, complete = manifests[COMPLETE_SCHEMA]
    context_path = Path(str(context_marker.get("array", ""))).resolve()
    if (
        not context_path.is_file()
        or context_marker.get("array_sha256") != sha256(context_path)
    ):
        raise RuntimeError("blind context artifact changed after export")
    plan_path = _verified_path(complete.get("plan", {}), "blind shard plan")
    plan = _load(plan_path)
    assert_truth_free_payload(plan)
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("pass") is not True:
        raise RuntimeError("blind shard plan is not frozen")
    if (
        Path(str(plan.get("context", ""))).resolve() != context_path
        or plan.get("context_sha256") != context_marker.get("array_sha256")
        or complete.get("context", {}).get("sha256") != plan.get("context_sha256")
    ):
        raise RuntimeError("context, plan, and completion identities disagree")

    posterior_draws = int(candidate.get("posterior_draws", -1))
    audit_draw_rows = int(candidate.get("audit_draw_rows", -1))
    if (
        posterior_draws != PRODUCTION_POSTERIOR_DRAWS
        or audit_draw_rows != PRODUCTION_AUDIT_ROWS
    ):
        raise RuntimeError("candidate posterior/audit draw contract changed")
    candidate_record = frozen.get("p12a_candidate", {})
    candidate_path = _verified_path(candidate_record, "P12-A candidate")
    candidate_hash = sha256(candidate_path)
    checkpoint_record = candidate.get("artifacts", {}).get("checkpoint", {})
    checkpoint_path = _verified_path(checkpoint_record, "P12-A checkpoint")
    quality_path = _verified_path(
        candidate.get("artifacts", {}).get("quality_thresholds", {}),
        "P12-A quality thresholds",
    )
    quality_contract = _load(quality_path)

    evidence = complete.get("shards", [])
    planned = plan.get("shards", [])
    if len(evidence) != int(plan.get("shard_count", -1)) or len(evidence) != len(planned):
        raise RuntimeError("complete export does not contain every planned shard")
    cursor = 0
    for shard_index, expected in enumerate(planned):
        if (
            int(expected.get("shard", -1)) != shard_index
            or int(expected.get("start", -1)) != cursor
            or int(expected.get("rows", -1)) <= 0
            or int(expected.get("stop", -1))
            != int(expected.get("start", -1)) + int(expected.get("rows", -1))
        ):
            raise RuntimeError("blind shard plan is not an exact ordered partition")
        cursor = int(expected["stop"])
    if cursor != int(plan.get("rows", -1)):
        raise RuntimeError("blind shard plan does not cover every context row")

    with np.load(context_path, mmap_mode="r") as context:
        required = {
            "parent_node_id",
            "core_id",
            "support_random",
            "audit_selected",
            "context",
        }
        if not required.issubset(context.files):
            raise RuntimeError("blind context lacks frozen audit identities")
        parent = np.asarray(context["parent_node_id"], dtype=np.int64)
        core = np.asarray(context["core_id"], dtype=np.int64)
        support = np.asarray(context["support_random"], dtype=bool)
        audit_selected = np.asarray(context["audit_selected"], dtype=bool)
        conditioning = np.asarray(context["context"], dtype=np.float32)
        if (
            len(parent) != int(context_marker.get("rows", -1))
            or len(parent) != int(plan.get("rows", -1))
            or len(parent) != int(complete.get("rows", -1))
            or len(np.unique(parent)) != len(parent)
            or np.any(core < 0)
            or np.any(np.diff(core) < 0)
            or not np.all(support)
            or audit_selected.shape != parent.shape
            or int(audit_selected.sum()) != audit_draw_rows
            or conditioning.shape != (len(parent), 7)
            or not np.all(np.isfinite(conditioning))
        ):
            raise RuntimeError("blind context row/support/audit identity is invalid")

        # Bind the exported order and observed cap/shell metadata back to the
        # immutable P1 identity table.  The context is a supported subset, so it
        # need not be parent-sorted, but every parent must be an active valid P1
        # row and its cap/shell values must agree exactly.
        with np.load(P1_CANONICAL_INDEX, mmap_mode="r") as canonical:
            if not {
                "parent_node_id",
                "cap",
                "shell",
                "active",
                "valid_target",
            }.issubset(canonical.files):
                raise RuntimeError("P1 canonical identity archive schema changed")
            canonical_parent = np.asarray(
                canonical["parent_node_id"], dtype=np.int64
            )
            if (
                not np.array_equal(
                    canonical_parent,
                    np.arange(len(canonical_parent), dtype=np.int64),
                )
                or np.any(parent < 0)
                or np.any(parent >= len(canonical_parent))
                or not np.all(np.asarray(canonical["active"][parent], dtype=bool))
                or not np.all(
                    np.asarray(canonical["valid_target"][parent], dtype=bool)
                )
                or not np.array_equal(
                    np.asarray(canonical["cap"][parent]),
                    np.asarray(context["cap"]),
                )
                or not np.array_equal(
                    np.asarray(canonical["shell"][parent]),
                    np.asarray(context["shell"]),
                )
            ):
                raise RuntimeError(
                    "blind context ordering/metadata is not nested in canonical P1"
                )

        observed_audit: list[np.ndarray] = []
        for expected, record in zip(planned, evidence):
            manifest_path = _verified_path(record, "posterior shard manifest")
            shard = _load(manifest_path)
            assert_truth_free_payload(shard)
            start, stop = int(expected["start"]), int(expected["stop"])
            rows = stop - start
            if (
                shard.get("schema_version") != SHARD_SCHEMA
                or shard.get("phase") != "ph001"
                or shard.get("pass") is not True
                or any(int(shard.get(key, -1)) != int(expected[key]) for key in ("start", "stop", "rows"))
                or int(shard.get("draws", -1)) != posterior_draws
                or int(shard.get("seed", -1))
                != PRODUCTION_SHARD_SEED_BASE + int(expected["shard"])
                or shard.get("candidate_sha256") != candidate_hash
                or shard.get("checkpoint_sha256") != checkpoint_record.get("sha256")
                or Path(str(shard.get("checkpoint", ""))).resolve() != checkpoint_path
                or shard.get("context_sha256") != plan.get("context_sha256")
            ):
                raise RuntimeError(f"posterior shard violates frozen plan: {manifest_path}")
            summary_path = Path(str(shard.get("summary", ""))).resolve()
            audit_path = Path(str(shard.get("audit_draws", ""))).resolve()
            if (
                not summary_path.is_file()
                or shard.get("summary_sha256") != sha256(summary_path)
                or not audit_path.is_file()
                or shard.get("audit_draws_sha256") != sha256(audit_path)
            ):
                raise RuntimeError("posterior shard artifacts changed after completion")
            expected_parent = parent[start:stop]
            expected_core = core[start:stop]
            with np.load(audit_path, mmap_mode="r") as audit:
                if not {"parent_node_id", "eigenvalue_draws"}.issubset(audit.files):
                    raise RuntimeError("posterior audit archive schema mismatch")
                audit_parent = np.asarray(audit["parent_node_id"], dtype=np.int64)
                expected_audit_parent = expected_parent[audit_selected[start:stop]]
                draws = np.asarray(audit["eigenvalue_draws"], dtype=np.float32)
                if not np.array_equal(audit_parent, expected_audit_parent):
                    raise RuntimeError("posterior audit rows differ from frozen subset")
                if (
                    draws.shape != (len(audit_parent), posterior_draws, 3)
                    or not np.all(np.isfinite(draws))
                    or np.any(
                        np.diff(draws, axis=2)
                        < -64.0 * np.finfo(np.float32).eps
                    )
                ):
                    raise RuntimeError("posterior audit draws are invalid or unordered")
                observed_audit.append(audit_parent)
            _validate_summary(
                summary_path,
                rows=rows,
                expected_parent=expected_parent,
                expected_core=expected_core,
                expected_context={
                    name: np.asarray(context[name][start:stop])
                    for name in (
                        "base_prediction_eigenvalues",
                        "redshift",
                        "ntilde_mpc3",
                        "cap",
                        "shell",
                        "distance_to_support_boundary_mpc",
                    )
                }
                | {"support_random": np.ones(rows, dtype=bool)},
                expected_response_covariate=conditioning[start:stop, 4],
                quality_contract=quality_contract,
                audit_mask=audit_selected[start:stop],
                audit_draws=draws,
            )

        joined_audit = np.concatenate(observed_audit)
        if not np.array_equal(joined_audit, parent[audit_selected]):
            raise RuntimeError("posterior export does not exactly cover frozen audit rows")
    return {
        "summary_rows": int(complete["rows"]),
        "audit_rows": int(audit_draw_rows),
        "posterior_draws": int(posterior_draws),
        "shards": int(len(evidence)),
    }
