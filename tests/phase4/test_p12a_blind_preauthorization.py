from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from workflows.sbi.p12_production_contract import posterior_summaries, quality_bitmask
from workflows.sbi.p12a_blind_preauthorization import validate_frozen_audit_export


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": _sha(path), "bytes": path.stat().st_size}


class P12ABlindPreauthorizationTest(unittest.TestCase):
    def _fixture(
        self,
        root: Path,
        *,
        wrong_audit_membership: bool = False,
        wrong_audit_summary: bool = False,
    ):
        rows, draws = 8, 8
        parent = np.arange(rows, dtype=np.int64)
        core = np.repeat(np.arange(4, dtype=np.int64), 2)
        base = np.column_stack(
            (
                np.linspace(-0.2, 0.0, rows),
                np.linspace(0.1, 0.3, rows),
                np.linspace(0.4, 0.6, rows),
            )
        ).astype(np.float32)
        redshift = np.linspace(0.1, 0.5, rows, dtype=np.float32)
        ntilde = np.linspace(1e-4, 8e-4, rows, dtype=np.float32)
        cap = (parent % 2).astype(np.uint8)
        shell = (parent % 4).astype(np.int8)
        boundary = np.linspace(1.0, 20.0, rows, dtype=np.float32)
        conditioning = np.column_stack(
            (base, redshift, np.log(ntilde), cap.astype(np.float32), np.log1p(boundary))
        ).astype(np.float32)
        audit_selected = np.asarray([True, False, True, False, True, False, True, False])
        p1_index = root / "canonical_index.npz"
        np.savez(
            p1_index,
            parent_node_id=parent,
            targetid=parent + 1,
            cap=cap,
            shell=shell,
            active=np.ones(rows, dtype=bool),
            context=np.ones(rows, dtype=bool),
            valid_target=np.ones(rows, dtype=bool),
        )
        context_path = root / "ph001_context.npz"
        np.savez(
            context_path,
            parent_node_id=parent,
            core_id=core,
            base_prediction_eigenvalues=base,
            redshift=redshift,
            ntilde_mpc3=ntilde,
            cap=cap,
            shell=shell,
            distance_to_support_boundary_mpc=boundary,
            support_random=np.ones(rows, dtype=bool),
            audit_selected=audit_selected,
            context=conditioning,
        )
        context_marker_path = root / "context.json"
        context_marker_path.write_text(
            json.dumps(
                {
                    "schema_version": "p12a-blind-base-context-v1",
                    "phase": "ph001",
                    "rows": rows,
                    "array": str(context_path.resolve()),
                    "array_sha256": _sha(context_path),
                    "truth_files_read": [],
                    "open_count": 0,
                    "sealed_phase_opened": False,
                    "pass": True,
                }
            )
        )
        checkpoint = root / "fmpe.pt"
        checkpoint.write_bytes(b"checkpoint bytes")
        quality_contract = {
            "schema_version": "p12a-production-quality-thresholds-v1",
            "response_covariate": {
                "name": "log_ntilde_mpc3",
                "context_index": 4,
                "training_minimum": -20.0,
                "training_maximum": 0.0,
            },
            "prior_dominated_width": {
                "threshold_by_ordered_eigenvalue": [10.0, 10.0, 10.0]
            },
            "boundary_distance": {
                "threshold_r_mpc": 0.1,
                "threshold_2r_mpc": 0.2,
            },
        }
        quality_path = root / "quality.json"
        quality_path.write_text(json.dumps(quality_contract))
        candidate_path = root / "candidate.json"
        candidate = {
            "posterior_draws": draws,
            "audit_draw_rows": int(audit_selected.sum()),
            "artifacts": {
                "checkpoint": _record(checkpoint),
                "quality_thresholds": _record(quality_path),
            },
        }
        candidate_path.write_text(json.dumps(candidate))
        candidate_record = _record(candidate_path)
        plan_path = root / "plan.json"
        shards = [
            {"shard": 0, "start": 0, "stop": 4, "rows": 4},
            {"shard": 1, "start": 4, "stop": 8, "rows": 4},
        ]
        plan_path.write_text(
            json.dumps(
                {
                    "schema_version": "p12a-blind-core-shard-plan-v1",
                    "phase": "ph001",
                    "context": str(context_path.resolve()),
                    "context_sha256": _sha(context_path),
                    "rows": rows,
                    "shard_count": 2,
                    "shards": shards,
                    "truth_files_read": [],
                    "open_count": 0,
                    "sealed_phase_opened": False,
                    "pass": True,
                }
            )
        )
        shard_records = []
        for shard in shards:
            start, stop = shard["start"], shard["stop"]
            # Deterministic, finite, ordered mock posterior draws.
            draw = np.repeat(base[start:stop, None, :], draws, axis=1)
            draw += np.linspace(-0.02, 0.02, draws, dtype=np.float32)[None, :, None]
            summaries = posterior_summaries(draw)
            if wrong_audit_summary and shard["shard"] == 0:
                summaries["eigenvalue_mean"] = summaries["eigenvalue_mean"].copy()
                summaries["eigenvalue_mean"][0, 0] += np.float32(0.01)
            quality = quality_bitmask(
                redshift=redshift[start:stop],
                boundary_distance_mpc_h=boundary[start:stop],
                response_covariate=conditioning[start:stop, 4],
                posterior_width=(
                    summaries["eigenvalue_q84"] - summaries["eigenvalue_q16"]
                ),
                response_training_range=(-20.0, 0.0),
                prior_width_threshold=np.asarray([10.0, 10.0, 10.0]),
                boundary_r_mpc=0.1,
                boundary_2r_mpc=0.2,
            )
            summary_path = root / f"shard_{shard['shard']:03d}.npz"
            np.savez_compressed(
                summary_path,
                parent_node_id=parent[start:stop],
                core_id=core[start:stop],
                base_prediction_eigenvalues=base[start:stop],
                redshift=redshift[start:stop],
                ntilde_mpc3=ntilde[start:stop],
                cap=cap[start:stop],
                shell=shell[start:stop],
                distance_to_support_boundary_mpc=boundary[start:stop],
                support_random=np.ones(stop - start, dtype=bool),
                quality_bitmask=quality,
                **summaries,
            )
            selected = audit_selected[start:stop]
            audit_parent = parent[start:stop][selected]
            if wrong_audit_membership and shard["shard"] == 0:
                audit_parent = audit_parent[::-1]
            audit_path = root / f"shard_{shard['shard']:03d}_audit_draws.npz"
            np.savez_compressed(
                audit_path,
                parent_node_id=audit_parent,
                eigenvalue_draws=draw[selected],
            )
            marker_path = root / f"shard_{shard['shard']:03d}.json"
            marker_path.write_text(
                json.dumps(
                    {
                        "schema_version": "p12a-blind-posterior-shard-v1",
                        "phase": "ph001",
                        **shard,
                        "draws": draws,
                        "seed": 42 + shard["shard"],
                        "candidate_sha256": candidate_record["sha256"],
                        "checkpoint": str(checkpoint.resolve()),
                        "checkpoint_sha256": _sha(checkpoint),
                        "context_sha256": _sha(context_path),
                        "summary": str(summary_path.resolve()),
                        "summary_sha256": _sha(summary_path),
                        "audit_draws": str(audit_path.resolve()),
                        "audit_draws_sha256": _sha(audit_path),
                        "truth_files_read": [],
                        "open_count": 0,
                        "sealed_phase_opened": False,
                        "pass": True,
                    }
                )
            )
            shard_records.append(_record(marker_path))
        complete_path = root / "complete.json"
        complete_path.write_text(
            json.dumps(
                {
                    "schema_version": "p12a-blind-export-complete-v1",
                    "phase": "ph001",
                    "plan": _record(plan_path),
                    "context": _record(context_path),
                    "rows": rows,
                    "shards": shard_records,
                    "truth_files_read": [],
                    "open_count": 0,
                    "sealed_phase_opened": False,
                    "pass": True,
                }
            )
        )
        frozen = {
            "p12a_candidate": candidate_record,
            "prediction_manifests": [
                _record(context_marker_path),
                _record(complete_path),
            ],
        }
        return frozen, candidate, p1_index

    def test_deep_export_replay_validates_every_summary_and_audit_draw(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "workflows.sbi.p12a_blind_preauthorization.PRODUCTION_POSTERIOR_DRAWS", 8
        ), mock.patch(
            "workflows.sbi.p12a_blind_preauthorization.PRODUCTION_AUDIT_ROWS", 4
        ):
            frozen, candidate, p1_index = self._fixture(Path(temporary))
            with mock.patch(
                "workflows.sbi.p12a_blind_preauthorization.P1_CANONICAL_INDEX",
                p1_index,
            ):
                report = validate_frozen_audit_export(frozen, candidate)
            self.assertEqual(report["summary_rows"], 8)
            self.assertEqual(report["audit_rows"], 4)
            self.assertEqual(report["posterior_draws"], 8)

    def test_deep_export_replay_rejects_audit_membership_drift(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "workflows.sbi.p12a_blind_preauthorization.PRODUCTION_POSTERIOR_DRAWS", 8
        ), mock.patch(
            "workflows.sbi.p12a_blind_preauthorization.PRODUCTION_AUDIT_ROWS", 4
        ):
            frozen, candidate, p1_index = self._fixture(
                Path(temporary), wrong_audit_membership=True
            )
            with mock.patch(
                "workflows.sbi.p12a_blind_preauthorization.P1_CANONICAL_INDEX",
                p1_index,
            ), self.assertRaises(RuntimeError):
                validate_frozen_audit_export(frozen, candidate)

    def test_deep_export_replay_rejects_summary_draw_disagreement(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "workflows.sbi.p12a_blind_preauthorization.PRODUCTION_POSTERIOR_DRAWS", 8
        ), mock.patch(
            "workflows.sbi.p12a_blind_preauthorization.PRODUCTION_AUDIT_ROWS", 4
        ):
            frozen, candidate, p1_index = self._fixture(
                Path(temporary), wrong_audit_summary=True
            )
            with mock.patch(
                "workflows.sbi.p12a_blind_preauthorization.P1_CANONICAL_INDEX",
                p1_index,
            ), self.assertRaises(RuntimeError):
                validate_frozen_audit_export(frozen, candidate)


if __name__ == "__main__":
    unittest.main()
