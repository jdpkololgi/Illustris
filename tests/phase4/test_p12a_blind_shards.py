from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from workflows.sbi.p12a_blind_shards import core_safe_shards
from workflows.sbi.p12a_blind_array_worker import validate_existing_shard
from workflows.abacus_tweb.p8_deterministic_common import sha256


class P12ABlindShardTest(unittest.TestCase):
    def test_four_shards_are_complete_and_do_not_split_cores(self):
        core = np.repeat(np.arange(11), np.arange(1, 12))
        shards = core_safe_shards(core, 4)
        self.assertEqual(len(shards), 4)
        self.assertEqual(shards[0]["start"], 0)
        self.assertEqual(shards[-1]["stop"], len(core))
        self.assertEqual(sum(item["rows"] for item in shards), len(core))
        for left, right in zip(shards[:-1], shards[1:]):
            self.assertEqual(left["stop"], right["start"])
            self.assertNotEqual(core[left["stop"] - 1], core[right["start"]])

    def test_invalid_core_order_or_excess_shards_fails(self):
        with self.assertRaises(ValueError):
            core_safe_shards(np.asarray([0, 2, 1]), 2)
        with self.assertRaises(ValueError):
            core_safe_shards(np.asarray([0, 0, 1, 1]), 3)

    def test_existing_blind_shard_reuse_is_content_addressed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {
                name: root / name
                for name in ("output.npz", "audit.npz", "checkpoint.pt", "candidate.json", "quality.json")
            }
            for path in paths.values():
                path.write_bytes(path.name.encode())
            marker = root / "output.json"
            payload = {
                "schema_version": "p12a-blind-posterior-shard-v1",
                "pass": True,
                "start": 0,
                "stop": 10,
                "draws": 512,
                "seed": 42,
                "context_sha256": "context",
                "checkpoint_sha256": sha256(paths["checkpoint.pt"]),
                "candidate_sha256": sha256(paths["candidate.json"]),
                "quality_thresholds_sha256": sha256(paths["quality.json"]),
                "summary_sha256": sha256(paths["output.npz"]),
                "audit_draws": str(paths["audit.npz"]),
                "audit_draws_sha256": sha256(paths["audit.npz"]),
                "truth_files_read": [],
                "open_count": 0,
            }
            marker.write_text(json.dumps(payload))
            found = validate_existing_shard(
                output=paths["output.npz"], marker=marker,
                shard={"start": 0, "stop": 10}, draws=512, seed=42,
                context_sha256="context", checkpoint=paths["checkpoint.pt"],
                candidate=paths["candidate.json"], quality_thresholds=paths["quality.json"],
            )
            self.assertTrue(found["pass"])
            paths["audit.npz"].write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                validate_existing_shard(
                    output=paths["output.npz"], marker=marker,
                    shard={"start": 0, "stop": 10}, draws=512, seed=42,
                    context_sha256="context", checkpoint=paths["checkpoint.pt"],
                    candidate=paths["candidate.json"], quality_thresholds=paths["quality.json"],
                )


if __name__ == "__main__":
    unittest.main()
