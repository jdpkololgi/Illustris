import json
from pathlib import Path
import tempfile
import unittest

from workflows.abacus_tweb.p8_run_density_downstream import (
    context_complete,
    evaluation_complete,
    stitched_complete,
)


class DensityDownstreamLauncherTests(unittest.TestCase):
    def test_stitched_stage_requires_matching_checkpoint_and_full_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "D0_STITCHED_FIELD_READY").write_text("ready\n")
            (root / "stitched_field_manifest.json").write_text(json.dumps({
                "status": "PASS",
                "checkpoint_sha256": "abc",
                "support_coverage": {
                    "NGC": {"coverage_fraction": 1.0},
                    "SGC": {"coverage_fraction": 1.0},
                },
            }))
            self.assertTrue(stitched_complete(root, "abc"))
            with self.assertRaises(RuntimeError):
                stitched_complete(root, "different")

    def test_evaluation_and_context_require_matching_stitched_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root / "evaluation"
            context = root / "context"
            evaluation.mkdir()
            context.mkdir()
            (evaluation / "D0_FIELD_DOWNSTREAM_EVALUATED").write_text("ready\n")
            (evaluation / "field_downstream_metrics.json").write_text(json.dumps({
                "status": "PASS", "inputs": {"stitched_manifest_sha256": "abc"}
            }))
            (context / "D0_LEARNED_CONTEXT_COMPLETE").write_text("ready\n")
            (context / "learned_context_report.json").write_text(json.dumps({
                "status": "PASS", "inputs": {"stitched_manifest_sha256": "abc"}
            }))
            self.assertTrue(evaluation_complete(evaluation, "abc"))
            self.assertTrue(context_complete(context, "abc"))
            with self.assertRaises(RuntimeError):
                evaluation_complete(evaluation, "different")
            with self.assertRaises(RuntimeError):
                context_complete(context, "different")


if __name__ == "__main__":
    unittest.main()
