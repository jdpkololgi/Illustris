import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from workflows.abacus_tweb import p3br_export_evidence as evidence


class TestP3brEvidence(unittest.TestCase):
    def test_blind_contamination_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "training_contract/P3BR_PIPELINE_COMPLETE.json"
            marker.parent.mkdir(parents=True)
            marker.write_text(json.dumps({
                "pass": True,
                "ph001_opened": True,
                "products": {phase: {} for phase in evidence.PHASES},
            }))
            with mock.patch("sys.argv", ["p3br_export_evidence.py", "--root", str(root)]):
                with self.assertRaisesRegex(RuntimeError, "opened ph001"):
                    evidence.main()


if __name__ == "__main__":
    unittest.main()
