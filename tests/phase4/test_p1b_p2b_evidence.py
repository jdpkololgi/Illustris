from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs" / "evidence" / "p1b_p2b"


class FullFootprintEvidenceTest(unittest.TestCase):
    def test_audit_and_manifests_are_consistent(self) -> None:
        audit = json.loads((EVIDENCE / "full_footprint_audit.json").read_text())
        p1 = json.loads((EVIDENCE / "p1b_manifest.json").read_text())
        p2 = json.loads((EVIDENCE / "p2b_union_manifest.json").read_text())

        self.assertTrue(audit["pass"])
        self.assertTrue(all(audit["gates"].values()))
        self.assertEqual(audit["counts"]["cross_cap_pairs"], 0)
        self.assertEqual(p1["catalogue_id"], "ph000_path1_full_ngc_sgc_v1")
        self.assertEqual(p1["counts"]["total"], audit["counts"]["parent_rows"])
        self.assertEqual(p1["counts"]["active"], audit["counts"]["active_rows"])
        self.assertEqual(
            p2["counts"]["context_nodes"],
            p1["counts"]["context"],
        )
        self.assertEqual(p2["assembly_contract"]["cross_cap_pairs"], 0)
        self.assertEqual(
            p2["counts"]["union_pairs_context"],
            p2["counts"]["delaunay_pairs_context"]
            + p2["counts"]["radius_only_pairs"],
        )
        self.assertGreater(p1["counts"]["active"], 5_000_000)
        self.assertGreater(
            p1["counts"]["by_shell"]["0.45_0.55"]["all"],
            70_000,
        )


if __name__ == "__main__":
    unittest.main()
