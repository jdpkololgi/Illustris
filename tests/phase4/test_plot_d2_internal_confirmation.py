import copy
import json
from pathlib import Path
import unittest

from workflows.sbi.plot_d2_internal_confirmation import chart_rows


class InternalConfirmationPlotTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[2]
        self.report = json.loads((root / "docs/evidence/p12/p12f3_d2_20260905/D2_INTERNAL_CONFIRMATION.json").read_text())

    def test_paired_contrasts_not_unpaired_panel_difference(self):
        rows = chart_rows(self.report)
        self.assertAlmostEqual(rows[0][2], 0.4399657844860656)
        self.assertAlmostEqual(rows[1][2], 0.4261745393558563)
        self.assertEqual([row[1] for row in rows], [128, 127])
        self.assertTrue(all(0 < row[3] < 0.04 for row in rows))

    def test_changed_frozen_inputs_refused(self):
        report = copy.deepcopy(self.report)
        report["frozen_inputs"]["policy"] = "changed"
        with self.assertRaises(ValueError):
            chart_rows(report)

    def test_failed_or_test_selected_report_refused(self):
        for key, value in (("pass", False), ("ph006_used_for_selection", True), ("ph001_opened", True)):
            report = copy.deepcopy(self.report)
            report[key] = value
            with self.assertRaises(ValueError):
                chart_rows(report)

    def test_changed_error_bar_convention_refused(self):
        report = copy.deepcopy(self.report)
        report["paired_contrasts"]["capacity"]["confirmation"]["selection"]["paired_interval_convention"] = "iid_voxel"
        with self.assertRaises(ValueError):
            chart_rows(report)


if __name__ == "__main__":
    unittest.main()
