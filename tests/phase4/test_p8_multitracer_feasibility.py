import unittest

import numpy as np

from workflows.abacus_tweb.p8_multitracer_feasibility import (
    classify_targets,
    feasibility_decision,
    shell_counts,
    summarize_chunk,
)


class P8MultitracerFeasibilityTests(unittest.TestCase):
    def test_target_masks_separate_bright_faint_and_hip(self):
        values = np.asarray([0, 1, 2, 3, 8, 9], dtype=np.int64)
        bright, faint = classify_targets(values, bright_mask=2, faint_mask=1 | 8)
        np.testing.assert_array_equal(bright, [False, False, True, True, False, False])
        np.testing.assert_array_equal(faint, [False, True, False, True, True, True])

    def test_shell_counts_use_registered_half_open_shells(self):
        z = np.asarray([0.149, 0.15, 0.249, 0.25, 0.549, 0.55])
        got = shell_counts(z, np.ones(z.size, dtype=bool))
        self.assertEqual(got, {
            "0p15_0p25": 2,
            "0p25_0p35": 1,
            "0p35_0p45": 0,
            "0p45_0p55": 1,
        })

    def test_chunk_summary_tracks_selection_spectra_and_truth(self):
        dtype = [
            ("BGS_TARGET", "i8"),
            ("R_MAG_APP", "f8"),
            ("Z", "f8"),
            ("ZWARN", "i8"),
            ("DELTACHI2", "f8"),
            ("SPECTYPE", "S16"),
            ("FILE_NUM", "i8"),
            ("BOX_INDEX", "i8"),
            ("HALO_INDEX", "i8"),
        ]
        tab = np.zeros(4, dtype=dtype)
        tab["BGS_TARGET"] = [2, 1, 9, 0]
        tab["R_MAG_APP"] = [19.0, 19.7, 20.0, 21.0]
        tab["Z"] = [0.2, 0.3, 0.5, 0.1]
        tab["DELTACHI2"] = [30, 30, 10, 30]
        tab["SPECTYPE"] = [b"GALAXY", b"GALAXY", b"GALAXY", b"STAR"]
        tab["FILE_NUM"] = [0, 0, 0, -1]
        tab["BOX_INDEX"] = [1, 2, 3, -1]
        tab["HALO_INDEX"] = [4, 5, 6, -1]
        got = summarize_chunk(tab, bright_mask=2, faint_mask=1 | 8)
        self.assertEqual(got["bright_rows"], 1)
        self.assertEqual(got["faint_rows"], 2)
        self.assertEqual(got["good_spectrum_rows"], {"all": 2, "bright": 1, "faint": 1})
        self.assertEqual(got["valid_truth_link_rows"], {"all": 3, "bright": 1, "faint": 2})

    def test_decision_is_conditional_even_when_faint_rows_survive(self):
        columns = ["TARGETID", "BGS_TARGET", "RA", "DEC", "Z"]
        stages = {
            "forfa_targets": {"summary": {"faint_rows": 100}},
            "fiberassign_input": {"summary": {"faint_rows": 100}},
            "fiberassign_assigned": {"summary": {"faint_rows": 70}, "columns": columns},
            "spectroscopic_join": {"summary": {"faint_rows": 50}, "columns": columns},
            "graphweb_bright_final": {"summary": {"faint_rows": 0}},
        }
        got = feasibility_decision(stages)
        self.assertEqual(got["verdict"], "CONDITIONAL_GO_BUILD_RESPONSE_COMPLETE_FAINT")
        self.assertFalse(got["multitracer_training_ready"])
        self.assertTrue(got["upstream_faint_exists"])

    def test_decision_rejects_absent_upstream_faint(self):
        stages = {
            "forfa_targets": {"summary": {"faint_rows": 0}},
            "fiberassign_input": {"summary": {"faint_rows": 0}},
        }
        got = feasibility_decision(stages)
        self.assertEqual(got["verdict"], "NO_GO_REGENERATE_UPSTREAM_TARGETS")


if __name__ == "__main__":
    unittest.main()
