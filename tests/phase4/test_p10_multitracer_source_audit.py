import unittest

import numpy as np

from workflows.abacus_tweb.p10_multitracer_source_audit import (
    BRIGHT_BITS,
    FAINT_BITS,
    paths_for_phase,
    resolve_observation_path,
    target_counts,
)


class P10MultitracerSourceAuditTests(unittest.TestCase):
    def setUp(self):
        self.registry = {
            "path_templates": {
                "forfa": "/data/forFA{mock}_nomask.fits",
                "fiberassign": "/data/altmtl{mock}/fba{mock}/assigned.fits",
            },
            "phases": {
                "ph000": {
                    "mock": 0,
                    "observation_path_overrides": {
                        "fiberassign": "/data/altmtl0/fba0_bkp/assigned.fits"
                    },
                },
                "ph002": {"mock": 2},
            },
        }

    def test_phase_override_is_explicit_and_asset_specific(self):
        path, policy = resolve_observation_path(
            self.registry, "ph000", "fiberassign"
        )
        self.assertEqual(str(path), "/data/altmtl0/fba0_bkp/assigned.fits")
        self.assertEqual(policy, "phase_override")
        forfa, assigned, resolution = paths_for_phase(self.registry, "ph000")
        self.assertEqual(str(forfa), "/data/forFA0_nomask.fits")
        self.assertEqual(str(assigned), "/data/altmtl0/fba0_bkp/assigned.fits")
        self.assertEqual(
            resolution, {"forfa": "registry_template", "assigned": "phase_override"}
        )

    def test_standard_phase_uses_templates(self):
        _, assigned, resolution = paths_for_phase(self.registry, "ph002")
        self.assertEqual(str(assigned), "/data/altmtl2/fba2/assigned.fits")
        self.assertEqual(resolution["assigned"], "registry_template")

    def test_target_counts_or_bits_across_duplicate_assignments(self):
        targetid = np.asarray([4, 2, 4, 7, 2], dtype=np.int64)
        bits = np.asarray([0, BRIGHT_BITS, FAINT_BITS, FAINT_BITS, 0], dtype=np.int64)
        row = target_counts(targetid, bits)
        np.testing.assert_array_equal(row["targetid"], [2, 4, 7])
        np.testing.assert_array_equal(row["bright"], [True, False, False])
        np.testing.assert_array_equal(row["faint"], [False, True, True])
        self.assertEqual(row["duplicate_rows"], 2)


if __name__ == "__main__":
    unittest.main()
