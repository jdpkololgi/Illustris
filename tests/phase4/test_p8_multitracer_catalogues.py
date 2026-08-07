from __future__ import annotations

import unittest

import numpy as np

from workflows.abacus_tweb.p8_build_multitracer_catalogues import (
    deterministic_uniform,
    faint_response_probability,
    sky_to_points,
    _make_faint_rows,
    FAINT_NORTH_BITS,
    FAINT_SOUTH_BITS,
    CATALOGUE_DTYPE,
)


class MultitracerCatalogueTests(unittest.TestCase):
    def test_targetid_random_is_order_independent(self):
        targetid = np.asarray([11, 4, 99, 123456], dtype=np.int64)
        reference = deterministic_uniform(targetid, 17)
        order = np.asarray([2, 0, 3, 1])
        scrambled = deterministic_uniform(targetid[order], 17)
        np.testing.assert_array_equal(scrambled, reference[order])
        self.assertTrue(np.all((reference >= 0.0) & (reference < 1.0)))

    def test_sky_points_are_mpc_with_binary_caps(self):
        points = sky_to_points(
            np.asarray([0.0, 180.0]),
            np.asarray([30.0, -30.0]),
            np.asarray([0.2, 0.3]),
        )
        self.assertEqual(points.shape, (2, 4))
        self.assertTrue(np.all(np.isfinite(points)))
        self.assertTrue(set(np.unique(points[:, 3])).issubset({0.0, 1.0}))
        self.assertGreater(np.linalg.norm(points[1, :3]), np.linalg.norm(points[0, :3]))

    def test_response_uses_target_bits_with_explicit_fallback(self):
        calibration = {
            "all": {"pass_probability": 0.8},
            "north": {"pass_probability": 0.9},
            "south": {"pass_probability": 0.7},
        }
        target = np.asarray([FAINT_NORTH_BITS, FAINT_SOUTH_BITS, 0], dtype=np.int64)
        probability, audit = faint_response_probability(target, calibration)
        np.testing.assert_allclose(probability, [0.9, 0.7, 0.8])
        self.assertEqual(audit["north_rows"], 1)
        self.assertEqual(audit["south_rows"], 1)
        self.assertEqual(audit["overall_fallback_rows"], 1)
        self.assertIn("never Galactic cap", audit["mapping"])

    def test_response_rejects_ambiguous_regional_bits(self):
        calibration = {
            "all": {"pass_probability": 0.8},
            "north": {"pass_probability": 0.9},
            "south": {"pass_probability": 0.7},
        }
        with self.assertRaisesRegex(RuntimeError, "both NORTH and SOUTH"):
            faint_response_probability(
                np.asarray([FAINT_NORTH_BITS | FAINT_SOUTH_BITS]), calibration
            )

    def test_prepared_faint_rows_can_be_filtered_for_proxy_repair(self):
        source = np.zeros(3, dtype=CATALOGUE_DTYPE)
        source["TARGETID"] = [1, 2, 3]
        source["TRACER_TYPE"] = 1
        source["ASSIGNED"] = 1
        source["SPEC_SUCCESS"] = 1
        source["CONTEXT"] = [1, 0, 1]
        source["BRIGHT_PARENT_ID"] = -1
        output = _make_faint_rows(source, np.asarray([True, False, True]))
        np.testing.assert_array_equal(output["TARGETID"], [1, 3])
        np.testing.assert_array_equal(output["CONTEXT"], [1, 1])
        np.testing.assert_array_equal(output["BRIGHT_PARENT_ID"], -1)

    def test_faint_rows_are_context_only(self):
        dtype = np.dtype(
            [
                ("TARGETID", "i8"), ("RA", "f8"), ("DEC", "f8"),
                ("Z_COSMO", "f8"), ("RSDZ", "f8"), ("R_MAG_APP", "f4"),
                ("FILE_NUM", "i4"), ("BOX_INDEX", "i4"),
                ("HALO_INDEX", "i8"), ("BGS_TARGET", "i8"),
                ("SOURCE_ROW", "i8"),
            ]
        )
        source = np.zeros(3, dtype=dtype)
        source["TARGETID"] = [1, 2, 3]
        source["RSDZ"] = [0.2, 0.59, 0.7]
        source["Z_COSMO"] = source["RSDZ"]
        source["FILE_NUM"] = 0
        source["BOX_INDEX"] = 0
        source["HALO_INDEX"] = [10, 20, 30]
        output = _make_faint_rows(source, np.asarray([True, True, True]))
        np.testing.assert_array_equal(output["BRIGHT_PARENT_ID"], -1)
        np.testing.assert_array_equal(output["TRACER_TYPE"], 1)
        # z=0.59 is the explicitly excluded sentinel region; z=0.7 is outside context.
        np.testing.assert_array_equal(output["CONTEXT"], [1, 0, 0])


if __name__ == "__main__":
    unittest.main()
