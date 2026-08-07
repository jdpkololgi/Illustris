import unittest

import numpy as np

from workflows.abacus_tweb.p8_build_multitracer_control_fields import RED_SHIFT_STRATA
from workflows.abacus_tweb.p8_evaluate_multitracer_controls import (
    stratum_scale,
    thin_response_factors,
)


class MultitracerControlEvaluationTests(unittest.TestCase):
    def test_stratum_scale_uses_registered_half_open_shells(self):
        factors = {name: index / 10 for index, (name, _, _) in enumerate(RED_SHIFT_STRATA, 1)}
        redshift = np.asarray([0.10, 0.149999, 0.15, 0.249999, 0.55, 0.599999, 0.60])
        scale = stratum_scale(redshift, factors)
        np.testing.assert_allclose(scale, [0.1, 0.1, 0.2, 0.2, 0.6, 0.6, 0.0])

    def test_none_scale_is_identity(self):
        np.testing.assert_array_equal(stratum_scale(np.asarray([0.0, 0.2, 1.0]), None), 1.0)

    def test_thin_response_factors_keep_tracers_separate(self):
        shell_rows = {}
        for index, (name, _, _) in enumerate(RED_SHIFT_STRATA):
            shell_rows[name] = {
                "retention_fraction_by_tracer": {
                    "bright": 0.10 + index / 100,
                    "faint": 0.20 + index / 100,
                }
            }
        manifest = {
            "density_matched_thinning": {
                "17": {"audit": {"NGC": shell_rows}}
            }
        }
        factors = thin_response_factors(manifest, 17, "NGC")
        first = RED_SHIFT_STRATA[0][0]
        last = RED_SHIFT_STRATA[-1][0]
        self.assertAlmostEqual(factors["bright"][first], 0.10)
        self.assertAlmostEqual(factors["faint"][first], 0.20)
        self.assertAlmostEqual(factors["bright"][last], 0.15)
        self.assertAlmostEqual(factors["faint"][last], 0.25)


if __name__ == "__main__":
    unittest.main()
