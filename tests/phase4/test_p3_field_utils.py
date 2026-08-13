from __future__ import annotations

import json
import unittest
from pathlib import Path

import healpy as hp
import numpy as np

from workflows.abacus_tweb.p3a_build_canonical_fields import (
    GridSpec,
    cic_deposit,
    cosmology_lookup,
    field_block,
    fractional_index,
    log_count_ratio,
)
from workflows.abacus_tweb.p3a_catalogue_field_closure import host_consistency


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "docs" / "evidence" / "p3" / "p3_field_schema_v1.json"


class P3FieldUtilityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = json.loads(SCHEMA_PATH.read_text())

    def test_schema_uses_comoving_mpc_and_raw_channels(self) -> None:
        self.assertEqual(self.schema["coordinate_frame"]["units"], "Mpc")
        self.assertEqual(self.schema["grid"]["cell_mpc"], 5.0)
        self.assertAlmostEqual(self.schema["coordinate_frame"]["planck18_h"], 0.6766)
        self.assertAlmostEqual(self.schema["grid"]["cell_mpc_h"], 3.383)
        self.assertNotAlmostEqual(self.schema["grid"]["literal_5_mpc_h_cell_mpc"], 5.0)
        self.assertEqual(self.schema["normalization"]["stored_fields"],
                         "raw canonical channels only")
        self.assertFalse(self.schema["angular_support"]["target_columns_used"])
        self.assertFalse(self.schema["selection"]["split_ownership_used"])
        self.assertEqual(self.schema["completion_contract"]["required_unit_gate"],
                         "unit_audit.json pass=true")

    def test_cic_conserves_mass_and_fractional_coordinates(self) -> None:
        spec = GridSpec(origin=(0.0, 0.0, 0.0), shape=(8, 8, 8),
                        cell_mpc=5.0, padding_mpc=0.0)
        xyz = np.array([
            [12.5, 12.5, 12.5],
            [13.0, 17.0, 21.0],
            [28.0, 19.0, 23.0],
        ])
        counts, stats = cic_deposit(xyz, spec)
        self.assertAlmostEqual(float(counts.sum(dtype=np.float64)), len(xyz), places=6)
        self.assertAlmostEqual(stats["lost_weight"], 0.0, places=12)
        expected = (xyz - np.asarray(spec.origin)) / spec.cell_mpc - 0.5
        np.testing.assert_allclose(fractional_index(xyz, spec), expected)

    def test_log_ratio_is_finite_and_zero_outside_support(self) -> None:
        counts = np.array([0.0, 1.0, 3.0], dtype=np.float32)
        expected = np.array([0.2, 0.5, 1.0], dtype=np.float32)
        exposure = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        result = log_count_ratio(counts, expected, exposure, 1e-3, 1e-4)
        self.assertEqual(float(result[0]), 0.0)
        self.assertTrue(np.isfinite(result).all())

    def test_cweb_closure_uses_native_float32_threshold(self) -> None:
        dtype = np.dtype([
            ("FILE_NUM", "i4"), ("BOX_INDEX", "i4"), ("HALO_INDEX", "i8"),
            ("LAMBDA1", "f4"), ("LAMBDA2", "f4"), ("LAMBDA3", "f4"),
            ("CWEB", "i1"),
        ])
        table = np.zeros(1, dtype=dtype)
        table["LAMBDA1"] = np.float32(0.2)
        table["LAMBDA2"] = np.float32(0.2)
        table["LAMBDA3"] = np.float32(0.2)
        table["CWEB"] = 0
        report = host_consistency(table, np.ones(1, dtype=bool))
        self.assertEqual(report["cweb_mismatch_rows"], 0)

    def test_apodization_is_chunk_halo_stable(self) -> None:
        spec = GridSpec(origin=(600.0, -40.0, -40.0), shape=(16, 16, 16),
                        cell_mpc=5.0, padding_mpc=0.0)
        counts = np.zeros(spec.shape, dtype=np.float32)
        angular = np.ones(hp.nside2npix(self.schema["angular_support"]["nside"]),
                          dtype=bool)
        spline = {
            "grid_z": [0.0, 0.75],
            "ntilde": [1e-4, 1e-4],
            "ntilde_floor": 1e-6,
        }
        z_grid, r_grid = cosmology_lookup()
        slices = (slice(0, 8), slice(0, 8), slice(0, 8))
        base = field_block(
            spec, slices, counts, angular,
            self.schema["angular_support"]["nside"],
            z_grid, r_grid, spline, self.schema,
        )
        sigma_vox = self.schema["apodization"]["sigma_mpc"] / spec.cell_mpc
        halo = int(np.ceil(
            sigma_vox * self.schema["apodization"]["truncate_sigma"]
        ))
        wider = field_block(
            spec, slices, counts, angular,
            self.schema["angular_support"]["nside"],
            z_grid, r_grid, spline, self.schema,
            halo_override=halo + 2,
        )
        np.testing.assert_allclose(
            base["exposure_apodized"], wider["exposure_apodized"], atol=1e-7
        )
        for key, value in base.items():
            if not key.startswith("_"):
                self.assertTrue(np.isfinite(value).all(), key)


if __name__ == "__main__":
    unittest.main()
