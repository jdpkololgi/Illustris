import tempfile
import unittest
from pathlib import Path

from workflows.abacus_tweb.p10_build_density_field import (
    DensityBuildError,
    default_output,
    inspect_particle_inputs,
    resolve_particle_roots,
)
from workflows.abacus_tweb.p10_phase_assets import DEFAULT_REGISTRY, load_registry


class P10DensityBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load_registry(DEFAULT_REGISTRY)

    @staticmethod
    def _make_directory(root: Path, name: str, prefix: str) -> None:
        directory = root / name
        directory.mkdir(parents=True)
        for index in range(34):
            (directory / f"{prefix}_{index:03d}.asdf").write_bytes(
                f"{name}-{index}".encode()
            )
        (directory / "checksums.crc32").write_text("unit-test-placeholder\n")

    def test_default_output_is_phase_and_contract_explicit(self):
        output = default_output(self.registry, "ph002", 2048)
        self.assertIn("/ph002/targets/density/", str(output))
        self.assertTrue(
            output.name.endswith("ph002_z0.200_ngrid2048_ab10_tsc_counts.npy")
        )

    def test_online_phase_resolves_one_root_for_a_and_b(self):
        a_root, b_root, marker = resolve_particle_roots(
            DEFAULT_REGISTRY,
            self.registry,
            "ph006",
            Path("/unused"),
        )
        self.assertEqual(a_root, b_root)
        self.assertIsNone(marker)
        self.assertTrue(str(a_root).endswith("ph006/halos/z0.200"))

    def test_hpss_phase_refuses_unverified_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(DensityBuildError):
                resolve_particle_roots(
                    DEFAULT_REGISTRY,
                    self.registry,
                    "ph002",
                    Path(temporary),
                )

    def test_four_particle_directories_must_pass_together(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a_root = root / "a"
            b_root = root / "b"
            self._make_directory(a_root, "field_rv_A", "field_rv_A")
            self._make_directory(a_root, "halo_rv_A", "halo_rv_A")
            self._make_directory(b_root, "field_rv_B", "field_rv_B")
            self._make_directory(b_root, "halo_rv_B", "halo_rv_B")
            report = inspect_particle_inputs(
                self.registry,
                "ph002",
                a_root,
                b_root,
                inspect_headers=False,
            )
            self.assertTrue(report["ready"])
            self.assertEqual(
                {
                    name: record["file_count"]
                    for name, record in report["directories"].items()
                },
                {"field_A": 34, "halo_A": 34, "field_B": 34, "halo_B": 34},
            )

            next((b_root / "halo_rv_B").glob("*.asdf")).unlink()
            with self.assertRaises(DensityBuildError):
                inspect_particle_inputs(
                    self.registry,
                    "ph002",
                    a_root,
                    b_root,
                    inspect_headers=False,
                )


if __name__ == "__main__":
    unittest.main()
