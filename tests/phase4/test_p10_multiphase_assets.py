import copy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = REPO_ROOT / "workflows/abacus_tweb"
sys.path.insert(0, str(WORKFLOW_DIR))

from p10_phase_assets import (  # noqa: E402
    DEFAULT_REGISTRY,
    RegistryError,
    expand_phase,
    load_registry,
    parse_htar_listing,
    summarize_b_listing,
    validate_registry,
)
from p10_stage_particle_b import (  # noqa: E402
    parse_checksum_manifest,
    phase_staging_paths,
    verify_checksums,
)


class P10PhaseRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load_registry(DEFAULT_REGISTRY)

    def test_roles_and_uniform_particle_contract_are_frozen(self):
        self.assertEqual(self.registry["phases"]["ph001"]["role"], "sealed_blind")
        self.assertEqual(
            self.registry["phases"]["ph006"]["role"], "validation_and_selection"
        )
        samples = self.registry["target_contract"]["particle_subsamples"]
        self.assertEqual(samples["A_fraction"], 0.03)
        self.assertEqual(samples["B_fraction"], 0.07)
        self.assertEqual(samples["total_fraction"], 0.1)
        self.assertFalse(self.registry["target_contract"]["phase_is_model_input"])

    def test_registry_rejects_nonuniform_total(self):
        broken = copy.deepcopy(self.registry)
        broken["target_contract"]["particle_subsamples"]["total_fraction"] = 0.03
        with self.assertRaises(RegistryError):
            validate_registry(broken)

    def test_phase_paths_expand_consistently(self):
        ph002 = expand_phase(self.registry, "ph002")
        self.assertTrue(ph002["assets"]["cutsky"].endswith("c000_ph002.fits"))
        self.assertTrue(ph002["assets"]["forfa"].endswith("forFA2_nomask.fits"))
        self.assertTrue(
            ph002["assets"]["fiberassign"].endswith(
                "altmtl2/fba2/datcomb_brightwdup.fits"
            )
        )
        self.assertEqual(ph002["particle_b"]["kind"], "hpss")
        self.assertEqual(expand_phase(self.registry, "ph006")["particle_b"]["kind"], "cfs")


class P10HtarListingTests(unittest.TestCase):
    def _listing(self, field_count=34, halo_count=34):
        lines = [
            "-rw-r----- user group 123 2026-01-01 "
            "./halos/z0.200/field_rv_B/checksums.crc32"
        ]
        lines.extend(
            "-rw-r----- user group 1000 2026-01-01 "
            f"./halos/z0.200/field_rv_B/field_rv_B_{index:03d}.asdf"
            for index in range(field_count)
        )
        lines.append(
            "-rw-r----- user group 456 2026-01-01 "
            "./halos/z0.200/halo_rv_B/checksums.crc32"
        )
        lines.extend(
            "-rw-r----- user group 2000 2026-01-01 "
            f"./halos/z0.200/halo_rv_B/halo_rv_B_{index:03d}.asdf"
            for index in range(halo_count)
        )
        return "\n".join(f"HTAR: {line}" for line in lines) + "\nHTAR: HTAR SUCCESSFUL"

    def test_verbose_listing_proves_counts_manifests_and_payload(self):
        records = parse_htar_listing(self._listing())
        summary = summarize_b_listing(records, 34, 34)
        self.assertTrue(summary["ready"])
        self.assertEqual(summary["field_asdf_count"], 34)
        self.assertEqual(summary["halo_asdf_count"], 34)
        self.assertEqual(summary["checksum_manifest_count"], 2)
        self.assertEqual(summary["payload_bytes"], 34 * 1000 + 34 * 2000)

    def test_listing_rejects_missing_slab(self):
        records = parse_htar_listing(self._listing(field_count=33))
        self.assertFalse(summarize_b_listing(records, 34, 34)["ready"])


class P10StageVerificationTests(unittest.TestCase):
    def test_checksum_manifest_and_files_are_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            files = [
                directory / "field_rv_B_000.asdf",
                directory / "field_rv_B_001.asdf",
            ]
            files[0].write_bytes(b"alpha")
            files[1].write_bytes(b"beta")
            result = subprocess.run(
                ["/usr/bin/cksum", *map(str, files)],
                check=True,
                capture_output=True,
                text=True,
            )
            records = []
            for line in result.stdout.splitlines():
                crc, size, name = line.split(maxsplit=2)
                records.append(f"{crc} {size} {Path(name).name}")
            manifest = directory / "checksums.crc32"
            manifest.write_text("\n".join(records) + "\n")

            parsed = parse_checksum_manifest(manifest)
            self.assertEqual(set(parsed), {path.name for path in files})
            verified = verify_checksums(
                directory, "field_rv_B_*.asdf", "checksums.crc32"
            )
            self.assertTrue(verified["verified"])
            self.assertEqual(verified["file_count"], 2)

    def test_staging_layout_matches_htar_member_layout(self):
        phase_root, particle_root, marker = phase_staging_paths(
            Path("/scratch/p10"), "ph002"
        )
        self.assertEqual(
            particle_root,
            phase_root / "halos/z0.200",
        )
        self.assertEqual(marker, phase_root / "B_STAGE_COMPLETE.json")


if __name__ == "__main__":
    unittest.main()
