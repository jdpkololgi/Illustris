import json
from pathlib import Path
import tempfile
import unittest

from workflows.sbi.p12_authorize_blind_response import (
    build_authority,
    validate_blind_authority,
)


class BlindResponseAuthorityTest(unittest.TestCase):
    def _artifact(self, root: Path, name: str, payload: dict) -> Path:
        path = root / name
        path.write_text(json.dumps(payload, sort_keys=True))
        return path

    def test_authority_is_content_addressed_and_fails_after_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = self._artifact(root, "candidate.json", {
                "schema_version": "p12a-production-candidate-frozen-v1",
                "pass": True, "sealed_phase_opened": False,
                "truth_files_read": [], "open_count": 0,
            })
            selection = self._artifact(root, "selection.json", {
                "schema_version": "p12f-no-field-finalist-v1",
                "pass": True, "ph001_opened": False,
                "truth_files_read": ["ph006 density/T-web"], "open_count": 0,
            })
            blind = self._artifact(root, "blind.json", {
                "schema_version": "p10-phase-input-complete-v1", "pass": True,
                "phase": "ph001", "role": "sealed_blind",
                "blind_gates": {"density_absent": True, "tweb_absent": True},
            })
            output = root / "authority.json"
            build_authority(
                p12a_candidate=candidate, p12f_selection=selection,
                blind_input=blind, output=output,
            )
            self.assertEqual(validate_blind_authority(output)["phase"], "ph001")
            candidate.write_text(candidate.read_text() + "\n")
            with self.assertRaises(PermissionError):
                validate_blind_authority(output)

    def test_authority_rejects_opened_blind_phase(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = self._artifact(root, "candidate.json", {
                "schema_version": "p12a-production-candidate-frozen-v1",
                "pass": True, "sealed_phase_opened": True,
            })
            selection = self._artifact(root, "selection.json", {
                "schema_version": "p12f-no-field-finalist-v1", "pass": True,
            })
            blind = self._artifact(root, "blind.json", {
                "schema_version": "p10-phase-input-complete-v1", "pass": True,
                "phase": "ph001", "role": "sealed_blind",
                "blind_gates": {"density_absent": True},
            })
            with self.assertRaises(PermissionError):
                build_authority(
                    p12a_candidate=candidate, p12f_selection=selection,
                    blind_input=blind, output=root / "authority.json",
                )


if __name__ == "__main__":
    unittest.main()
