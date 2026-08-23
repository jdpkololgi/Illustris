"""P10 R3-RF loader for the high-S/N empirical random response field."""
from __future__ import annotations

from pathlib import Path

from workflows.abacus_tweb.p10_training_contract import P10PhaseBalancedLoader
from workflows.abacus_tweb.p6_field_patch_utils import CanonicalFieldPatchAdapter


R3_RF_MODEL_CHANNELS = (
    "counts",
    "exposure_apodized",
    "log_count_ratio",
    "expected_counts_random",
    "angular_response",
    "support_random",
)


class P10RawRandomFieldLoader(P10PhaseBalancedLoader):
    """Use the frozen all-18 P3b-R spatial response overlay without re-derivation."""

    def field_adapter(self, phase: str) -> CanonicalFieldPatchAdapter:
        if phase not in self._field:
            self._field[phase] = CanonicalFieldPatchAdapter(
                self.root / "adapters" / phase / "field",
                selection_manifest=None,
                rotation=None,
            )
        return self._field[phase]

