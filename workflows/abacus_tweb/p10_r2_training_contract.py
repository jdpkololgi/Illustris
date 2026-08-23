"""P10 loader for the stored R2 random-plus-assignment response view."""
from __future__ import annotations

from workflows.abacus_tweb.p10_training_contract import P10PhaseBalancedLoader
from workflows.abacus_tweb.p6_field_patch_utils import CanonicalFieldPatchAdapter


class P10AssignmentResponseLoader(P10PhaseBalancedLoader):
    """Read response channels exactly as frozen in the P3c R2 overlays."""

    def field_adapter(self, phase: str) -> CanonicalFieldPatchAdapter:
        if phase not in self._field:
            self._field[phase] = CanonicalFieldPatchAdapter(
                self.root / "adapters" / phase / "field",
                selection_manifest=None,
                rotation=None,
            )
        return self._field[phase]
