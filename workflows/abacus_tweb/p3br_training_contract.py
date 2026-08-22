"""P10 loader for the stored, random-derived P3b-R three-channel view."""
from __future__ import annotations

from pathlib import Path

from workflows.abacus_tweb.p10_training_contract import P10PhaseBalancedLoader
from workflows.abacus_tweb.p6_field_patch_utils import CanonicalFieldPatchAdapter


class P10RandomResponseLoader(P10PhaseBalancedLoader):
    """Use stored random-response channels rather than re-deriving P3a channels.

    The P3b-R overlay already combines the frozen BRIGHT radial curve with the
    random-derived angular response.  Passing the old P3a selection manifest to
    ``CanonicalFieldPatchAdapter`` would silently discard that angular factor.
    """

    def field_adapter(self, phase: str) -> CanonicalFieldPatchAdapter:
        if phase not in self._field:
            self._field[phase] = CanonicalFieldPatchAdapter(
                self.root / "adapters" / phase / "field",
                selection_manifest=None,
                rotation=None,
            )
        return self._field[phase]

