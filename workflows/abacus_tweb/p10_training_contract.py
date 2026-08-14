#!/usr/bin/env python3
"""Shared P10 multi-phase deterministic training contracts.

The loader exposes phase and core identifiers only as sampling provenance.  Model
inputs remain the frozen graph or field arrays; phase is never concatenated to a
node, edge, voxel, or global feature.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np

from workflows.abacus_tweb.p5_graph_patch_utils import CanonicalGraphPatchAdapter
from workflows.abacus_tweb.p6_field_patch_utils import CanonicalFieldPatchAdapter
from workflows.abacus_tweb.p8_epoch_training import epoch_order


TRAINING_PHASES = ("ph000", "ph002", "ph003", "ph004", "ph005")
VALIDATION_PHASE = "ph006"
BLIND_PHASE = "ph001"


def sha256(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class PatchRef:
    """One optimization or evaluation unit.

    Phase is deliberately provenance only.  The dataclass is consumed by the
    loader, never passed to a model.
    """

    phase: str
    core_id: int
    phase_index: int
    phase_objective_scale: float


def _phase_seed(seed: int, epoch: int, phase_index: int) -> int:
    sequence = np.random.SeedSequence([int(seed), int(epoch), int(phase_index)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def phase_balanced_epoch(
    core_ids_by_phase: dict[str, np.ndarray],
    *,
    seed: int,
    epoch: int,
    core_weight_by_phase: dict[str, np.ndarray] | None = None,
) -> tuple[PatchRef, ...]:
    """Interleave one complete without-replacement epoch across phases.

    Every eligible core appears exactly once.  A shuffled round-robin keeps
    phase exposure balanced at every prefix while more than one phase remains.
    The final short tail is only the unavoidable difference in core counts.
    """

    phases = tuple(core_ids_by_phase)
    if not phases:
        raise ValueError("at least one phase is required")
    orders: dict[str, np.ndarray] = {}
    for phase_index, phase in enumerate(phases):
        ids = np.asarray(core_ids_by_phase[phase], dtype=np.int64)
        weights = (
            None
            if core_weight_by_phase is None
            else np.asarray(core_weight_by_phase[phase], dtype=np.float64)
        )
        orders[phase] = epoch_order(
            ids,
            seed=_phase_seed(seed, epoch, phase_index),
            epoch=epoch,
            core_weight=weights,
        )
    total = sum(len(order) for order in orders.values())
    n_phase = len(phases)
    cursors = {phase: 0 for phase in phases}
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch), 991]))
    refs: list[PatchRef] = []
    while len(refs) < total:
        active = [phase for phase in phases if cursors[phase] < len(orders[phase])]
        for phase in np.asarray(active, dtype=object)[rng.permutation(len(active))]:
            phase = str(phase)
            cursor = cursors[phase]
            refs.append(
                PatchRef(
                    phase=phase,
                    core_id=int(orders[phase][cursor]),
                    phase_index=phases.index(phase),
                    phase_objective_scale=float(total) / float(n_phase),
                )
            )
            cursors[phase] += 1
    validate_epoch_refs(refs, core_ids_by_phase)
    return tuple(refs)


def validate_epoch_refs(
    refs: tuple[PatchRef, ...] | list[PatchRef],
    expected_by_phase: dict[str, np.ndarray],
) -> None:
    expected_total = sum(len(np.asarray(values)) for values in expected_by_phase.values())
    if len(refs) != expected_total:
        raise ValueError("epoch does not contain the expected number of cores")
    for phase, expected in expected_by_phase.items():
        found = np.asarray([ref.core_id for ref in refs if ref.phase == phase], dtype=np.int64)
        target = np.asarray(expected, dtype=np.int64)
        if len(found) != len(target) or not np.array_equal(np.sort(found), np.sort(target)):
            raise ValueError(f"epoch coverage mismatch for {phase}")
        if len(np.unique(found)) != len(found):
            raise ValueError(f"duplicate core in epoch for {phase}")


def epoch_hash(refs: tuple[PatchRef, ...] | list[PatchRef]) -> str:
    digest = hashlib.sha256()
    for ref in refs:
        digest.update(
            f"{ref.phase}:{ref.core_id}:{ref.phase_index}:{ref.phase_objective_scale:.17g}\n".encode()
        )
    return digest.hexdigest()


def phase_equal_patch_objective(
    weighted_loss_numerator,
    *,
    phase_weight_denominator: float,
    phase_objective_scale: float,
):
    """Return an optimizer-step loss whose epoch mean is the equal-phase loss.

    Across one complete epoch this yields

        (1/P) sum_phase [sum_i w_i loss_i / sum_i w_i],

    even when phases contain different numbers of cores or rows.
    """

    if not np.isfinite(phase_weight_denominator) or phase_weight_denominator <= 0:
        raise ValueError("phase_weight_denominator must be finite and positive")
    if not np.isfinite(phase_objective_scale) or phase_objective_scale <= 0:
        raise ValueError("phase_objective_scale must be finite and positive")
    return (
        weighted_loss_numerator
        * float(phase_objective_scale)
        / float(phase_weight_denominator)
    )


def resume_state(
    *,
    seed: int,
    epoch: int,
    cursor: int,
    refs: tuple[PatchRef, ...] | list[PatchRef],
    loss_accumulator: dict | None = None,
) -> dict:
    if cursor < 0 or cursor > len(refs):
        raise ValueError("cursor lies outside the epoch")
    return {
        "schema_version": "p10-phase-balanced-resume-v1",
        "seed": int(seed),
        "epoch": int(epoch),
        "cursor": int(cursor),
        "epoch_length": int(len(refs)),
        "epoch_sha256": epoch_hash(refs),
        "loss_accumulator": loss_accumulator or {},
    }


def validate_resume_state(state: dict, refs: tuple[PatchRef, ...] | list[PatchRef]) -> None:
    if state.get("schema_version") != "p10-phase-balanced-resume-v1":
        raise ValueError("unsupported resume schema")
    if int(state["epoch_length"]) != len(refs):
        raise ValueError("resume epoch length mismatch")
    if state["epoch_sha256"] != epoch_hash(refs):
        raise ValueError("resume epoch identity mismatch")
    cursor = int(state["cursor"])
    if cursor < 0 or cursor > len(refs):
        raise ValueError("resume cursor lies outside the epoch")


class P10PhaseBalancedLoader:
    """Lazy graph/field loader backed by a passed training contract."""

    def __init__(
        self,
        contract_root: Path | str = Path(
            "/pscratch/sd/d/dkololgi/abacus/p10_multiphase/training_contract"
        ),
        *,
        include_blind: bool = False,
    ):
        self.root = Path(contract_root)
        marker = self.root / "TRAINING_LOADER_READY.json"
        if not marker.is_file():
            raise RuntimeError(f"missing training readiness marker: {marker}")
        self.manifest = json.loads(marker.read_text())
        if not self.manifest.get("pass"):
            raise RuntimeError("training readiness marker does not pass")
        self.training_phases = tuple(self.manifest["roles"]["training"])
        self.validation_phase = str(self.manifest["roles"]["validation_and_selection"])
        self.blind_phase = str(self.manifest["roles"]["sealed_blind_test"])
        phases = self.training_phases + (self.validation_phase,)
        if include_blind:
            phases += (self.blind_phase,)
        self.phase_records = {
            phase: json.loads((self.root / "phases" / phase / "phase_contract.json").read_text())
            for phase in phases
        }
        self._graph: dict[str, CanonicalGraphPatchAdapter] = {}
        self._field: dict[str, CanonicalFieldPatchAdapter] = {}

    def _array(self, phase: str, name: str, *, mmap_mode: str = "r") -> np.ndarray:
        return np.load(self.root / "phases" / phase / name, mmap_mode=mmap_mode)

    def training_epoch(self, *, seed: int, epoch: int) -> tuple[PatchRef, ...]:
        ids = {
            phase: self._array(phase, "training_core_id.npy")
            for phase in self.training_phases
        }
        weights = {
            phase: self._array(phase, "training_core_weight.npy")
            for phase in self.training_phases
        }
        return phase_balanced_epoch(ids, seed=seed, epoch=epoch, core_weight_by_phase=weights)

    def validation_refs(self) -> tuple[PatchRef, ...]:
        ids = self._array(self.validation_phase, "validation_core_id.npy")
        return tuple(
            PatchRef(
                phase=self.validation_phase,
                core_id=int(core_id),
                phase_index=0,
                phase_objective_scale=1.0,
            )
            for core_id in ids
        )

    def row_weights(self, phase: str) -> np.ndarray:
        return self._array(phase, "active_row_weight.npy")

    def targets_by_parent(self, phase: str) -> np.ndarray:
        if phase == self.blind_phase:
            raise PermissionError("ph001 truth is sealed; loader exposes inputs only")
        return self._array(phase, "parent_eigenvalues.npy")

    def graph_adapter(self, phase: str) -> CanonicalGraphPatchAdapter:
        if phase not in self._graph:
            adapter = CanonicalGraphPatchAdapter(
                self.root / "adapters" / phase / "graph"
            )
            adapter.node_features = np.load(
                self.root / "transforms" / "graph" / phase / "node_features_8d.npy",
                mmap_mode="r",
            )
            self._graph[phase] = adapter
        return self._graph[phase]

    def field_adapter(self, phase: str) -> CanonicalFieldPatchAdapter:
        if phase not in self._field:
            self._field[phase] = CanonicalFieldPatchAdapter(
                self.root / "adapters" / phase / "field",
                selection_manifest=self.root / "transforms" / "field" / "selection_manifest.json",
                rotation=0,
            )
        return self._field[phase]

    @property
    def field_normalization(self) -> dict:
        manifest = json.loads(
            (self.root / "transforms" / "field" / "field_transform.json").read_text()
        )
        return manifest["normalization"]

    @property
    def target_scaler(self) -> dict:
        return json.loads(
            (self.root / "transforms" / "target_scaler.json").read_text()
        )

