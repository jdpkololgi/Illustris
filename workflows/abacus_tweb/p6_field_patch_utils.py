#!/usr/bin/env python3
"""Canonical P6 field-patch views for U-Net and field-based estimators.

Patches are immutable slices of the P3 cap lattices. The adapter never
re-voxelizes galaxies and never computes patch-local normalization.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


CAP_NAME = {0: "SGC", 1: "NGC"}


@dataclass(frozen=True)
class FieldPatch:
    core_id: int
    fold: int
    cap: int
    channel_names: tuple[str, ...]
    values: np.ndarray
    context_start: np.ndarray
    context_stop: np.ndarray
    core_start: np.ndarray
    core_stop: np.ndarray
    core_slice: tuple[slice, slice, slice]
    authoritative_parent_id: np.ndarray
    authoritative_frac_index_global: np.ndarray
    authoritative_frac_index_local: np.ndarray
    available_halo_low: np.ndarray
    available_halo_high: np.ndarray

    @property
    def core_values(self) -> np.ndarray:
        return self.values[(slice(None),) + self.core_slice]

    @property
    def unsupported_mask(self) -> np.ndarray:
        index = self.channel_names.index("exposure_binary")
        return self.values[index] <= 0


def fractional_cell_index(
    xyz_mpc: np.ndarray, origin_mpc: np.ndarray, cell_mpc: float
) -> np.ndarray:
    """Fractional index of cell centres for the P3 ix,iy,iz convention."""
    return (
        (np.asarray(xyz_mpc, dtype=np.float64) - np.asarray(origin_mpc, dtype=np.float64))
        / float(cell_mpc)
        - 0.5
    )


def trilinear_sample(
    field: np.ndarray, frac_index: np.ndarray, *, mode: str = "nearest"
) -> np.ndarray:
    """Sample a CXYZ or XYZ array at fractional cell-centre indices.

    Nearest padding matches PyTorch grid_sample border padding with
    align_corners=True at the lattice boundary.
    """
    values = np.asarray(field)
    scalar = values.ndim == 3
    if scalar:
        values = values[None, ...]
    if values.ndim != 4:
        raise ValueError("field must have shape (C,nx,ny,nz) or (nx,ny,nz)")
    coords = np.asarray(frac_index, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("frac_index must have shape (N,3)")
    shape = np.asarray(values.shape[1:], dtype=np.int64)
    if mode != "nearest":
        raise ValueError("only border/nearest padding is registered")
    clipped = np.clip(coords, 0.0, shape - 1.0)
    lo = np.floor(clipped).astype(np.int64)
    hi = np.minimum(lo + 1, shape - 1)
    weight = clipped - lo
    out = np.zeros((len(coords), values.shape[0]), dtype=np.float64)
    for dx in (0, 1):
        wx = (1.0 - weight[:, 0]) if dx == 0 else weight[:, 0]
        ix = lo[:, 0] if dx == 0 else hi[:, 0]
        for dy in (0, 1):
            wy = (1.0 - weight[:, 1]) if dy == 0 else weight[:, 1]
            iy = lo[:, 1] if dy == 0 else hi[:, 1]
            for dz in (0, 1):
                wz = (1.0 - weight[:, 2]) if dz == 0 else weight[:, 2]
                iz = lo[:, 2] if dz == 0 else hi[:, 2]
                out += (
                    values[:, ix, iy, iz].T.astype(np.float64)
                    * (wx * wy * wz)[:, None]
                )
    out = out.astype(np.float32)
    return out[:, 0] if scalar else out


def channel_transform(name: str, values: np.ndarray) -> np.ndarray:
    """Registered pre-normalization transform; never fitted per patch."""
    values = np.asarray(values, dtype=np.float32)
    if name in {"counts", "expected_counts"}:
        return np.log1p(np.maximum(values, 0.0))
    if name == "ntilde_mpc3":
        return np.log(np.maximum(values, np.float32(1e-12)))
    return values


def apply_frozen_normalization(
    patch: FieldPatch, normalization: dict
) -> np.ndarray:
    """Apply one fold's frozen statistics to a patch without refitting."""
    output = np.empty_like(patch.values, dtype=np.float32)
    for channel, name in enumerate(patch.channel_names):
        transformed = channel_transform(name, patch.values[channel])
        spec = normalization["channels"][name]
        if spec["policy"] == "identity":
            output[channel] = transformed
        elif spec["policy"] == "zscore":
            output[channel] = (
                (transformed - np.float32(spec["mean"]))
                / np.float32(max(spec["std"], 1e-6))
            )
        else:
            raise ValueError(f"unknown normalization policy for {name}: {spec}")
    return output


class CanonicalFieldPatchAdapter:
    """Lazy cap-lattice reader backed by P3 and a compact P6 core index."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "adapter_manifest.json").read_text())
        self.channel_names = tuple(self.manifest["channel_order"])
        self.core_start = np.load(self.root / "core_voxel_start.npy", mmap_mode="r")
        self.core_stop = np.load(self.root / "core_voxel_stop.npy", mmap_mode="r")
        self.core_fold = np.load(self.root / "core_fold.npy", mmap_mode="r")
        self.core_cap = np.load(self.root / "core_cap.npy", mmap_mode="r")
        self.core_offsets = np.load(self.root / "core_active_offsets.npy", mmap_mode="r")
        self.core_parent = np.load(self.root / "core_active_parent.npy", mmap_mode="r")
        self.core_frac = np.load(self.root / "core_active_frac_index.npy", mmap_mode="r")
        self._handles: dict[int, h5py.File] = {}

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _handle(self, cap: int) -> h5py.File:
        cap = int(cap)
        if cap not in self._handles:
            path = self.manifest["caps"][CAP_NAME[cap]]["field_path"]
            self._handles[cap] = h5py.File(path, "r")
        return self._handles[cap]

    def authoritative(self, core_id: int) -> tuple[np.ndarray, np.ndarray]:
        start = int(self.core_offsets[core_id])
        stop = int(self.core_offsets[core_id + 1])
        return (
            np.asarray(self.core_parent[start:stop], dtype=np.int64),
            np.asarray(self.core_frac[start:stop], dtype=np.float64),
        )

    def extract(
        self,
        core_id: int,
        context_halo_voxels: int | Iterable[int],
        channel_names: Iterable[str] | None = None,
    ) -> FieldPatch:
        core_id = int(core_id)
        cap = int(self.core_cap[core_id])
        handle = self._handle(cap)
        names = self.channel_names if channel_names is None else tuple(channel_names)
        unknown = set(names).difference(handle.keys())
        if unknown:
            raise KeyError(f"unknown P3 channels: {sorted(unknown)}")
        halo = np.broadcast_to(
            np.asarray(context_halo_voxels, dtype=np.int64), (3,)
        ).copy()
        if np.any(halo < 0):
            raise ValueError("context halo must be non-negative")
        start = np.asarray(self.core_start[core_id], dtype=np.int64)
        stop = np.asarray(self.core_stop[core_id], dtype=np.int64)
        shape = np.asarray(handle[names[0]].shape, dtype=np.int64)
        requested_start = start - halo
        requested_stop = stop + halo
        context_start = np.maximum(requested_start, 0)
        context_stop = np.minimum(requested_stop, shape)
        selection = tuple(
            slice(int(left), int(right))
            for left, right in zip(context_start, context_stop)
        )
        values = np.stack(
            [np.asarray(handle[name][selection], dtype=np.float32) for name in names],
            axis=0,
        )
        parent, frac_global = self.authoritative(core_id)
        frac_local = frac_global - context_start[None, :]
        local_start = start - context_start
        local_stop = stop - context_start
        core_slice = tuple(
            slice(int(left), int(right))
            for left, right in zip(local_start, local_stop)
        )
        return FieldPatch(
            core_id=core_id,
            fold=int(self.core_fold[core_id]),
            cap=cap,
            channel_names=names,
            values=values,
            context_start=context_start,
            context_stop=context_stop,
            core_start=start,
            core_stop=stop,
            core_slice=core_slice,
            authoritative_parent_id=parent,
            authoritative_frac_index_global=frac_global,
            authoritative_frac_index_local=frac_local,
            available_halo_low=start - context_start,
            available_halo_high=context_stop - stop,
        )


def sample_patch(patch: FieldPatch) -> np.ndarray:
    return trilinear_sample(patch.values, patch.authoritative_frac_index_local)


def sample_core_values(patch: FieldPatch) -> np.ndarray:
    local_to_core = (
        patch.authoritative_frac_index_global - patch.core_start[None, :]
    )
    return trilinear_sample(patch.core_values, local_to_core)
