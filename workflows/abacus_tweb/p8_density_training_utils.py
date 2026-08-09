"""Shared exact-owner patch utilities for U-DENSITY-PHYS-v1."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from workflows.abacus_tweb.p6_field_patch_utils import CanonicalFieldPatchAdapter
from workflows.abacus_tweb.p8_prepare_density_training import CAP_NAME, shell_index_for_core
from workflows.abacus_tweb.p8_train_unet_patch import (
    ALIGNMENT_VOXELS,
    CHANNELS,
    HALO_VOXELS,
    model_inputs,
)


ROOT = Path("/pscratch/sd/d/dkololgi/abacus")
TRAINING_CONTRACT = ROOT / "p8_density_phys_v1/training_contract"
OUTPUT_CORES = ROOT / "p8_density_phys_v1/field_output_tiling/field_output_cores.npz"
TARGET_MANIFEST = ROOT / "p8_density_phys_v1/targets/target_manifest.json"
FIELD_ADAPTER = ROOT / "p6_unet_patch_adapter"
SELECTION = FIELD_ADAPTER / "fullcap_selection_v1/selection_manifest.json"


def extract_core_prediction(prediction: torch.Tensor, core_slice: tuple[slice, slice, slice]) -> torch.Tensor:
    if prediction.ndim != 5 or prediction.shape[0] != 1 or prediction.shape[1] != 1:
        raise ValueError("density model prediction must have shape [1,1,nx,ny,nz]")
    return prediction[(0, 0) + core_slice]


class DensityUnitAdapter:
    """Join one exact output owner to its contextual Bright inputs and privileged target."""

    def __init__(
        self,
        *,
        rotation: int,
        contract_root: Path = TRAINING_CONTRACT,
        output_cores: Path = OUTPUT_CORES,
        target_manifest: Path = TARGET_MANIFEST,
        field_adapter: Path = FIELD_ADAPTER,
        selection: Path = SELECTION,
    ):
        self.rotation = int(rotation)
        self.contract_dir = Path(contract_root) / f"rotation_{self.rotation}"
        self.config = json.loads((self.contract_dir / "d0_config.json").read_text())
        self.scaler = json.loads((self.contract_dir / "target_scaler.json").read_text())
        self.units = np.load(self.contract_dir / "density_units.npy", mmap_mode="r")
        self.cores = np.load(output_cores, mmap_mode="r")
        output_ids = np.asarray(self.cores["output_core_id"], dtype=np.int64)
        if not np.array_equal(output_ids, np.arange(len(output_ids))):
            raise RuntimeError("output_core_id must be exact field-output row identity")
        self.target_manifest = json.loads(Path(target_manifest).read_text())
        selection_manifest = json.loads(Path(selection).read_text())
        self.normalization = selection_manifest["rotations"][str(self.rotation)]["normalization"]
        self.field = CanonicalFieldPatchAdapter(
            field_adapter, selection_manifest=selection, rotation=self.rotation
        )
        self.target_handles: dict[int, h5py.File] = {}

    def close(self) -> None:
        self.field.close()
        for handle in self.target_handles.values():
            handle.close()
        self.target_handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def target_handle(self, cap: int) -> h5py.File:
        cap = int(cap)
        if cap not in self.target_handles:
            path = self.target_manifest["components"][CAP_NAME[cap]]["file"]
            self.target_handles[cap] = h5py.File(path, "r")
        return self.target_handles[cap]

    def find_unit(self, output_core_id: int, shell: int) -> np.void:
        selected = np.flatnonzero(
            (self.units["output_core_id"] == int(output_core_id))
            & (self.units["shell"] == int(shell))
        )
        if len(selected) != 1:
            raise RuntimeError(
                f"expected one unit for output_core={output_core_id}, shell={shell}; got {len(selected)}"
            )
        return self.units[int(selected[0])]

    def extract(self, unit: np.void, device: str) -> tuple[object, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        row = int(unit["output_core_id"])
        if bool(self.cores["inference_only"][row]) or not bool(self.cores["owns_density_loss"][row]):
            raise RuntimeError("density training unit points to a non-loss output owner")
        cap = int(self.cores["cap"][row])
        start = np.asarray(self.cores["voxel_start"][row], dtype=np.int64)
        stop = np.asarray(self.cores["voxel_stop"][row], dtype=np.int64)
        patch = self.field.extract_bounds(
            cap=cap,
            core_start=start,
            core_stop=stop,
            context_halo_voxels=HALO_VOXELS,
            channel_names=CHANNELS,
            alignment_voxels=ALIGNMENT_VOXELS,
            core_id=int(unit["nominal_core_id"]),
            fold=int(unit["fold"]),
            authoritative_parent_id=np.empty(0, dtype=np.int64),
            authoritative_frac_index_global=np.empty((0, 3), dtype=np.float64),
        )
        values, _ = model_inputs(patch, self.normalization, device)
        selection = tuple(slice(int(start[a]), int(stop[a])) for a in range(3))
        handle = self.target_handle(cap)
        target = np.asarray(handle["delta_r7"][selection], dtype=np.float32)
        support = np.asarray(handle["science_support"][selection], dtype=bool)
        grid = self.target_manifest["components"][CAP_NAME[cap]]["grid"]
        shell_index = shell_index_for_core(
            start, stop,
            origin_mpc=np.asarray(grid["origin_mpc"], dtype=np.float64),
            cell_mpc=float(grid["cell_mpc"]),
        )
        mask = support & (shell_index == int(unit["shell"]))
        if int(mask.sum()) != int(unit["supported_voxels"]):
            raise RuntimeError(
                f"unit voxel count mismatch: {int(mask.sum())} != {int(unit['supported_voxels'])}"
            )
        target_scaled = (
            (target - np.float32(self.scaler["mean"]))
            / np.float32(self.scaler["std"])
        )
        diagnostics = {
            "output_core_id": row,
            "nominal_core_id": int(unit["nominal_core_id"]),
            "cap": cap,
            "fold": int(unit["fold"]),
            "shell": int(unit["shell"]),
            "supported_voxels": int(mask.sum()),
            "context_shape": list(values.shape[2:]),
            "core_shape": list(target.shape),
        }
        return (
            patch,
            values,
            torch.from_numpy(target_scaled).to(device),
            torch.from_numpy(mask).to(device),
            diagnostics,
        )
