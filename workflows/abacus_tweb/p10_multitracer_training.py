"""P10 U-PATCH adapter for separate BRIGHT and FAINT field channels."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

from workflows.abacus_tweb.p6_field_patch_utils import (
    CAP_NAME,
    channel_transform,
    derive_selection_channels,
    patch_redshift,
)
from workflows.abacus_tweb.p8_train_unet_patch import (
    ALIGNMENT_VOXELS,
    CHANNELS,
    HALO_VOXELS,
    UNet3D,
    grid_coordinates,
)


class P10MultitracerUPatch(nn.Module):
    def __init__(self, base: int = 24, latent_channels: int = 32, head_width: int = 128):
        super().__init__()
        self.unet = UNet3D(6, latent_channels, base)
        self.head = nn.Sequential(
            nn.Linear(latent_channels, head_width),
            nn.SiLU(),
            nn.Linear(head_width, head_width),
            nn.SiLU(),
            nn.Linear(head_width, 3),
        )

    def sample_latent(self, values: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        latent = self.unet(values)
        return torch.nn.functional.grid_sample(
            latent, points, mode="bilinear", align_corners=True, padding_mode="border"
        )[0, :, 0, 0].T

    def forward(self, values: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        return self.head(self.sample_latent(values, points))


class P10MultitracerFieldAdapter:
    """Overlay phase-local FAINT Proxy/Null counts on the frozen BRIGHT adapter."""

    def __init__(self, *, loader, phase: str, root: Path, view: str):
        if view not in ("proxy", "null"):
            raise ValueError("multitracer view must be proxy or null")
        self.phase = phase
        self.view = view
        self.root = Path(root)
        ready = json.loads((self.root / "P10_MULTITRACER_VIEWS_READY.json").read_text())
        if not ready.get("pass") or ready.get("sealed_phase_opened"):
            raise RuntimeError("P10 multitracer view contract does not pass")
        self.selection = json.loads((self.root / "selection_manifest.json").read_text())
        phase_root = self.root / "phases" / phase
        phase_row = json.loads((phase_root / "PHASE_MULTITRACER_VIEWS_READY.json").read_text())
        if not phase_row.get("pass"):
            raise RuntimeError(f"{phase} multitracer phase contract does not pass")
        self.proxy = phase_row["proxy"]
        self.null = phase_row["null"]
        self.base = loader.field_adapter(phase)
        self.bright_normalization = loader.field_normalization
        self.core_cap = self.base.core_cap
        self.proxy_handles: dict[int, h5py.File] = {}
        self.count_handles: dict[int, h5py.File] = {}
        cosmology = self.selection["cosmology"]
        self.radius_grid = np.asarray(cosmology["radius_grid_mpc"], dtype=np.float64)
        self.redshift_grid = np.asarray(cosmology["redshift_grid"], dtype=np.float64)

    def close(self) -> None:
        for handle in self.proxy_handles.values():
            handle.close()
        for handle in self.count_handles.values():
            handle.close()
        self.proxy_handles.clear()
        self.count_handles.clear()

    def _proxy_handle(self, cap: int) -> h5py.File:
        if cap not in self.proxy_handles:
            path = self.proxy["components"][CAP_NAME[cap]]["file"]
            self.proxy_handles[cap] = h5py.File(path, "r")
        return self.proxy_handles[cap]

    def _count_handle(self, cap: int) -> h5py.File:
        if self.view == "proxy":
            return self._proxy_handle(cap)
        if cap not in self.count_handles:
            path = self.null["components"][CAP_NAME[cap]]["file"]
            self.count_handles[cap] = h5py.File(path, "r")
        return self.count_handles[cap]

    def extract(self, core_id: int):
        bright = self.base.extract(
            core_id, HALO_VOXELS, CHANNELS, alignment_voxels=ALIGNMENT_VOXELS
        )
        selection = tuple(
            slice(int(start), int(stop))
            for start, stop in zip(bright.context_start, bright.context_stop)
        )
        proxy = self._proxy_handle(bright.cap)
        count_handle = self._count_handle(bright.cap)
        count_name = "counts" if self.view == "proxy" else "faint_counts"
        faint_counts = np.asarray(count_handle[count_name][selection], dtype=np.float32)
        faint_exposure = np.asarray(proxy["exposure_apodized"][selection], dtype=np.float32)
        cap_name = CAP_NAME[bright.cap]
        grid = self.proxy["components"][cap_name]["grid"]
        redshift = patch_redshift(
            origin_mpc=np.asarray(grid["origin_mpc"], dtype=np.float64),
            cell_mpc=float(grid["cell_mpc"]),
            context_start=bright.context_start,
            shape=tuple(int(value) for value in bright.context_stop - bright.context_start),
            radius_grid_mpc=self.radius_grid,
            redshift_grid=self.redshift_grid,
        )
        curve = self.selection["caps"][cap_name]
        derived = derive_selection_channels(
            faint_counts,
            faint_exposure,
            redshift,
            cell_mpc=float(grid["cell_mpc"]),
            grid_z=np.asarray(curve["grid_z"], dtype=np.float64),
            ntilde=np.asarray(curve["ntilde"], dtype=np.float64),
            epsilon=float(self.selection["contrast"]["epsilon"]),
            minimum_exposure=float(self.selection["contrast"]["minimum_exposure"]),
        )
        return bright, faint_counts, faint_exposure, derived["log_count_ratio"]


def _bright_zscore(values: np.ndarray, spec: dict, transform: str) -> np.ndarray:
    transformed = channel_transform(transform, values)
    return ((transformed - spec["mean"]) / max(spec["std"], 1.0e-6)).astype(np.float32)


def model_inputs(adapter: P10MultitracerFieldAdapter, extracted, device: str):
    bright, faint_counts, faint_exposure, faint_log_ratio = extracted
    at = {name: index for index, name in enumerate(bright.channel_names)}
    bright_norm = adapter.bright_normalization["channels"]
    bright_count = _bright_zscore(bright.values[at["counts"]], bright_norm["counts"], "counts")
    bright_density = np.clip(
        np.expm1(np.clip(bright.values[at["log_count_ratio"]], -20.0, 4.0)),
        -1.0,
        20.0,
    ).astype(np.float32)
    faint_density = np.clip(
        np.expm1(np.clip(faint_log_ratio, -20.0, 4.0)), -1.0, 20.0
    ).astype(np.float32)
    spec = adapter.selection["faint_count_normalization"]
    faint_count = ((np.log1p(faint_counts) - spec["mean"]) / max(spec["std"], 1.0e-6)).astype(np.float32)
    values = np.stack((
        bright_count,
        bright_density,
        bright.values[at["exposure_apodized"]],
        faint_count,
        faint_density,
        faint_exposure,
    )).astype(np.float32)
    tensor = torch.from_numpy(values[None]).to(device)
    points = grid_coordinates(
        bright.authoritative_frac_index_local, tuple(values.shape[1:]), device
    )
    return bright, tensor, points


def predict_phase(model, adapter, core_ids, device):
    model.eval()
    parents, predictions = [], []
    with torch.inference_mode():
        for core_id in core_ids:
            bright, values, points = model_inputs(adapter, adapter.extract(int(core_id)), device)
            parents.append(bright.authoritative_parent_id)
            predictions.append(model(values, points).cpu().numpy())
    return np.concatenate(parents), np.concatenate(predictions), []
