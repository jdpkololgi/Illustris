"""Literature-grade, CutSky-safe diffusion components for P12-F3-D2.

The model deliberately retains the exact F3-L2d Fourier target and seven-channel
condition.  Only the denoiser architecture changes.  All convolutions use
ordinary zero padding, and normalization is per voxel across channels.  Spatial
GroupNorm/InstanceNorm would make a core prediction depend on patch extent and
therefore violates the frozen P6/P8 patch-safety contract.
"""
from __future__ import annotations

import math
import os
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from workflows.abacus_tweb.p8_train_unet_patch import ChannelLayerNorm3d
from workflows.sbi.p12f3_conditional_models import cosine_alpha_sigma
from workflows.sbi.p12f3_fourier_modes import (
    FourierModeLayout,
    pack_fourier_components,
    unpack_fourier_components,
    unwhiten_components,
    whiten_velocity,
)


CUDA_DETERMINISM_POLICY = {
    "cuda_deterministic_algorithms": True,
    "cublas_workspace_config": ":4096:8",
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "allow_tf32": False,
    "float32_matmul_precision": "highest",
    "numerical_replay_absolute_tolerance": 1.0e-6,
    "resume_claim_policy": (
        "state serialization is exact; post-update numerical replay is claimed only "
        "after the registered one-GPU smoke passes and is not generalized beyond the "
        "tested topology"
    ),
}


def configure_d2_determinism(policy: Mapping[str, object], device: str) -> dict:
    """Apply the frozen one-GPU deterministic-kernel policy before CUDA use.

    ``CUBLAS_WORKSPACE_CONFIG`` must already be present in the process
    environment, because changing it after a cuBLAS handle is created is too
    late.  The reviewed launchers export it before Python starts; direct entry
    points fail closed rather than silently running with weaker semantics.
    """
    if dict(policy) != CUDA_DETERMINISM_POLICY:
        raise RuntimeError("D2 CUDA determinism policy changed")
    status = {
        "requested_device": str(device),
        "policy": dict(CUDA_DETERMINISM_POLICY),
        "applied": False,
    }
    if not str(device).startswith("cuda"):
        return status
    actual_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    expected_workspace = str(policy["cublas_workspace_config"])
    if actual_workspace != expected_workspace:
        raise RuntimeError(
            "D2 CUDA requires CUBLAS_WORKSPACE_CONFIG="
            f"{expected_workspace} before Python starts; found {actual_workspace!r}"
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    status.update(
        {
            "applied": True,
            "torch_deterministic_algorithms_enabled": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cublas_workspace_config": actual_workspace,
        }
    )
    if (
        not status["torch_deterministic_algorithms_enabled"]
        or not status["cudnn_deterministic"]
        or status["cudnn_benchmark"]
        or status["cuda_matmul_allow_tf32"]
        or status["cudnn_allow_tf32"]
        or status["float32_matmul_precision"] != "highest"
    ):
        raise RuntimeError("D2 CUDA deterministic policy was not applied exactly")
    return status


def log_snr(time: torch.Tensor, *, epsilon: float = 1.0e-5) -> torch.Tensor:
    """Cosine-VP log signal-to-noise ratio used by the time embedding."""
    clipped = torch.clamp(time, epsilon, 1.0 - epsilon)
    alpha, sigma = cosine_alpha_sigma(clipped)
    return 2.0 * (torch.log(alpha) - torch.log(sigma))


class SinusoidalLogSNREmbedding(nn.Module):
    """Embed continuous diffusion time through the schedule's log-SNR."""

    def __init__(self, dimension: int):
        super().__init__()
        if dimension < 8 or dimension % 2:
            raise ValueError("time embedding dimension must be even and at least eight")
        self.dimension = int(dimension)
        half = dimension // 2
        frequencies = torch.exp(
            -math.log(10_000.0) * torch.arange(half, dtype=torch.float32) / max(half - 1, 1)
        )
        self.register_buffer("frequencies", frequencies, persistent=True)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        if time.ndim != 1:
            raise ValueError("diffusion time must have one value per batch element")
        phase = log_snr(time)[:, None] * self.frequencies[None]
        return torch.cat((torch.sin(phase), torch.cos(phase)), dim=1)


class TimeResidualBlock3D(nn.Module):
    """Patch-safe channel-normalized residual block with time conditioning."""

    def __init__(self, input_channels: int, output_channels: int, time_channels: int):
        super().__init__()
        self.norm1 = ChannelLayerNorm3d(input_channels)
        self.conv1 = nn.Conv3d(input_channels, output_channels, 3, padding=1)
        self.time_projection = nn.Linear(time_channels, output_channels)
        self.norm2 = ChannelLayerNorm3d(output_channels)
        self.conv2 = nn.Conv3d(output_channels, output_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv3d(input_channels, output_channels, 1)
        )

    def forward(self, values: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(values)))
        hidden = hidden + self.time_projection(F.silu(time_embedding))[:, :, None, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return hidden + self.skip(values)


class CoarseAttention3D(nn.Module):
    """Self-attention restricted to the coarsest spatial representation."""

    def __init__(self, channels: int, heads: int):
        super().__init__()
        if heads <= 0 or channels % heads:
            raise ValueError("attention heads must divide bottleneck channels")
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, values: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        batch, channels, nx, ny, nz = values.shape
        if support.shape != (batch, 1, nx, ny, nz):
            raise ValueError("coarse attention support mask is not spatially aligned")
        valid = support[:, 0].reshape(batch, nx * ny * nz).bool()
        if not torch.all(valid.any(dim=1)):
            raise RuntimeError("coarse attention received a wholly unsupported patch")
        sequence = values.reshape(batch, channels, nx * ny * nz).transpose(1, 2)
        sequence = self.norm(sequence)
        attended, _ = self.attention(
            sequence,
            sequence,
            sequence,
            key_padding_mask=~valid,
            need_weights=False,
        )
        attended = attended * valid[:, :, None]
        attended = attended.transpose(1, 2).reshape(batch, channels, nx, ny, nz)
        return values + attended


def coarsen_support_any(
    support: torch.Tensor, shape: tuple[int, int, int], *, levels: int = 3
) -> torch.Tensor:
    """Propagate exact support through the U-Net's stride-2 receptive fields."""
    if support.ndim != 5 or support.shape[1] != 1:
        raise ValueError("exact support must have shape [batch,1,x,y,z]")
    if any(int(value) <= 0 for value in shape):
        raise ValueError("coarse support shape must be positive")
    if levels <= 0:
        raise ValueError("support propagation requires a positive level count")
    coarse = support.to(dtype=torch.float32)
    for _ in range(levels):
        coarse = F.max_pool3d(coarse, kernel_size=3, stride=2, padding=1)
    if tuple(coarse.shape[-3:]) != tuple(shape):
        raise RuntimeError("support propagation differs from bottleneck convolution shape")
    return coarse > 0


class D2ResidualUNet3D(nn.Module):
    """Four-level residual U-Net with multilevel time injection."""

    def __init__(
        self,
        *,
        input_channels: int,
        output_channels: int = 1,
        base: int = 8,
        time_channels: int = 64,
        coarse_attention: bool = False,
        attention_heads: int = 4,
    ):
        super().__init__()
        if base < 4:
            raise ValueError("D2 base width must be at least four")
        self.time_embedding = SinusoidalLogSNREmbedding(time_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_channels, time_channels * 4),
            nn.SiLU(),
            nn.Linear(time_channels * 4, time_channels),
        )
        self.input = nn.Conv3d(input_channels, base, 3, padding=1)
        self.enc0 = TimeResidualBlock3D(base, base, time_channels)
        self.down1 = nn.Conv3d(base, base * 2, 3, stride=2, padding=1)
        self.enc1 = TimeResidualBlock3D(base * 2, base * 2, time_channels)
        self.down2 = nn.Conv3d(base * 2, base * 4, 3, stride=2, padding=1)
        self.enc2 = TimeResidualBlock3D(base * 4, base * 4, time_channels)
        self.down3 = nn.Conv3d(base * 4, base * 4, 3, stride=2, padding=1)
        self.bottleneck = TimeResidualBlock3D(base * 4, base * 4, time_channels)
        self.attention = (
            CoarseAttention3D(base * 4, attention_heads)
            if coarse_attention
            else None
        )
        self.dec2 = TimeResidualBlock3D(base * 8, base * 4, time_channels)
        self.dec1 = TimeResidualBlock3D(base * 6, base * 2, time_channels)
        self.dec0 = TimeResidualBlock3D(base * 3, base, time_channels)
        self.output_norm = ChannelLayerNorm3d(base)
        self.output = nn.Conv3d(base, output_channels, 3, padding=1)

    @staticmethod
    def _up(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(values, size=reference.shape[-3:], mode="trilinear", align_corners=False)

    def forward(
        self,
        values: torch.Tensor,
        time: torch.Tensor,
        *,
        support: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if values.ndim != 5 or time.shape != (values.shape[0],):
            raise ValueError("D2 U-Net expects [batch,channel,x,y,z] and one time per batch")
        embedded = self.time_mlp(self.time_embedding(time))
        enc0 = self.enc0(self.input(values), embedded)
        enc1 = self.enc1(self.down1(enc0), embedded)
        enc2 = self.enc2(self.down2(enc1), embedded)
        bottleneck = self.bottleneck(self.down3(enc2), embedded)
        if self.attention is not None:
            if support is None:
                raise ValueError("attention-enabled D2 model requires explicit support")
            # A coarse token is valid when any exact-M fine voxel in the three
            # stride-2 convolutional receptive fields is valid.  Nearest resize
            # and generic adaptive binning both misdescribe this architecture.
            coarse_support = coarsen_support_any(support, bottleneck.shape[-3:])
            bottleneck = self.attention(bottleneck, coarse_support)
        dec2 = self.dec2(torch.cat((self._up(bottleneck, enc2), enc2), dim=1), embedded)
        dec1 = self.dec1(torch.cat((self._up(dec2, enc1), enc1), dim=1), embedded)
        dec0 = self.dec0(torch.cat((self._up(dec1, enc0), enc0), dim=1), embedded)
        return self.output(F.silu(self.output_norm(dec0)))


class D2ConditionalFourierVDenoiser(nn.Module):
    """Predict the VP ``v`` target in the exact F3-L2 Fourier coordinates."""

    def __init__(
        self,
        *,
        condition_channels: int = 7,
        base: int = 8,
        time_channels: int = 64,
        coarse_attention: bool = False,
        attention_heads: int = 4,
    ):
        super().__init__()
        self.condition_channels = int(condition_channels)
        self.base = int(base)
        self.time_channels = int(time_channels)
        self.coarse_attention = bool(coarse_attention)
        self.net = D2ResidualUNet3D(
            input_channels=1 + self.condition_channels,
            output_channels=1,
            base=self.base,
            time_channels=self.time_channels,
            coarse_attention=self.coarse_attention,
            attention_heads=int(attention_heads),
        )

    def forward(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
        *,
        layout: FourierModeLayout,
        whitening: Mapping[str, list[float]],
        support_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if state.ndim != 2 or condition.ndim != 5 or state.shape[0] != condition.shape[0]:
            raise ValueError("D2 Fourier state and condition batches differ")
        if condition.shape[1] != self.condition_channels or tuple(condition.shape[-3:]) != layout.shape:
            raise ValueError("D2 condition changed from the frozen Fourier geometry")
        time = torch.as_tensor(time, device=state.device, dtype=state.dtype)
        if time.ndim == 0:
            time = time.repeat(state.shape[0])
        if time.shape != (state.shape[0],):
            raise ValueError("D2 time must be scalar or one value per draw")
        physical = unwhiten_components(state, whitening, layout)
        state_field = unpack_fourier_components(physical, layout)
        support = None
        if self.coarse_attention:
            if support_mask is None:
                raise ValueError("attention-enabled D2 model requires exact support_random metadata")
            if support_mask.shape != (condition.shape[0], 1, *layout.shape):
                raise ValueError("D2 exact support_random mask is not aligned")
            support = support_mask.bool()
        velocity_field = self.net(
            torch.cat((state_field, condition), dim=1), time, support=support
        )
        return whiten_velocity(pack_fourier_components(velocity_field, layout), whitening, layout)


def clone_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Detached state on the model device for synchronization-free EMA updates."""
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def update_ema_state(
    ema_state: dict[str, torch.Tensor],
    model: nn.Module,
    *,
    decay: float,
    update: int,
) -> None:
    """Update EMA with a short deterministic warm-up, including model buffers."""
    if not 0.0 <= decay < 1.0 or update <= 0:
        raise ValueError("EMA decay/update contract is invalid")
    warm_decay = min(float(decay), (1.0 + update) / (10.0 + update))
    state = model.state_dict()
    if set(state) != set(ema_state):
        raise RuntimeError("EMA/model state keys changed")
    for name, value in state.items():
        source = value.detach()
        if source.device != ema_state[name].device:
            raise RuntimeError("EMA state moved away from the model device")
        if torch.is_floating_point(source):
            ema_state[name].mul_(warm_decay).add_(source, alpha=1.0 - warm_decay)
        else:
            ema_state[name].copy_(source)


def load_model_state_copy(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    model.load_state_dict(dict(state), strict=True)


@torch.inference_mode()
def sample_fourier_d2(
    model: nn.Module,
    condition: torch.Tensor,
    *,
    layout: FourierModeLayout,
    whitening: Mapping[str, list[float]],
    draws: int,
    steps: int,
    generator: torch.Generator,
    eta: float = 0.0,
    support_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generalized DDIM sampler; ``eta=0`` deterministic, ``eta=1`` stochastic."""
    if condition.shape[0] != 1 or draws <= 0 or steps <= 0 or not 0.0 <= eta <= 1.0:
        raise ValueError("D2 sampler contract is invalid")
    condition = condition.expand(draws, -1, -1, -1, -1)
    if support_mask is not None:
        if support_mask.shape != (1, 1, *layout.shape):
            raise ValueError("D2 sampler support_random metadata is not aligned")
        support_mask = support_mask.expand(draws, -1, -1, -1, -1)
    state = torch.randn(
        (draws, layout.components),
        device=condition.device,
        dtype=condition.dtype,
        generator=generator,
    )
    for index in range(steps, 0, -1):
        time = torch.full((draws,), index / steps, device=state.device, dtype=state.dtype)
        alpha, sigma = cosine_alpha_sigma(time)
        velocity = model(
            state,
            time,
            condition,
            layout=layout,
            whitening=whitening,
            support_mask=support_mask,
        )
        estimate = alpha[:, None] * state - sigma[:, None] * velocity
        epsilon = sigma[:, None] * state + alpha[:, None] * velocity

        next_time = torch.full(
            (draws,), (index - 1) / steps, device=state.device, dtype=state.dtype
        )
        next_alpha, next_sigma = cosine_alpha_sigma(next_time)
        if eta == 0.0 or index == 1:
            state = next_alpha[:, None] * estimate + next_sigma[:, None] * epsilon
            continue

        alpha_bar = alpha.square()
        next_alpha_bar = next_alpha.square()
        stochastic_sigma = eta * torch.sqrt(
            torch.clamp(
                (1.0 - next_alpha_bar)
                / torch.clamp(1.0 - alpha_bar, min=1.0e-12)
                * (1.0 - alpha_bar / torch.clamp(next_alpha_bar, min=1.0e-12)),
                min=0.0,
            )
        )
        directional = torch.sqrt(
            torch.clamp(1.0 - next_alpha_bar - stochastic_sigma.square(), min=0.0)
        )
        fresh = torch.randn(
            state.shape, device=state.device, dtype=state.dtype, generator=generator
        )
        state = (
            next_alpha[:, None] * estimate
            + directional[:, None] * epsilon
            + stochastic_sigma[:, None] * fresh
        )
    return unpack_fourier_components(unwhiten_components(state, whitening, layout), layout)[:, 0]


@torch.inference_mode()
def sample_fourier_d2_batched(
    model: nn.Module,
    condition: torch.Tensor,
    *,
    layout: FourierModeLayout,
    whitening: Mapping[str, list[float]],
    draws: int,
    draw_batch: int,
    steps: int,
    generator: torch.Generator,
    eta: float = 0.0,
    support_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample in bounded draw batches while preserving one deterministic RNG stream."""
    if draw_batch <= 0 or draws <= 0:
        raise ValueError("D2 draw and draw-batch counts must be positive")
    pieces = []
    for start in range(0, draws, draw_batch):
        pieces.append(
            sample_fourier_d2(
                model,
                condition,
                layout=layout,
                whitening=whitening,
                draws=min(draw_batch, draws - start),
                steps=steps,
                generator=generator,
                eta=eta,
                support_mask=support_mask,
            )
        )
    return torch.cat(pieces, dim=0)


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


__all__ = [
    "CUDA_DETERMINISM_POLICY",
    "CoarseAttention3D",
    "D2ConditionalFourierVDenoiser",
    "D2ResidualUNet3D",
    "SinusoidalLogSNREmbedding",
    "TimeResidualBlock3D",
    "clone_model_state",
    "configure_d2_determinism",
    "coarsen_support_any",
    "load_model_state_copy",
    "log_snr",
    "parameter_count",
    "sample_fourier_d2",
    "sample_fourier_d2_batched",
    "update_ema_state",
]
