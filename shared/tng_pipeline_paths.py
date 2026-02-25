"""Shared path resolution helpers for active TNG/Illustris pipelines."""

from __future__ import annotations

import os
from dataclasses import dataclass

from config_paths import CANONICAL_CACHE_ROOT, CANONICAL_OUTPUT_ROOT

DEFAULT_JRAPH_CACHE_DIR = "/pscratch/sd/d/dkololgi/Cosmic_env_TNG_cache"
DEFAULT_JRAPH_OUTPUT_DIR = "/pscratch/sd/d/dkololgi/TNG_Illustris_outputs/regression/"

DEFAULT_SBI_CACHE_DIR = "/pscratch/sd/d/dkololgi/Cosmic_env_TNG_cache"
DEFAULT_SBI_OUTPUT_DIR = "/pscratch/sd/d/dkololgi/TNG_Illustris_outputs/sbi/"


@dataclass(frozen=True)
class PipelinePaths:
    cache_dir: str
    cache_path: str
    pyg_cache_path: str
    output_dir: str


def resolve_pipeline_paths(
    masscut: float,
    use_v2: bool,
    use_transformed_eig: bool,
    output_dir: str | None = None,
    cache_dir: str | None = None,
) -> PipelinePaths:
    """Resolve cache/output paths for the active Jraph pipeline."""
    version = "v2" if use_v2 else "v1"
    cache_suffix = "_transformed_eig" if use_transformed_eig else "_raw_eig"

    resolved_cache_dir = cache_dir or os.environ.get(
        "TNG_JRAPH_CACHE_DIR",
        os.environ.get("TNG_CANONICAL_CACHE_ROOT", CANONICAL_CACHE_ROOT),
    )
    resolved_output_dir = output_dir or os.environ.get(
        "TNG_JRAPH_OUTPUT_DIR",
        os.path.join(
            os.environ.get("TNG_CANONICAL_OUTPUT_ROOT", CANONICAL_OUTPUT_ROOT),
            "regression",
        ),
    )
    # Preserve current behavior unless explicitly overridden.
    if "TNG_JRAPH_OUTPUT_DIR" not in os.environ and output_dir is None:
        resolved_output_dir = DEFAULT_JRAPH_OUTPUT_DIR
    if "TNG_JRAPH_CACHE_DIR" not in os.environ and cache_dir is None:
        resolved_cache_dir = DEFAULT_JRAPH_CACHE_DIR

    cache_path = (
        f"{resolved_cache_dir}/processed_jraph_data_mc{masscut:.0e}_{version}_scaled_3{cache_suffix}.pkl"
    )
    pyg_cache_path = f"{resolved_cache_dir}/processed_gcn_data_mc{masscut:.0e}.pt"
    return PipelinePaths(
        cache_dir=resolved_cache_dir,
        cache_path=cache_path,
        pyg_cache_path=pyg_cache_path,
        output_dir=resolved_output_dir,
    )


@dataclass(frozen=True)
class SBIPaths:
    cache_dir: str
    data_path: str
    output_dir: str


def resolve_sbi_paths(
    use_transformed_eig: bool,
    output_dir: str | None = None,
    cache_dir: str | None = None,
) -> SBIPaths:
    """Resolve cache and output paths for the FlowJAX SBI pipeline."""
    resolved_cache_dir = cache_dir or os.environ.get(
        "TNG_SBI_CACHE_DIR",
        os.environ.get("TNG_CANONICAL_CACHE_ROOT", CANONICAL_CACHE_ROOT),
    )
    resolved_output_dir = output_dir or os.environ.get(
        "TNG_SBI_OUTPUT_DIR",
        os.path.join(
            os.environ.get("TNG_CANONICAL_OUTPUT_ROOT", CANONICAL_OUTPUT_ROOT),
            "sbi",
        ),
    )
    # Preserve current behavior unless explicitly overridden.
    if "TNG_SBI_CACHE_DIR" not in os.environ and cache_dir is None:
        resolved_cache_dir = DEFAULT_SBI_CACHE_DIR
    if "TNG_SBI_OUTPUT_DIR" not in os.environ and output_dir is None:
        resolved_output_dir = DEFAULT_SBI_OUTPUT_DIR

    suffix = "_transformed_eig" if use_transformed_eig else "_raw_eig"
    data_path = f"{resolved_cache_dir}/processed_jraph_data_mc1e+09_v2_scaled_3{suffix}.pkl"
    return SBIPaths(
        cache_dir=resolved_cache_dir,
        data_path=data_path,
        output_dir=resolved_output_dir,
    )
