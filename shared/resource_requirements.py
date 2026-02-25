"""Runtime resource guards for HPC workflow entrypoints.

These checks prevent accidentally launching heavy workflows on the wrong
resource type (e.g., login node instead of GPU/CPU SLURM allocations).
"""

from __future__ import annotations

import os


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _slurm_tasks() -> int:
    ntasks = _parse_int(os.environ.get("SLURM_NTASKS"))
    if ntasks is not None:
        return ntasks
    nodes = _parse_int(os.environ.get("SLURM_JOB_NUM_NODES")) or 1
    ppn = _parse_int(os.environ.get("SLURM_NTASKS_PER_NODE")) or 1
    return nodes * ppn


def _visible_gpus() -> int:
    # Preferred in SLURM jobs if set.
    for key in ("SLURM_GPUS_ON_NODE", "SLURM_GPUS", "SLURM_JOB_GPUS"):
        raw = os.environ.get(key)
        if raw:
            # Could be "4", "gpu:4", or comma-separated device ids.
            if raw.isdigit():
                return int(raw)
            if ":" in raw:
                tail = raw.split(":")[-1]
                if tail.isdigit():
                    return int(tail)
            return len([x for x in raw.split(",") if x.strip()])

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cuda_visible or cuda_visible in {"-1", "NoDevFiles"}:
        return 0
    return len([x for x in cuda_visible.split(",") if x.strip()])


def _guards_disabled() -> bool:
    return os.environ.get("TNG_SKIP_RESOURCE_GUARDS", "0") == "1"


def require_gpu_slurm(script_name: str, min_gpus: int = 1) -> None:
    """Require a SLURM GPU allocation before running heavy GPU workflows."""
    if _guards_disabled():
        return
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError(
            f"{script_name} must run inside a SLURM GPU allocation. "
            f"Submit via the matching SLURM script under workflows/."
        )
    n_gpus = _visible_gpus()
    if n_gpus < min_gpus:
        raise RuntimeError(
            f"{script_name} requires at least {min_gpus} visible GPU(s); detected {n_gpus}. "
            f"Use a GPU node/allocation (e.g. --constraint=gpu)."
        )


def require_cpu_mpi_slurm(script_name: str, min_tasks: int = 2) -> None:
    """Require a SLURM CPU MPI allocation before running heavy MPI workflows."""
    if _guards_disabled():
        return
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError(
            f"{script_name} must run inside a SLURM CPU allocation. "
            f"Submit via workflows/abacus_tweb/submit_abacus_tweb_cpu.slurm."
        )
    ntasks = _slurm_tasks()
    if ntasks < min_tasks:
        raise RuntimeError(
            f"{script_name} requires MPI with at least {min_tasks} task(s); detected {ntasks}. "
            f"Request more tasks/nodes in your SLURM job."
        )
    if _visible_gpus() > 0:
        raise RuntimeError(
            f"{script_name} is a CPU MPI workflow, but GPUs were detected in this allocation. "
            f"Run on a CPU constraint job (e.g. --constraint=cpu)."
        )
