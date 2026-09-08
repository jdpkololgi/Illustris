#!/usr/bin/env python3
"""Compute-node E2E preparation with explicit, fail-closed readiness stages.

Screening uses random support and geometry only. Phase-level roles prevent any
source-box alias crossing data roles. Confirmation targets are packaging inputs,
never used to choose geometry, normalization, physics conventions or a model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import socket
import time

import h5py
import numpy as np
from astropy.cosmology import Planck18
from scipy import fft

from workflows.sbi.e2e_field_prepare_data import REPO, safe_path, periodic_intervals, footprints_overlap

DEFAULT_CONFIG = REPO / "configs/e2e_field_build_v1.json"
ALLOWED_ROOT = Path("/pscratch/sd/d/dkololgi/abacus/e2e_field_v1")
PHASE_ROLES = {"ph000": "train", "ph002": "train", "ph003": "train",
               "ph004": "internal_selection", "ph005": "internal_confirmation"}
TENSOR_COMPONENTS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def sha256(path):
    h = hashlib.sha256()
    with safe_path(Path(path)).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(safe_path(Path(path)).read_text())


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def require_compute():
    if not os.environ.get("SLURM_JOB_ID") or not socket.gethostname().startswith("nid"):
        raise RuntimeError("E2E array processing requires a Slurm compute node")


def validate_config(c):
    if c.get("schema_version") != "e2e-field-build-v1" or c["phase_roles"] != PHASE_ROLES:
        raise ValueError("unexpected schema or phase-role change")
    if c["parent_sides"] != [64, 96] or c["science_core_side"] != 32:
        raise ValueError("geometry is not the registered two-domain comparison")
    if c["coordinate_h"] != 0.6766 or c["cell_mpc"] != 5.0:
        raise ValueError("observer units changed")
    if c["science_training_authorized"] or c["science_error_budget_frozen"]:
        raise ValueError("this preparation entrypoint cannot authorize science training")
    if c["native_reference_audit_phase"] != "ph000":
        raise ValueError("physics audit must use the fixed training phase")
    if not 1 <= c["candidate_centers_per_cap"] <= 8192:
        raise ValueError("unbounded geometry screening")
    root = safe_path(Path(c["output_root"])).resolve()
    if not root.is_relative_to(ALLOWED_ROOT.resolve()) or root == ALLOWED_ROOT.resolve():
        raise ValueError("output is not a child of the E2E Scratch namespace")


def context_slice(center, side):
    return tuple(slice(int(x) - side // 2, int(x) + side // 2) for x in center)


def native_indices(grid, start, side, c):
    return [np.floor(np.mod((grid["origin_mpc"][a] +
                            (np.arange(start[a], start[a] + side) + 0.5) * c["cell_mpc"])
                           * c["coordinate_h"] + c["box_offset_mpc_h"], c["box_mpc_h"])
                     / (c["box_mpc_h"] / c["native_ngrid"])).astype(np.int64)
            for a in range(3)]


def tensor_from_delta(delta, cell, dc=True):
    """Declared periodic spectral comparison, not a physical survey boundary model."""
    n = delta.shape[0]
    if delta.shape != (n, n, n):
        raise ValueError("expected cubic density")
    k = [2 * np.pi * fft.fftfreq(n, d=cell),
         2 * np.pi * fft.fftfreq(n, d=cell),
         2 * np.pi * fft.rfftfreq(n, d=cell)]
    axes = np.meshgrid(*k, indexing="ij", sparse=True)
    k2 = sum(x*x for x in axes)
    k2[0, 0, 0] = 1.0
    spectrum = fft.rfftn(np.asarray(delta, dtype=np.float64))
    out = []
    for a, b in TENSOR_COMPONENTS:
        multiplier = axes[a] * axes[b] / k2
        multiplier[0, 0, 0] = 1 / 3 if dc and a == b else 0
        out.append(fft.irfftn(spectrum * multiplier, s=delta.shape))
    return np.stack(out, axis=-1)


def eigs(t):
    full = np.empty(t.shape[:-1] + (3, 3), dtype=t.dtype)
    for i, (a, b) in enumerate(TENSOR_COMPONENTS):
        full[..., a, b] = full[..., b, a] = t[..., i]
    return np.linalg.eigvalsh(full)


def screen(c, config_path):
    root = Path(c["output_root"])
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    inventory = read_json(REPO / c["metadata_preparation"] / "source_inventory.json")
    anchors, sources, screening = [], [], []
    z_grid = np.linspace(0, 0.8, 4001)
    r_grid = Planck18.comoving_distance(z_grid).value
    for source in inventory["sources"]:
        phase, cap = source["phase"], source["cap"]
        if phase not in PHASE_ROLES:
            continue  # no legacy/ph001 payload access
        grid = source["grid"]
        for key in ("target_metadata", "adapter_metadata", "response_metadata"):
            if sha256(source[key]["path"]) != source[key]["sha256"]:
                raise ValueError(f"source metadata drift: {phase}/{cap}/{key}")
        response_path = safe_path(Path(source["response_file"]["path"]))
        with h5py.File(response_path, "r") as f:
            support = f["support_random"][:].astype(bool)
            if list(support.shape) != grid["shape"]:
                raise ValueError("response grid mismatch")
            virtual_sources = []
            for name in ("counts", "los_x", "los_y", "los_z"):
                for v in f[name].virtual_sources():
                    p = safe_path(Path(v.file_name))
                    if not p.is_file() or phase not in str(p):
                        raise ValueError("unavailable or wrong-phase virtual source")
                    virtual_sources.append({"path": str(p), "dataset": v.dset_name})
            seed = int(hashlib.sha256(f"{c['seed']}:{phase}:{cap}".encode()).hexdigest()[:16], 16)
            rng = np.random.default_rng(seed)
            margin = max(c["parent_sides"]) // 2 + c["observation_halo"]
            # Uniform aligned lattice proposals; acceptance sees no density/count values.
            centers = np.column_stack([rng.integers(margin // 8, (n - margin) // 8 + 1,
                                        size=c["candidate_centers_per_cap"]) * 8 for n in support.shape])
            seen, counts = set(), {}
            reasons = {"duplicate": 0, "outside_shell": 0, "low_support": 0, "stratum_full": 0}
            for center in centers:
                key = tuple(int(x) for x in center)
                if key in seen:
                    reasons["duplicate"] += 1
                    continue
                seen.add(key)
                xyz = np.asarray(grid["origin_mpc"]) + (center + 0.5) * c["cell_mpc"]
                redshift = float(np.interp(np.linalg.norm(xyz), r_grid, z_grid))
                shell = int(np.searchsorted(c["shell_edges"], redshift, side="right") - 1)
                if not 0 <= shell < 4:
                    reasons["outside_shell"] += 1
                    continue
                frac = float(support[context_slice(center, c["science_core_side"])].mean())
                if frac < c["core_support_fraction_min"]:
                    reasons["low_support"] += 1
                    continue
                stratum = "interior" if frac >= c["interior_support_fraction_min"] else "boundary"
                slot = (shell, stratum)
                ordinal = counts.get(slot, 0)
                if ordinal >= c["anchors_per_cap_shell_support_stratum"]:
                    reasons["stratum_full"] += 1
                    continue
                counts[slot] = ordinal + 1
                anchors.append({"anchor_id": f"{phase}_{cap}_s{shell}_{stratum}_{ordinal:02d}",
                                "phase": phase, "cap": cap, "role": PHASE_ROLES[phase],
                                "center": center.tolist(), "grid": grid, "shell": shell,
                                "redshift": redshift, "support_stratum": stratum,
                                "science_core_support_fraction": frac})
            del support
        source = dict(source, role=PHASE_ROLES[phase], virtual_sources=virtual_sources)
        sources.append(source)
        screening.append({"phase": phase, "cap": cap, "counts": {f"{s}:{t}": n for (s,t),n in counts.items()},
                          "rejected": reasons, "candidate_count": len(centers)})
        print(f"screen {phase}/{cap}: {sum(counts.values())} anchors", flush=True)
    # Full phase roles are deliberately conservative: every alias in a phase has one role.
    pairs = []
    radius = max(c["parent_sides"]) // 2 + c["observation_halo"]
    for a in anchors:
        lo = (np.asarray(a["grid"]["origin_mpc"]) + (np.array(a["center"]) - radius) * c["cell_mpc"]) * c["coordinate_h"] + c["box_offset_mpc_h"]
        width = 2 * radius * c["cell_mpc"] * c["coordinate_h"]
        a["source_box_footprint"] = [periodic_intervals(x, x+width, c["box_mpc_h"]) for x in lo]
    for i, a in enumerate(anchors):
        for b in anchors[i+1:]:
            if a["phase"] == b["phase"] and footprints_overlap(a["source_box_footprint"], b["source_box_footprint"]):
                if a["role"] != b["role"]:
                    raise ValueError("source-box aliases cross roles")
                pairs.append([a["anchor_id"], b["anchor_id"]])
    payload = {"schema_version": "e2e-screened-parents-v1", "config_sha256": sha256(config_path),
               "builder_sha256": sha256(__file__), "created_utc": datetime.now(timezone.utc).isoformat(),
               "job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(),
               "phase_roles": PHASE_ROLES, "anchors": anchors, "sources": sources,
               "screening": screening, "overlapping_anchor_pairs": pairs,
               "role_grouping": "entire simulation phase including all caps and source-box replicas",
               "target_values_used_for_selection": False, "training_ready": False,
               "external_confirmation": None, "elapsed_seconds": time.monotonic()-started}
    write_json(root / "SCREENED_PARENTS.json", payload)
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("screen",))
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = p.parse_args()
    c = read_json(args.config)
    validate_config(c)
    require_compute()
    if args.stage == "screen":
        screen(c, args.config)


if __name__ == "__main__":
    main()
