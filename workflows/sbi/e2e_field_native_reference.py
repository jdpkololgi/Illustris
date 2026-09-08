#!/usr/bin/env python3
"""Independent full-box potential and native-stencil tensor references for E2E.

Only the registered training phase is audited. A full native FFT supplies the
potential; tensors are sampled with the native centred-first-derivative twice,
avoiding six full-volume Hessian allocations. Frozen eigenvalues are an
independent replay check, not the input to tensor reconstruction.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import struct
import time
import zipfile

import h5py
import numpy as np
from scipy import fft

from workflows.sbi.e2e_field_build_products import (
    DEFAULT_CONFIG, TENSOR_COMPONENTS, context_slice, eigs, native_indices,
    read_json, require_compute, sha256, tensor_from_delta, validate_config, write_json,
)
from workflows.sbi.e2e_field_prepare_data import safe_path


def stored_npz_array(path, name):
    """Read-only mapping of an uncompressed NPY member; no slab-sized copy."""
    path = safe_path(Path(path))
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(name + ".npy")
        if info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 1:
            raise ValueError("native reference requires an uncompressed, unencrypted member")
        with path.open("rb") as f:
            f.seek(info.header_offset)
            header = f.read(30)
            if header[:4] != b"PK\x03\x04":
                raise ValueError("invalid ZIP local header")
            filename_bytes, extra_bytes = struct.unpack_from("<HH", header, 26)
            offset = info.header_offset + 30 + filename_bytes + extra_bytes
            f.seek(offset)
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
            else:
                raise ValueError("unsupported native NPY format")
            if dtype.hasobject or f.tell() - offset + np.prod(shape)*dtype.itemsize != info.file_size:
                raise ValueError("native member size/type mismatch")
            return np.memmap(path, mode="r", dtype=dtype, shape=shape,
                             offset=f.tell(), order="F" if fortran else "C")


def sample_native_tensor(potential, axes, cell):
    """Equivalent to two periodic centred first differences (native FIESTA)."""
    n = potential.shape[0]
    result = []
    for a, b in TENSOR_COMPONENTS:
        value = np.zeros(tuple(len(x) for x in axes), dtype=np.float64)
        for sa, sb in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            ix = [x.copy() for x in axes]
            ix[a] += sa
            ix[b] += sb
            value += sa * sb * potential[np.ix_(*(x % n for x in ix))]
        result.append(value / (4*cell*cell))
    return np.stack(result, axis=-1)


def reference_eigenvalues(slabs, axes):
    result = np.full(tuple(len(x) for x in axes) + (3,), np.nan, dtype=np.float32)
    used = []
    for slab in slabs:
        selected = np.flatnonzero((axes[0] >= slab["x_start"]) & (axes[0] < slab["x_end"]))
        if not len(selected):
            continue
        values = stored_npz_array(slab["path"], "eig_vals")
        indices = np.ix_(axes[0][selected] - slab["x_start"], axes[1], axes[2])
        for i in range(3):
            result[selected, ..., i] = values[i][indices]
        used.append(slab["path"])
        del values
    if not np.isfinite(result).all():
        raise ValueError("native eigenvalue reference has missing slab coverage")
    return result, used


def potential_from_counts(path, box, smoothing, workers=32):
    started = time.monotonic()
    counts = np.load(safe_path(Path(path)), mmap_mode="r", allow_pickle=False)
    n = counts.shape[0]
    if counts.shape != (n, n, n) or counts.dtype != np.float32:
        raise ValueError("wrong native counts shape/dtype")
    mean = float(np.mean(counts, dtype=np.float64))
    if not mean > 0:
        raise ValueError("invalid native count mean")
    print(f"native mean {mean:.12g}; constructing {n}^3 potential", flush=True)
    delta = np.empty(counts.shape, dtype=np.float64)
    for x in range(0, n, 16):
        slab = np.asarray(counts[x:x+16])
        if not np.isfinite(slab).all() or np.min(slab) < 0:
            raise ValueError("invalid source counts")
        delta[x:x+16] = slab / mean - 1
    del counts
    spectrum = fft.rfftn(delta, workers=workers, overwrite_x=True)
    del delta
    gc.collect()
    k = 2*np.pi*fft.fftfreq(n, d=box/n)
    kz = 2*np.pi*fft.rfftfreq(n, d=box/n)
    ky2z2 = k[:, None]**2 + kz[None, :]**2
    for x in range(n):
        k2 = k[x]**2 + ky2z2
        if x == 0:
            k2[0, 0] = 1
        spectrum[x] *= -np.exp(-0.5*smoothing*smoothing*k2) / k2
    spectrum[0, 0, 0] = 0
    potential = fft.irfftn(spectrum, s=(n,n,n), workers=workers, overwrite_x=True)
    del spectrum
    gc.collect()
    print(f"native potential ready in {time.monotonic()-started:.1f}s", flush=True)
    return potential, mean


def metrics(pred, truth, mask):
    p, t = pred[mask], truth[mask]
    residual = p-t
    sigma = np.std(t, axis=0)
    return {"n": len(p), "rmse": np.sqrt(np.mean(residual**2, axis=0)).tolist(),
            "bias": np.mean(residual, axis=0).tolist(),
            "truth_std": sigma.tolist(),
            "rmse_over_truth_std": (np.sqrt(np.mean(residual**2, axis=0))/sigma).tolist(),
            "bias_over_truth_std": (np.mean(residual, axis=0)/sigma).tolist(),
            "two_collapsed_fraction_difference": float(np.mean(p[:,1] > .2)-np.mean(t[:,1] > .2)),
            "two_collapsed_label_disagreement": float(np.mean((p[:,1] > .2)!=(t[:,1] > .2)))}


def audit(c, config_path):
    root = Path(c["output_root"])
    screened = read_json(root / "SCREENED_PARENTS.json")
    if screened["config_sha256"] != sha256(config_path):
        raise ValueError("screening config mismatch")
    out = root / "native_audit"
    out.mkdir(exist_ok=False)
    phase = c["native_reference_audit_phase"]
    # Fixed one anchor per shell/cap, favouring interior for a domain feasibility
    # check. Selection is entirely response/geometry based, before truth reads.
    eligible = [a for a in screened["anchors"] if a["phase"] == phase]
    chosen = []
    for cap in ("NGC", "SGC"):
        for shell in range(4):
            candidates = [a for a in eligible if a["cap"] == cap and a["shell"] == shell]
            candidates.sort(key=lambda a: (a["support_stratum"] != "interior", a["anchor_id"]))
            if candidates:
                chosen.append(candidates[0])
    chosen = chosen[:c["native_reference_max_anchors"]]
    if len(chosen) != c["native_reference_max_anchors"]:
        raise ValueError("insufficient training-phase reference anchors")
    write_json(out / "REFERENCE_SELECTION.json", {"anchor_ids": [a["anchor_id"] for a in chosen],
                                                "truth_used": False, "config_sha256": sha256(config_path)})
    source = next(s for s in screened["sources"] if s["phase"] == phase)
    marker = read_json(source["target_metadata"]["path"])
    inputs = marker["inputs"]
    slabs = inputs["tweb_rank_files"]
    density_dir = Path(slabs[0]["path"]).parents[2] / "density"
    density_path = density_dir / f"AbacusSummit_base_c000_{phase}_z0.200_ngrid2048_ab10_tsc_counts.npy"
    density_manifest = density_path.with_suffix(".manifest.json")
    dm = read_json(density_manifest)
    if dm["phase"] != phase:
        raise ValueError("density phase mismatch")
    density_hash = sha256(density_path)
    expected_hash = dm.get("legacy_source", {}).get("sha256")
    if expected_hash and density_hash != expected_hash:
        raise ValueError("native density checksum mismatch")
    potential, mean = potential_from_counts(density_path, c["box_mpc_h"], c["native_smoothing_mpc_h"])
    report = {"schema_version": "e2e-native-reference-audit-v1", "phase": phase,
              "config_sha256": sha256(config_path), "builder_sha256": sha256(__file__),
              "job_id": os.environ.get("SLURM_JOB_ID"), "density_path": str(density_path),
              "density_sha256": density_hash, "density_mean": mean,
              "derivative": "two centred periodic first differences of Gaussian-smoothed full-box potential",
              "tensor_order": [list(x) for x in TENSOR_COMPONENTS], "parents": [],
              "science_error_budget_frozen": False, "r0_physics_pass": False,
              "training_ready": False}
    used_slabs = set()
    tol = c["technical_tolerances"]
    for anchor in chosen:
        start = [x-c["science_core_side"]//2 for x in anchor["center"]]
        axes = native_indices(anchor["grid"], start, c["science_core_side"], c)
        tensor = sample_native_tensor(potential, axes, c["box_mpc_h"]/c["native_ngrid"])
        native_eigs, used = reference_eigenvalues(slabs, axes)
        used_slabs.update(used)
        replay = float(np.max(np.abs(eigs(tensor)-native_eigs)))
        if replay > tol["native_eigenvalue_max_abs"]:
            raise ValueError(f"independent native tensor replay failed: {anchor['anchor_id']} {replay}")
        s = next(s for s in screened["sources"] if s["phase"] == phase and s["cap"] == anchor["cap"])
        reference_path = out / (anchor["anchor_id"] + ".npz")
        with reference_path.open("xb") as f:
            np.savez_compressed(f, tensor_native=tensor, eigenvalues_native=native_eigs,
                                native_ix=axes[0], native_iy=axes[1], native_iz=axes[2])
        with h5py.File(s["target_file"]["path"], "r") as target, h5py.File(s["response_file"]["path"], "r") as response:
            core_slice = context_slice(anchor["center"], c["science_core_side"])
            source_core = target["delta_r7"][core_slice]
            # Match the sequential float32 sum used by the frozen target builder.
            trace = (native_eigs[...,0]+native_eigs[...,1])+native_eigs[...,2]
            trace_error = float(np.max(np.abs(source_core-trace)))
            if trace_error > tol["source_trace_max_abs"]:
                raise ValueError("native coordinate/trace replay failed")
            mask = response["support_random"][core_slice].astype(bool)
            row = {"anchor_id": anchor["anchor_id"], "reference_path": str(reference_path),
                   "reference_sha256": sha256(reference_path), "native_eigenvalue_max_abs": replay,
                   "source_trace_max_abs": trace_error, "domains": {}}
            for side in c["parent_sides"]:
                delta = target["delta_r7"][context_slice(anchor["center"], side)]
                t = tensor_from_delta(delta, c["cell_mpc"]*c["coordinate_h"])
                local_core = context_slice([side//2]*3, c["science_core_side"])
                prediction = eigs(t[local_core])
                row["domains"][str(side)] = {"parent_mean": float(delta.mean(dtype=np.float64)),
                                              "periodic_mean_completion": metrics(prediction, native_eigs, mask),
                                              "tensor_trace_max_abs": float(np.max(np.abs(t[...,[0,3,5]].sum(axis=-1)-delta)))}
        report["parents"].append(row)
        print(f"reference {anchor['anchor_id']}: native replay {replay:.3g}, trace {trace_error:.3g}", flush=True)
    del potential
    gc.collect()
    report["native_slabs"] = [{"path": p, "sha256": sha256(p)} for p in sorted(used_slabs)]
    report["native_tensor_replay_pass"] = True
    report["source_trace_coordinate_pass"] = True
    report["status"] = "independent_reference_complete_domain_adequacy_requires_scientific_decision"
    write_json(out / "NATIVE_REFERENCE_AUDIT.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = p.parse_args()
    c = read_json(args.config)
    validate_config(c)
    require_compute()
    audit(c, args.config)


if __name__ == "__main__":
    main()
