#!/usr/bin/env python3
"""Build training-only wide-coarse/local-fine products, then truth-only audit.

The full-box low tensor is an attribution oracle, NEVER an inference input.
No learned fit, selection/confirmation payload, or automatic training release.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import gc
import os
from pathlib import Path
import socket
import subprocess
import time

import h5py
import numpy as np
from scipy import fft

from workflows.sbi.e2e_field_build_products import (
    TENSOR_COMPONENTS, context_slice, eigs, read_json, require_compute, sha256,
    tensor_from_delta, validate_config, write_json,
)
from workflows.sbi.e2e_field_domain_gate import describe, failures, gate_values
from workflows.sbi.e2e_field_error_budget import (
    inverse_component, native_coordinates, science, smoothed_spectrum, tensor_metrics,
)
from workflows.sbi.e2e_field_prepare_data import REPO, safe_path
from workflows.sbi.e2e_field_regenerate_spectral import interpolate

DEFAULT_CONFIG = REPO / "configs/e2e_field_wide_coarse_v1.json"
PHASES = ["ph000", "ph002", "ph003"]
SUM_CHANNELS = ("counts", "expected_counts_random")
MEAN_CHANNELS = ("support_random", "angular_response", "exposure_apodized_random", "ntilde_mpc3")


def compact_lowpass(spec, box, nout, cutoff):
    """Exact retained Fourier coefficients, with backward-FFT normalization.

    The cutoff lies strictly below the output Nyquist: no Nyquist merging or
    self-conjugate aliasing. Only interpolation of this bandlimited field is
    approximate, not the Fourier truncation itself.
    """
    n = spec.shape[0]
    if nout % 2 or nout > n or not 0 < cutoff < np.pi*nout/box:
        raise ValueError("invalid compact Fourier grid/cutoff")
    inds = np.r_[np.arange(nout//2), np.arange(n-nout//2, n)]
    out = spec[np.ix_(inds, inds, np.arange(nout//2+1))].copy()
    k = 2*np.pi*fft.fftfreq(nout, d=box/nout)
    kz = 2*np.pi*fft.rfftfreq(nout, d=box/nout)
    for i in range(nout):
        out[i] *= (k[i]**2+k[:, None]**2+kz[None, :]**2 <= cutoff**2)
    out *= (nout/n)**3
    return out


def coarse_coords(row, variant, build, ngrid):
    span, factor = variant["span_fine_cells"], variant["factor"]
    return [np.mod((row["grid"]["origin_mpc"][a] +
                    (row["center"][a]-span/2+(np.arange(span//factor)+.5)*factor)*build["cell_mpc"])
                   *build["coordinate_h"]+build["box_offset_mpc_h"], build["box_mpc_h"])
            /(build["box_mpc_h"]/ngrid) for a in range(3)]


def local_coords(variant, side=96):
    span, factor = variant["span_fine_cells"], variant["factor"]
    x = (span/2-side/2+np.arange(side)+.5)/factor-.5
    if np.floor(x.min())-2 < 0 or np.floor(x.max())+3 >= span//factor:
        raise ValueError("local interpolation must not wrap the coarse boundary")
    return [x, x, x]


def block_sum(array, factor):
    """Conservative sums with explicit zero padding of incomplete cap cells."""
    a = np.asarray(array, dtype=np.float64)
    a = np.pad(a, [(0, (-n) % factor) for n in a.shape])
    n = a.shape
    return a.reshape(n[0]//factor, factor, n[1]//factor, factor,
                     n[2]//factor, factor).sum(axis=(1, 3, 5))


def padded_extract(dataset, start, side):
    """No observational periodic wrapping. Outside cap coverage is missing."""
    start = np.asarray(start, dtype=int)
    lo = np.maximum(start, 0)
    hi = np.minimum(start+side, dataset.shape)
    out = np.zeros((side,)*3, dtype=np.float32)
    if np.all(hi > lo):
        out[tuple(slice(a, b) for a, b in zip(lo-start, hi-start))] = dataset[
            tuple(slice(a, b) for a, b in zip(lo, hi))]
    return out


def response_build(source, out, verified):
    """Coarsen each cap once; parent windows reference these shared raw arrays."""
    paths = {source["response_file"]["path"], *[v["path"] for v in source["virtual_sources"]]}
    for path in sorted(paths):
        if sha256(path) != verified[path]:
            raise ValueError("response or virtual-source checksum mismatch")
    dest = out/f"{source['phase']}_{source['cap']}_response.h5"
    with h5py.File(source["response_file"]["path"], "r") as src, h5py.File(dest, "x") as dst:
        shape = tuple(source["grid"]["shape"])
        dst.attrs["outside_cap_semantics"] = "unknown observation; never periodic; consult geometry_valid_fraction"
        dst.attrs["normalization"] = "raw; no wide-context normalizer fitted"
        for factor in (2, 4):
            g = dst.create_group(f"factor{factor}")
            target_shape = tuple((n+factor-1)//factor for n in shape)
            for name in (*SUM_CHANNELS, *MEAN_CHANNELS, "geometry_valid_fraction", "log_count_ratio_random"):
                g.create_dataset(name, shape=target_shape, dtype="f4", chunks=True, compression="lzf")
            totals = {name: [0., 0.] for name in SUM_CHANNELS}
            for i in range(0, shape[0], 16):
                sl = slice(i, min(i+16, shape[0]))
                output_sl = slice(i//factor, (sl.stop+factor-1)//factor)
                for name in (*SUM_CHANNELS, *MEAN_CHANNELS):
                    x = src[name][sl]
                    if x.shape != (sl.stop-i, *shape[1:]) or not np.isfinite(x).all():
                        raise ValueError("invalid response shape/values")
                    y = block_sum(x, factor)
                    if name in MEAN_CHANNELS:
                        y /= factor**3
                    else:
                        totals[name][0] += float(np.sum(x, dtype=np.float64))
                        totals[name][1] += float(np.sum(y.astype(np.float32), dtype=np.float64))
                    g[name][output_sl] = y
                valid = np.ones((sl.stop-i, *shape[1:]), dtype=np.float32)
                g["geometry_valid_fraction"][output_sl] = block_sum(valid, factor)/factor**3
                g["log_count_ratio_random"][output_sl] = np.log(
                    (g["counts"][output_sl]+.5)/(g["expected_counts_random"][output_sl]+.5))
            for name, (before, after) in totals.items():
                if abs(before-after) > 2e-7*max(abs(before), 1.):
                    raise ValueError("coarse response count conservation failed")
                g.attrs[f"{name}_sum_input"] = before
                g.attrs[f"{name}_sum_output"] = after
    print(f"response complete {source['phase']}/{source['cap']}", flush=True)
    return {"path": str(dest), "sha256": sha256(dest), "sources": {p: verified[p] for p in sorted(paths)}}


def load_wide_parent(product_root, row, variant):
    """Raw research reader. No oracle returned; missing cap context never wraps."""
    if row["role"] != "train" or row["phase"] not in PHASES:
        raise PermissionError("prototype reader is restricted to training phases")
    with h5py.File(safe_path(Path(product_root))/f"{row['phase']}_fields.h5", "r") as f:
        g = f[row["anchor_id"]][variant["name"]]
        target = {name: g[name][:] for name in ("coarse_delta", "fine_residual")}
        start, side = g.attrs["response_start"], int(g.attrs["response_side"])
        with h5py.File(safe_path(Path(g.attrs["response_file"])), "r") as response:
            channels = {name: padded_extract(ds, start, side) for name, ds in response[g.attrs["response_group"]].items()}
    positions = [row["grid"]["origin_mpc"][a]+(np.arange(side)+start[a]+.5)*variant["factor"]*row["grid"]["cell_mpc"] for a in range(3)]
    xyz = np.meshgrid(*positions, indexing="ij", sparse=True)
    radius = np.sqrt(sum(x*x for x in xyz))
    for name, x in zip(("los_x", "los_y", "los_z"), xyz):
        channels[name] = np.asarray(x/np.maximum(radius, 1e-30), dtype=np.float32)
    channels["observer_radius_mpc"] = radius.astype(np.float32)
    return {**target, "condition_raw": channels, "local_source_shard": row["shard"],
            "local_source_group": row["group"], "training_ready": False}


def source_hashes():
    names = ["e2e_field_wide_coarse.py", "e2e_field_build_products.py", "e2e_field_domain_gate.py",
             "e2e_field_error_budget.py", "e2e_field_regenerate_spectral.py", "e2e_field_dataset.py",
             "e2e_field_prepare_data.py"]
    return {str(Path(__file__).with_name(n).relative_to(REPO)): sha256(Path(__file__).with_name(n)) for n in names}


def validate_inputs(config_path):
    c = read_json(config_path)
    if (c["schema_version"] != "e2e-wide-coarse-v1" or c["phases"] != PHASES
        or c["fine_side"] != 96 or c["science_side"] != 32 or c["training_ready"] or c["r0_physics_pass"]
        or c["interpolation_degree"] != 5 or c["compact_fullbox_ngrid"] != 512
        or c["lowpass_kmax_h_mpc"] != .08
        or c["variants"] != [{"name": "wide192_f4", "span_fine_cells": 192, "factor": 4},
                             {"name": "wide384_f4", "span_fine_cells": 384, "factor": 4},
                             {"name": "wide384_f2", "span_fine_cells": 384, "factor": 2}]):
        raise ValueError("unregistered wide-coarse contract")
    b = read_json(REPO/c["build_config"])
    validate_config(b)
    root = safe_path(Path(c["dataset_root"]))
    if sha256(root/"DATASET_INDEX.json") != c["dataset_index_sha256"]:
        raise ValueError("source spectral index drift")
    index = read_json(root/"DATASET_INDEX.json")
    rows = [r for r in index["parents"] if r["phase"] in PHASES]
    if (len(rows) != 96 or len({r["anchor_id"] for r in rows}) != 96
        or any(r["role"] != "train" or any(x % 4 for x in r["center"]) for r in rows)
        or any(sum(r["phase"] == p for r in rows) != 32 for p in PHASES)):
        raise ValueError("requires original 96 training anchors aligned to coarse lattice")
    screen_path = Path(b["output_root"])/"SCREENED_PARENTS.json"
    if sha256(screen_path) != index["screened_manifest_sha256"]:
        raise ValueError("screen geometry drift")
    screen = read_json(screen_path)
    native = read_json(Path(b["output_root"])/"native_truth/NATIVE_TRUTH_COMPLETE.json")
    if native["dataset_index_sha256"] != index["registration"]["source_index_sha256"]:
        raise ValueError("native source provenance drift")
    out = safe_path(Path(c["output_root"])).resolve()
    allowed = Path("/pscratch/sd/d/dkololgi/abacus/e2e_field_v2").resolve()
    if not out.is_relative_to(allowed) or out == allowed or out == root.resolve():
        raise ValueError("requires separate child output")
    return c, b, index, rows, screen, native, out


def build_phase(phase, c, b, index, rows, screen, native, out):
    phase_rows = [r for r in rows if r["phase"] == phase]
    density = next(p for p in native["phases"] if p["phase"] == phase)
    if sha256(density["density_path"]) != density["density_sha256"]:
        raise ValueError("native density checksum mismatch")
    for path in {r["shard"] for r in phase_rows}:
        if sha256(path) != next(s["sha256"] for s in index["shards"] if s["path"] == path):
            raise ValueError("training shard checksum mismatch")
    verified = {s["path"]: s["sha256"] for s in index["sources_verified"]}
    responses = [response_build(s, out, verified) for s in screen["sources"] if s["phase"] == phase]
    print(f"{phase}: native R7 Fourier projection starting", flush=True)
    full, mean = smoothed_spectrum(density["density_path"], b["box_mpc_h"], 7.)
    spec = compact_lowpass(full, b["box_mpc_h"], c["compact_fullbox_ngrid"], c["lowpass_kmax_h_mpc"])
    del full
    gc.collect()
    path = out/f"{phase}_fields.h5"
    checks = []
    # Independent sparse Fourier sum checks off-grid interpolation directly.
    nz = np.nonzero(spec)
    freq = fft.fftfreq(spec.shape[0])*spec.shape[0]
    modes = np.stack([freq[nz[0]], freq[nz[1]], nz[2]], axis=-1)
    coefficients = spec[nz]*np.where(nz[2] == 0, 1., 2.)/spec.shape[0]**3
    with h5py.File(path, "x") as dst:
        dst.attrs.update({"role": "train", "training_ready": False,
                          "oracle_semantics": "fullbox_low_tensor_core is diagnostic truth, never a condition"})
        field = fft.irfftn(spec, s=(c["compact_fullbox_ngrid"],)*3, workers=32)
        for row in phase_rows:
            g = dst.create_group(row["anchor_id"])
            coords = [x*c["compact_fullbox_ngrid"]/b["native_ngrid"] for x in native_coordinates(row, 96, b)]
            low = interpolate(field, coords, degree=5)
            interp_error = 0.
            for point in (0, 47, 95):
                position = np.array([x[point] for x in coords])
                exact = np.real(np.sum(coefficients*np.exp(2j*np.pi*(modes@position)/spec.shape[0])))
                interp_error = max(interp_error, float(abs(exact-low[point,point,point])))
            if interp_error > c["technical_max_abs"]:
                raise ValueError("compact interpolation failed direct Fourier check")
            g.attrs["direct_fourier_interpolation_max_abs"] = interp_error
            g.create_dataset("fullbox_low_delta_local96", data=low.astype("f4"), compression="lzf")
            g.create_dataset("fullbox_low_tensor_core", shape=(32,32,32,6), dtype="f4")
            with h5py.File(row["shard"], "r") as src:
                delta = src[row["group"]]["delta_r7_gaussian"][:].astype(np.float64)
            for v in c["variants"]:
                vg = g.create_group(v["name"])
                coarse = interpolate(field, coarse_coords(row, v, b, c["compact_fullbox_ngrid"]), degree=5).astype("f4")
                up = interpolate(coarse, local_coords(v), degree=5)
                residual = (delta-up).astype("f4")
                error = float(np.max(np.abs(up+residual-delta)))
                if error > c["technical_max_abs"]:
                    raise ValueError("coarse plus residual reconstruction failed")
                vg.create_dataset("coarse_delta", data=coarse, compression="lzf")
                vg.create_dataset("fine_residual", data=residual, compression="lzf")
                vg.attrs["response_file"] = str(out/f"{phase}_{row['cap']}_response.h5")
                vg.attrs["response_group"] = f"factor{v['factor']}"
                vg.attrs["response_start"] = (np.asarray(row["center"])-v["span_fine_cells"]//2)//v["factor"]
                vg.attrs["response_side"] = v["span_fine_cells"]//v["factor"]
                checks.append({"anchor_id": row["anchor_id"], "variant": v["name"], "reconstruction_max_abs": error})
            print(f"{phase}: density products {row['anchor_id']}", flush=True)
        del field
        for col, component in enumerate(TENSOR_COMPONENTS):
            field = inverse_component(spec, b["box_mpc_h"], component)
            for row in phase_rows:
                coords = [x*c["compact_fullbox_ngrid"]/b["native_ngrid"] for x in native_coordinates(row, 32, b)]
                dst[row["anchor_id"]]["fullbox_low_tensor_core"][..., col] = interpolate(field, coords, degree=5)
            del field
            print(f"{phase}: fullbox low tensor diagnostic {component}", flush=True)
        for row in phase_rows:
            g = dst[row["anchor_id"]]
            trace_error = np.max(np.abs(g["fullbox_low_tensor_core"][:][...,[0,3,5]].sum(-1)
                                       -g["fullbox_low_delta_local96"][context_slice([48]*3,32)]))
            if trace_error > c["technical_max_abs"]:
                raise ValueError("fullbox low density/tensor trace mismatch")
    del spec
    gc.collect()
    result = {"phase": phase, "fields": {"path": str(path), "sha256": sha256(path)},
              "responses": responses, "checks": checks, "density_source": density,
              "density_mean": mean, "training_ready": False}
    write_json(out/f"{phase}_PRODUCTS_COMPLETE.json", result)
    return result


def attribution(parts, residual, mask):
    stacked = np.stack([p[mask] for p in parts])
    gram = np.einsum("inc,jnc,c->ij", stacked, stacked, [1,2,2,1,2,1])/mask.sum()
    closure = float(np.max(np.abs(sum(parts)-residual)))
    if closure > 2e-6:
        raise ValueError("tensor error attribution does not close")
    return {"component_order": ["wide_low_boundary_and_sampling", "local_high_boundary", "coarse_interpolation_residual"],
            "frobenius_gram_including_cross_terms": gram.tolist(), "closure_max_abs": closure}


def audit(c, b, rows, out, registration):
    cfg = read_json(REPO/c["diagnostic_config"])
    write_json(out/"PHYSICS_TEST_STARTED.json", {"registration": registration, "training_ready": False})
    print("PHYSICS TEST STARTED: wider-coarse plus local-fine versus full-box truth", flush=True)
    core = context_slice([48]*3, 32)
    cell = b["cell_mpc"]*b["coordinate_h"]
    methods = ["parent_96", *[v["name"] for v in c["variants"]], "oracle_fullbox_low_plus_local_high"]
    results, technical = [], []
    with ExitStack() as stack:
        fields = {p: stack.enter_context(h5py.File(out/f"{p}_fields.h5", "r")) for p in PHASES}
        for row in rows:
            with h5py.File(row["shard"], "r") as src:
                g = src[row["group"]]
                delta = g["delta_r7_gaussian"][:].astype(np.float64)
                truth = g["tensor_spectral"][core].astype(np.float64)
                support = g["masks/observed_parent"][core].astype(bool)
            g = fields[row["phase"]][row["anchor_id"]]
            direct_low = g["fullbox_low_delta_local96"][:].astype(np.float64)
            true_low = g["fullbox_low_tensor_core"][:].astype(np.float64)
            high = tensor_from_delta(delta-direct_low, cell)[core]
            tensors = {"parent_96": tensor_from_delta(delta, cell)[core],
                       "oracle_fullbox_low_plus_local_high": true_low+high}
            parts = {}
            for v in c["variants"]:
                vg = g[v["name"]]
                coarse = vg["coarse_delta"][:].astype(np.float64)
                up = interpolate(coarse, local_coords(v), degree=5)
                low_t = tensor_from_delta(coarse, cell*v["factor"])
                low_t = np.stack([interpolate(low_t[..., j], local_coords(v, 32), degree=5) for j in range(6)], axis=-1)
                residual = vg["fine_residual"][:].astype(np.float64)
                combined = low_t+tensor_from_delta(residual, cell)[core]
                tensors[v["name"]] = combined
                # Stored f32 residual quantization is explicitly included in c.
                interpolation_error = tensor_from_delta(residual-(delta-direct_low), cell)[core]
                parts[v["name"]] = [low_t-true_low, high-(truth-true_low), interpolation_error]
                trace = float(np.max(np.abs(combined[...,[0,3,5]].sum(axis=-1)-delta[core])))
                if trace > c["technical_max_abs"] or not np.isfinite(combined).all():
                    raise ValueError("combined tensor trace/numerical failure")
                technical.append(trace)
            item = {k: row[k] for k in ("anchor_id", "phase", "cap", "shell", "support_stratum")}
            item["masks"] = {}
            for mask_name, mask in (("all_science_voxels", np.ones((32,)*3, bool)), ("observed_science_voxels", support)):
                if not mask.any():
                    raise ValueError("empty comparison support")
                reference = science(eigs(truth), mask, cell, cfg)
                values = {"reference": reference}
                for name, tensor in tensors.items():
                    candidate = science(eigs(tensor), mask, cell, cfg)
                    metrics = tensor_metrics(tensor, truth, mask)
                    gates = gate_values(metrics, candidate, reference)
                    values[name] = {"metrics": metrics, "science": candidate, "gate_values": gates,
                                    "failed_gates": failures(gates, cfg["tolerances"])}
                    if name in parts:
                        values[name]["attribution"] = attribution(parts[name], tensor-truth, mask)
                values["resolution_control_f2_vs_f4"] = tensor_metrics(tensors["wide384_f2"], tensors["wide384_f4"], mask)
                item["masks"][mask_name] = values
            results.append(item)
            write_json(out/"anchors"/f"{row['anchor_id']}.json", item)
            print(f"physics checked {len(results)}/96 {row['anchor_id']}", flush=True)
    report = {"registration": registration, "parents": results, "summary": describe(results, methods, cfg),
              "by_phase": {p: describe([r for r in results if r["phase"] == p], methods, cfg) for p in PHASES},
              "by_shell_support": {f"{s}/{t}": describe([r for r in results if r["shell"] == s and r["support_stratum"] == t], methods, cfg)
                                   for s, t in sorted({(r["shell"], r["support_stratum"]) for r in results})},
              "historical_tolerances_use": c["historical_tolerances_use"], "training_ready": False,
              "max_trace_residual": max(technical), "experimental_units": "3 phases; correlated caps/anchors/voxels; no IID claims"}
    write_json(out/"PHYSICS_TEST_REPORT.json", report)
    return report


def run(config_path):
    require_compute()
    c, b, index, rows, screen, native, out = validate_inputs(config_path)
    out.mkdir(parents=True, exist_ok=False)  # partial runs are preserved, never overwritten
    started = time.monotonic()
    watched = {r["shard"] for r in rows}
    for s in screen["sources"]:
        if s["phase"] in PHASES:
            watched.add(s["response_file"]["path"])
            watched.update(v["path"] for v in s["virtual_sources"])
    watched.update(p["density_path"] for p in native["phases"] if p["phase"] in PHASES)
    states = {p: (Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in watched}
    registration = {"config_sha256": sha256(config_path), "source_sha256": source_hashes(),
                    "build_config_sha256": sha256(REPO/c["build_config"]),
                    "diagnostic_config_sha256": sha256(REPO/c["diagnostic_config"]),
                    "dataset_index_sha256": c["dataset_index_sha256"],
                    "base_git": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()}
    write_json(out/"WIDE_COARSE_STARTED.json", {"registration": registration, "config": c,
        "job_id": os.environ["SLURM_JOB_ID"], "node": socket.gethostname()})
    receipts = [build_phase(p, c, b, index, rows, screen, native, out) for p in PHASES]
    write_json(out/"WIDE_COARSE_PRODUCTS_COMPLETE.json", {"registration": registration, "phases": receipts,
        "parents": rows, "training_ready": False, "normalization_fitted": False,
        "local_observations": "unchanged referenced spectral-v2 shards; checksum verified",
        "coarse_observations": "cap-specific raw channels; zero pad only with explicit coverage; no observation wrapping"})
    audit(c, b, rows, out, registration)
    if any((Path(p).stat().st_size, Path(p).stat().st_mtime_ns) != state for p, state in states.items()):
        raise ValueError("input payload changed during run")
    if (source_hashes() != registration["source_sha256"] or sha256(config_path) != registration["config_sha256"]
        or sha256(REPO/c["build_config"]) != registration["build_config_sha256"]
        or sha256(REPO/c["diagnostic_config"]) != registration["diagnostic_config_sha256"]):
        raise ValueError("source/config drift during run")
    write_json(out/"WIDE_COARSE_COMPLETE.json", {"registration": registration, "technical_pass": True,
        "training_ready": False, "r0_physics_pass": False, "holdout_payloads_read": False,
        "learned_training_performed": False, "report_sha256": sha256(out/"PHYSICS_TEST_REPORT.json"),
        "job_id": os.environ["SLURM_JOB_ID"], "node": socket.gethostname(), "elapsed_seconds": time.monotonic()-started})
    print("WIDE COARSE BUILD AND PHYSICS TEST COMPLETE; no training release", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run(parser.parse_args().config)
