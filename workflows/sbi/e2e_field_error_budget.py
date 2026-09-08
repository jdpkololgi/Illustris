#!/usr/bin/env python3
"""Training-only, no-fit physical error attribution against a full-box reference.

The spectral reference removes derivative truncation, not TSC/particle sampling
or simulation error. Parent errors remain coupled exterior/coarse-grid effects.
No quadrature addition of correlated errors and no retrospective science pass.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import os
from pathlib import Path
import subprocess
import time

import h5py
import numpy as np
from scipy import fft, ndimage

from workflows.sbi.e2e_field_build_products import (
    TENSOR_COMPONENTS, context_slice, eigs, read_json, require_compute,
    sha256, tensor_from_delta, validate_config, write_json,
)
from workflows.sbi.e2e_field_dataset import centred_crop
from workflows.sbi.e2e_field_prepare_data import REPO, safe_path

DEFAULT_CONFIG = REPO / "configs/e2e_field_error_budget_v1.json"


def native_coordinates(row, side, build):
    start = np.asarray(row["center"]) - side // 2
    return [np.mod((row["grid"]["origin_mpc"][a] +
                    (np.arange(start[a], start[a]+side)+.5)*build["cell_mpc"])
                   *build["coordinate_h"] + build["box_offset_mpc_h"],
                   build["box_mpc_h"])/(build["box_mpc_h"]/build["native_ngrid"])
            for a in range(3)]


def sample_grid(field, coords, method="linear"):
    """Tensor-product sampling at integer-centred periodic TSC lattice sites."""
    n = field.shape[0]
    base = [np.floor(c).astype(np.int64) for c in coords]
    if method == "legacy_floor":
        return np.asarray(field[np.ix_(*(b % n for b in base))], dtype=np.float64)
    if method == "nearest_round":
        return np.asarray(field[np.ix_(*(np.rint(c).astype(np.int64) % n for c in coords))], dtype=np.float64)
    frac = [c-b for c,b in zip(coords,base)]
    offsets = (0,1) if method == "linear" else (-1,0,1,2)
    if method not in ("linear", "local_cubic_lagrange"):
        raise ValueError("unknown sampling rule")
    weights = [[np.prod([(t-j)/(o-j) for j in offsets if j != o],axis=0)
                for o in offsets] for t in frac]
    result = np.zeros(tuple(len(c) for c in coords), dtype=np.float64)
    for ii in itertools.product(range(len(offsets)), repeat=3):
        ix = [(base[a]+offsets[ii[a]]) % n for a in range(3)]
        w = weights[0][ii[0]][:,None,None]*weights[1][ii[1]][None,:,None]*weights[2][ii[2]][None,None,:]
        result += w*field[np.ix_(*ix)]
    return result


def sample_fd_tensor(potential, coords, cell, order):
    """Two centred first derivatives, optionally higher-order, at legacy sites."""
    positive = {2: [1/2], 4: [2/3,-1/12],
                8: [4/5,-1/5,4/105,-1/280]}[order]
    stencil = [(s*j,s*w) for j,w in enumerate(positive,1) for s in (-1,1)]
    axes = [np.floor(c).astype(np.int64) for c in coords]
    n = potential.shape[0]
    output = np.zeros(tuple(len(c) for c in coords)+(6,), dtype=np.float64)
    for component,(a,b) in enumerate(TENSOR_COMPONENTS):
        # Aggregate coincident shifts on diagonal components before sampling.
        terms = {}
        for da,wa in stencil:
            for db,wb in stencil:
                shift = [0,0,0]
                shift[a] += da
                shift[b] += db
                key = tuple(shift)
                terms[key] = terms.get(key,0.) + wa*wb
        for shift,weight in terms.items():
            ix = [(axes[j]+shift[j]) % n for j in range(3)]
            output[...,component] += weight*potential[np.ix_(*ix)]/(cell*cell)
    return output


def smoothed_spectrum(count_path, box, smoothing, workers=32):
    counts = np.load(safe_path(Path(count_path)), mmap_mode="r", allow_pickle=False)
    n = counts.shape[0]
    if counts.shape != (n,n,n) or counts.dtype != np.float32:
        raise ValueError("invalid native density source")
    mean = float(np.mean(counts,dtype=np.float64))
    if not mean > 0:
        raise ValueError("invalid count mean")
    delta = np.empty(counts.shape,dtype=np.float64)
    for i in range(0,n,16):
        x = np.asarray(counts[i:i+16])
        if not np.isfinite(x).all() or np.min(x) < 0:
            raise ValueError("invalid counts")
        # Match native replay arithmetic, avoiding a normalization confound.
        delta[i:i+16] = x/mean-1
    del counts
    spec = fft.rfftn(delta,workers=workers,overwrite_x=True)
    del delta
    gc.collect()
    k = 2*np.pi*fft.fftfreq(n,d=box/n)
    kz = 2*np.pi*fft.rfftfreq(n,d=box/n)
    for i in range(n):
        k2 = k[i]**2+k[:,None]**2+kz[None,:]**2
        spec[i] *= np.exp(-.5*smoothing**2*k2)
    spec[0,0,0] = 0
    return spec, mean


def inverse_component(spec, box, component, workers=32):
    """One independent full-box spectral Hessian component (or potential)."""
    n = spec.shape[0]
    k = 2*np.pi*fft.fftfreq(n,d=box/n)
    kz = 2*np.pi*fft.rfftfreq(n,d=box/n)
    work = np.empty_like(spec)
    for i in range(n):
        axes = [k[i], k[:,None], kz[None,:]]
        k2 = k[i]**2+k[:,None]**2+kz[None,:]**2
        if i == 0:
            k2[0,0] = 1
        multiplier = -1/k2 if component is None else axes[component[0]]*axes[component[1]]/k2
        work[i] = spec[i]*multiplier
    work[0,0,0] = 0
    if component is not None and component[0] != component[1] and n % 2 == 0:
        # Odd derivatives at self-conjugate Nyquist modes must vanish.
        for a in component:
            sl = [slice(None)]*3
            sl[a] = n//2
            work[tuple(sl)] = 0
    result = fft.irfftn(work,s=(n,n,n),workers=workers,overwrite_x=True)
    del work
    gc.collect()
    return result


def pair_sums(mark, mask, cell, edges):
    """All nonperiodic ordered pairs; ratios equal unordered-pair ratios."""
    n = mask.shape[0]
    shape = (2*n-1,)*3
    shifts = np.arange(-(n-1),n)*cell
    radius = np.sqrt(shifts[:,None,None]**2+shifts[None,:,None]**2+shifts[None,None,:]**2)
    def autocorr(x):
        f = fft.rfftn(x,s=shape)
        return np.rint(fft.fftshift(fft.irfftn(f*f.conj(),s=shape))).astype(np.int64)
    numerator,denominator = autocorr(mark & mask),autocorr(mask)
    sums = []
    for lo,hi in zip(edges[:-1],edges[1:]):
        select = (radius>=lo)&(radius<hi)
        den,num = int(denominator[select].sum()),int(numerator[select].sum())
        sums.append({"ordered_supported_pairs":den,"marked_pairs":num,
                     "value":num/den if den else None})
    return sums


def science(eigen, mask, cell, cfg):
    threshold = cfg["threshold"]
    mark = (eigen[...,1]>threshold)&mask
    labels,_ = ndimage.label(mark,structure=ndimage.generate_binary_structure(3,1))
    connections = []
    for a in range(3):
        ends = []
        for lo in (6,24):
            sl = [slice(12,20)]*3
            sl[a] = slice(lo,lo+2)
            values = set(np.unique(labels[tuple(sl)]).tolist())-{0}
            ends.append(values)
        connections.append(bool(ends[0]&ends[1]))
    voids,_ = ndimage.label((eigen[...,2]<=threshold)&mask,
                           structure=ndimage.generate_binary_structure(3,1))
    sizes = np.bincount(voids.ravel())[1:]
    return {"n":int(mask.sum()),"filling_fraction":float(mark.sum()/mask.sum()),
            "pair":pair_sums(mark,mask,cell,cfg["pair_bin_edges_mpc_h"]),
            "connections_xyz":connections,
            "largest_void_fraction":float(sizes.max(initial=0)/mask.sum())}


def tensor_metrics(pred, truth, mask):
    p,t = eigs(pred),eigs(truth)
    residual = p[mask]-t[mask]
    scatter = np.std(t[mask],axis=0)
    rmse = np.sqrt(np.mean(residual**2,axis=0))
    dt = pred[mask]-truth[mask]
    bias = dt.mean(axis=0)
    weights = np.array([1,2,2,1,2,1])
    total = float(np.mean(np.sum(dt**2*weights,axis=-1)))
    centered = dt-bias
    bias_energy = float(np.sum(bias**2*weights))
    return {"eigen_rmse":rmse.tolist(),"eigen_bias":residual.mean(axis=0).tolist(),
            "eigen_rmse_over_truth_std":(rmse/np.maximum(scatter,1e-30)).tolist(),
            "eigengap_rmse":np.sqrt(np.mean(np.diff(residual,axis=-1)**2,axis=0)).tolist(),
            "tensor_frobenius_rmse":float(np.sqrt(total)),"tensor_mean_residual":bias.tolist(),
            "constant_tensor_residual_energy_fraction":bias_energy/max(total,1e-30),
            "oracle_constant_removed_tensor_rmse":float(np.sqrt(np.mean(np.sum(centered**2*weights,axis=-1)))),
            "two_collapsed_label_disagreement":float(np.mean((p[mask,1]>.2)!=(t[mask,1]>.2))),
            "four_class_disagreement":float(np.mean(np.sum(p[mask]>.2,axis=-1)!=np.sum(t[mask]>.2,axis=-1)))}


def config_and_inputs(path):
    cfg = read_json(path)
    if cfg["schema_version"] != "e2e-physical-error-budget-v1" or cfg["phases"] != ["ph000","ph002","ph003"]:
        raise ValueError("only the three fixed training phases are authorized")
    if cfg["training_ready"] or cfg["new_training_products"] or cfg["science_acceptance_thresholds_frozen"]:
        raise ValueError("an error attribution audit cannot release training")
    build_path = REPO/cfg["build_config"]
    build = read_json(build_path)
    validate_config(build)
    root = Path(build["output_root"])
    index_path = root/"parent_arrays/DATASET_INDEX.json"
    index = read_json(index_path)
    if index["config_sha256"] != sha256(build_path):
        raise ValueError("input build contract drift")
    native = read_json(root/"native_truth/NATIVE_TRUTH_COMPLETE.json")
    if native["dataset_index_sha256"] != sha256(index_path):
        raise ValueError("native truth does not match dataset")
    out = safe_path(Path(cfg["output_root"])).resolve()
    allowed = Path("/pscratch/sd/d/dkololgi/abacus/e2e_field_v1").resolve()
    if not out.is_relative_to(allowed) or out == allowed or out == root.resolve():
        raise ValueError("audit output must be a separate E2E child")
    return cfg,build,index,native,out


def run(path):
    cfg,build,index,native,out = config_and_inputs(path)
    out.mkdir(exist_ok=False)
    rows = [r for r in index["parents"] if r["phase"] in cfg["phases"]]
    if len(rows)!=96 or any(r["role"]!="train" for r in rows):
        raise ValueError("unexpected training panel")
    write_json(out/"FROZEN_AUDIT.json",{"config":cfg,"config_sha256":sha256(path),
               "code_sha256":sha256(__file__),
               "dependency_sha256":{name:sha256(Path(__file__).with_name(name)) for name in
                  ("e2e_field_build_products.py","e2e_field_dataset.py","e2e_field_prepare_data.py")},
               "base_git":subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),
               "anchor_ids":[r["anchor_id"] for r in rows],"job_id":os.environ["SLURM_JOB_ID"]})
    phase_reports = []
    for phase in cfg["phases"]:
        started = time.monotonic()
        selected = [r for r in rows if r["phase"]==phase]
        original = next(p for p in native["phases"] if p["phase"]==phase)
        for file,expected in [(original["density_path"],original["density_sha256"]),
                              (original["shard"],original["shard_sha256"])]:
            if sha256(file)!=expected:
                raise ValueError("input payload hash mismatch")
        for shard in [s for s in index["shards"] if any(r["shard"]==s["path"] for r in selected)]:
            if sha256(shard["path"]) != shard["sha256"]:
                raise ValueError("parent payload hash mismatch")
        print(f"{phase}: source hashes verified; constructing full-box Fourier reference",flush=True)
        spec,mean = smoothed_spectrum(original["density_path"],build["box_mpc_h"],build["native_smoothing_mpc_h"])
        shard_path = out/f"{phase}_spectral_reference.h5"
        with h5py.File(shard_path,"x") as saved:
            saved.attrs.update({"phase":phase,"training_ready":False,"config_sha256":sha256(path)})
            for row in selected:
                g = saved.create_group(row["anchor_id"])
                for rule in ("legacy_floor","linear"):
                    g.create_dataset("density_"+rule,shape=(128,)*3,dtype="f4",chunks=(32,)*3)
                for rule in cfg["interpolation"]:
                    g.create_dataset("tensor_"+rule,shape=(32,32,32,6),dtype="f8")
            for column,component in enumerate(TENSOR_COMPONENTS):
                field = inverse_component(spec,build["box_mpc_h"],component)
                for row in selected:
                    g = saved[row["anchor_id"]]
                    coords = native_coordinates(row,32,build)
                    for rule in cfg["interpolation"]:
                        g["tensor_"+rule][...,column] = sample_grid(field,coords,rule)
                    if component[0]==component[1]:
                        wide = native_coordinates(row,128,build)
                        for rule in ("legacy_floor","linear"):
                            ds = g["density_"+rule]
                            ds[:] = ds[:]+sample_grid(field,wide,rule).astype(np.float32)
                del field
                gc.collect()
                print(f"{phase}: spectral component {component} and sampling controls complete",flush=True)
            potential = inverse_component(spec,build["box_mpc_h"],None)
            del spec
            gc.collect()
            replay_max = 0.
            with h5py.File(original["shard"],"r") as legacy:
                for row in selected:
                    g = saved[row["anchor_id"]]
                    coords = native_coordinates(row,32,build)
                    truth = legacy[row["anchor_id"]]["tensor_native"][32:64,32:64,32:64,:]
                    for order in cfg["finite_difference_orders"]:
                        fd = sample_fd_tensor(potential,coords,build["box_mpc_h"]/build["native_ngrid"],order)
                        g.create_dataset(f"tensor_fd{order}",data=fd)
                        if order==2:
                            replay_max = max(replay_max,float(np.max(np.abs(fd-truth))))
            del potential
            gc.collect()
            if replay_max > 2e-6:
                raise ValueError(f"native FD replay failed: {replay_max}")
        phase_report = evaluate_phase(selected,original,shard_path,build,cfg)
        phase_report.update({"phase":phase,"reference_shard":str(shard_path),
            "reference_sha256":sha256(shard_path),"density_source_sha256":original["density_sha256"],
            "density_mean":mean,"native_tensor_replay_max_abs":replay_max,
            "elapsed_seconds":time.monotonic()-started})
        write_json(out/f"{phase}_ERROR_BUDGET.json",phase_report)
        phase_reports.append(phase_report)
        print(f"{phase}: error attribution and science diagnostics complete in {phase_report['elapsed_seconds']:.1f}s",flush=True)
    write_json(out/"ERROR_BUDGET_COMPLETE.json",{"schema_version":cfg["schema_version"],
        "config_sha256":sha256(path),"code_sha256":sha256(__file__),"job_id":os.environ["SLURM_JOB_ID"],
        "phases":phase_reports,"anchors":96,"science_acceptance_thresholds_frozen":False,
        "posterior_scaled_budget_evaluated":False,"r0_physics_pass":False,"training_ready":False})


def evaluate_phase(rows,original,shard_path,build,cfg):
    output = []
    cell = build["cell_mpc"]*build["coordinate_h"]
    center = context_slice([48]*3,32)
    with h5py.File(shard_path,"r") as spectral, h5py.File(original["shard"],"r") as native:
        for row in rows:
            g = spectral[row["anchor_id"]]
            tensors = {"native_fd2":native[row["anchor_id"]]["tensor_native"][center],
                       **{f"spectral_{rule}":g["tensor_"+rule][:] for rule in cfg["interpolation"]},
                       **{f"fd{order}":g[f"tensor_fd{order}"][:] for order in (4,8)}}
            with h5py.File(row["shard"],"r") as packed:
                old = packed[row["group"]]["delta_r7_trace"][:]
                support = packed[row["group"]]["masks/observed_parent"][center].astype(bool)
            means = {}
            for side in cfg["parent_sides"]:
                core = context_slice([side//2]*3,32)
                for rule in ("legacy_floor","linear"):
                    delta = centred_crop(g["density_"+rule][:],side)
                    key = f"clean_{rule}_{side}"
                    tensors[key] = tensor_from_delta(delta,cell)[core]
                    means[key] = float(delta.mean(dtype=np.float64))
                if side<=96:
                    delta = centred_crop(old,side)
                    tensors[f"inherited_{side}"] = tensor_from_delta(delta,cell)[core]
            # Controlled DC ablation, not a proposed production mean subtraction.
            nodc = tensors["clean_linear_96"].copy()
            nodc[...,[0,3,5]] -= means["clean_linear_96"]/3
            tensors["clean_linear_96_no_dc"] = nodc
            comparisons = [("fd2_only","native_fd2","spectral_legacy_floor"),
                           ("fd4_only","fd4","spectral_legacy_floor"),
                           ("fd8_only","fd8","spectral_legacy_floor"),
                           ("floor_vs_linear","spectral_legacy_floor","spectral_linear"),
                           ("round_vs_linear","spectral_nearest_round","spectral_linear"),
                           ("linear_vs_cubic","spectral_linear","spectral_local_cubic_lagrange")]
            for side in cfg["parent_sides"]:
                comparisons.extend([(f"clean_floor_parent_{side}",f"clean_legacy_floor_{side}","spectral_legacy_floor"),
                                    (f"clean_linear_parent_{side}",f"clean_linear_{side}","spectral_linear")])
                if side<=96:
                    comparisons.append((f"inherited_parent_{side}",f"inherited_{side}","native_fd2"))
            comparisons.append(("clean_linear_no_dc_96","clean_linear_96_no_dc","spectral_linear"))
            item = {"anchor_id":row["anchor_id"],"cap":row["cap"],"shell":row["shell"],
                    "support_stratum":row["support_stratum"],"parent_means":means,"masks":{}}
            for name,mask in (("all_science_voxels",np.ones((32,)*3,dtype=bool)),
                              ("observed_science_voxels",support)):
                if not mask.any():
                    raise ValueError("empty registered science mask")
                stats = {key:science(eigs(value),mask,cell,cfg) for key,value in tensors.items()}
                checks = {label:tensor_metrics(tensors[p],tensors[t],mask) for label,p,t in comparisons}
                # Exact vector-error decomposition: do not combine RMS in quadrature.
                pieces = [tensors["inherited_96"]-tensors["clean_legacy_floor_96"],
                          tensors["clean_legacy_floor_96"]-tensors["spectral_legacy_floor"],
                          tensors["spectral_legacy_floor"]-tensors["native_fd2"]]
                matrix = np.array([[np.mean(np.sum(a[mask]*b[mask]*[1,2,2,1,2,1],axis=-1))
                                    for b in pieces] for a in pieces])
                actual = np.mean(np.sum((tensors["inherited_96"][mask]-tensors["native_fd2"][mask])**2*[1,2,2,1,2,1],axis=-1))
                if abs(matrix.sum()-actual)>1e-12:
                    raise ValueError("non-closing tensor error decomposition")
                item["masks"][name] = {"science":stats,"comparisons":checks,
                    "tensor_error_gram_order":["propagated_trace_filter","parent_and_resampling","spectral_minus_native_tensor"],
                    "tensor_error_gram":matrix.tolist(),"total_tensor_mse":float(actual)}
            output.append(item)
            print(f"science checks: {row['anchor_id']}",flush=True)
    return {"parents":output,"experimental_unit":"phase; parents/caps/voxels correlated, descriptive summaries only"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    args = p.parse_args()
    require_compute()
    run(args.config)


if __name__ == "__main__":
    main()
