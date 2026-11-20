"""Fit corrected G-PATCH transforms on training phases only, then apply to visible phases.

Input JSON maps phase names to verified PATCH_GEOMETRY_READY directories. Never
loads targets. The eighth feature is the frozen cap/redshift selection density
used by corrected P12-A, not the old radial spline. Unsupported context gets
zero density (the selection contract's value), floored only inside the log.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.special import boxcox
from sklearn.preprocessing import PowerTransformer

from workflows.abacus_tweb.lowz_point_loader import DEV, ROOT, array, read, require, roles
from workflows.abacus_tweb.lowz_graph_union_pilot import digest
from workflows.abacus_tweb.lowz_graph_patch_index import Segments

N = 100_000
SI = (0,2,3,4,5,6)
EPS = 1e-6


def density_log(z, cap, selection):
    edges = np.asarray(selection["bin_edges"], dtype=float)
    z = np.asarray(z)
    require(np.isfinite(z).all() and np.all((z >= edges[0]) & (z <= edges[-1])),
            "redshift outside declared context")
    bins = np.minimum(np.searchsorted(edges, z, side="right")-1, len(edges)-2)
    n = np.asarray(selection["density"])[cap,bins]
    valid = np.asarray(selection["valid"],bool)[cap,bins]
    require(np.all(n[~valid]==0) and np.all(n[valid]>0), "invalid selection contract")
    return np.log(np.maximum(n,1e-12))


def scale_raw(x, cap, medians):
    x = np.array(x,dtype=float,copy=True)
    for c in (0,1):
        rows = cap == c
        x[np.ix_(rows, SI)] /= np.maximum(medians[c],1e-9)
    x += EPS
    require(np.isfinite(x).all() and np.all(x>0), "invalid Box-Cox input")
    return x


def phase_inputs(phase, directory):
    receipt = read(directory / "PATCH_GEOMETRY_READY.json")
    require(receipt["phase"]==phase and receipt["passed"] and len(receipt["checks"])==2
            and digest(receipt["union"])==receipt["union_sha256"], "unverified phase geometry")
    pc = read(ROOT / "phases" / phase / "phase_contract.json")
    p1 = read(Path(pc["inputs"]["p1_manifest"]))
    with np.load(p1["index"]) as idx:
        cap = np.asarray(idx["cap"],dtype=np.int8)
        ctx = idx["context"].astype(bool)
    raw = array(directory / "node_features.npy")
    z = array(ROOT / "phases" / phase / "parent_redshift.npy")
    require(raw.shape==(len(z),7) and len(ctx)==len(cap)==len(raw)==receipt["rows"], "node shape mismatch")
    return receipt, raw, z, cap, ctx


def fit(index, training, selection):
    require(tuple(training) == tuple(p for p in roles(ROOT) if p not in DEV)
            and tuple(selection["fit_phases"])==tuple(training), "not the registered 20-phase fit")
    nodes, logs, edges = ([[],[]] for _ in range(3))
    for i,p in enumerate(training):
        rec,raw,z,cap,ctx = phase_inputs(p,index[p])
        owned = array(ROOT / "adapters" / p / "field/core_active_parent.npy")
        pairs = Segments([array(q) for q in rec["source_pairs"]])
        attr = Segments([array(q) for q in rec["source_attrs"]])
        nrng, erng = np.random.default_rng(104729+i), np.random.default_rng(32452843+i)
        for c in (0,1):
            candidates = owned[cap[owned]==c]
            require(len(candidates)>=N,"not enough distinct training nodes")
            ids = np.sort(nrng.choice(candidates,N,replace=False))
            require(ctx[ids].all(), "sample outside context")
            nodes[c].append(np.asarray(raw[ids],dtype=float))
            logs[c].append(density_log(z[ids],cap[ids],selection))
            # Uniform random canonical edges, conditional on cap, with replacement.
            chunks, n = [], 0
            while n<N:
                draw = erng.integers(len(pairs),size=N)
                # Sort *after* selecting a random sample; truncating sorted draws
                # would bias the sample towards the Delaunay component.
                keep = draw[cap[pairs[draw][:,0]]==c][:N-n]
                chunks.append(keep); n+=len(keep)
            edges[c].append(attr[np.sort(np.concatenate(chunks))])
    med = [np.median(np.concatenate(a)[:,SI],axis=0) for a in nodes]
    mapped = [scale_raw(np.concatenate(v),np.full(sum(map(len,v)),c),med) for c,v in enumerate(nodes)]
    power = PowerTransformer(method="box-cox").fit(np.concatenate(mapped))
    l = np.concatenate([np.concatenate(v) for v in logs])
    edge = {}
    for c in (0,1):
        sample = np.concatenate(edges[c]).astype(float)
        require(np.isfinite(sample).all(), "invalid edge sample")
        length = float(np.median(sample[:,0]))
        require(length>0, "invalid edge length")
        ll = np.log(np.maximum(sample[:,0]/length,EPS))
        ld = np.log(np.maximum(sample[:,4],EPS))
        edge[str(c)] = dict(edge_length_si_median=length,log_length_mean=float(ll.mean()),
            log_length_std=float(ll.std()),log_density_contrast_mean=0.,
            log_density_contrast_std=float(np.sqrt(np.mean(ld**2))))
        require(ll.std()>0 and edge[str(c)]["log_density_contrast_std"]>0, "degenerate edge transform")
    require(l.std()>0,"degenerate radial selection")
    return dict(medians=np.array(med).tolist(),lambdas=power.lambdas_.tolist(),
        mean=power._scaler.mean_.tolist(),scale=power._scaler.scale_.tolist(),
        log_mean=float(l.mean()),log_std=float(l.std()),edge=edge)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("index",type=Path)
    ap.add_argument("output",type=Path)
    args = ap.parse_args()
    require("SLURM_STEP_ID" in os.environ,"compute step required")
    visible = roles(ROOT)
    index = {p:Path(d) for p,d in read(args.index).items()}
    require(set(index)==set(visible),"geometry manifest must contain exactly 23 visible phases")
    path = ROOT / "transforms/field/selection_manifest.json"
    training = tuple(p for p in visible if p not in DEV)
    sel = read(path)
    require(sel["pass"] and sel["omitted_phase"] is None and list(training)==sel["fit_phases"],
            "wrong radial selection")
    args.output.mkdir(exist_ok=False)
    spec = fit(index, training, sel["selection"])
    products = {}
    for p in visible:
        rec,raw,z,cap,ctx = phase_inputs(p,index[p])
        d = args.output / p
        d.mkdir()
        dst = d / "node_features_8d.npy"
        arr = np.lib.format.open_memmap(dst,mode="w+",dtype=np.float32,shape=(len(z),8))
        for start in range(0,len(z),250_000):
            sl = slice(start,start+250_000)
            mask = ctx[sl]
            c = cap[sl][mask]
            vals = scale_raw(raw[sl][mask],c,spec["medians"])
            # Out-of-domain rows are never indexed by a corrected patch.
            block = np.zeros((len(mask),8),dtype=np.float32)
            block[mask,:7] = (boxcox(vals,spec["lambdas"])-spec["mean"])/spec["scale"]
            block[mask,7] = (density_log(z[sl][mask],c,sel["selection"])-spec["log_mean"])/spec["log_std"]
            require(np.isfinite(block).all(),"nonfinite transformed graph node")
            arr[sl] = block
        arr.flush()
        del arr
        products[p] = dict(path=str(dst.resolve()),sha256=digest(dst),union_sha256=rec["union_sha256"])
    result = dict(schema="corrected-lowz-gpatch-features-v1",fit_phases=training,
        applied=products,geom_index_sha256=digest(args.index),selection_sha256=digest(path),
        samples_per_training_phase_cap=N,**spec,**{"pass":True})
    (args.output / "graph_transform.json").write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
