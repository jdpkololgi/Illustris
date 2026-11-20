"""Index corrected context unions in place; independently verify two real four-hop patches."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from workflows.abacus_tweb.lowz_graph_union_pilot import digest
from workflows.abacus_tweb.lowz_point_loader import ROOT, DEV, roles, read, require, array
from workflows.abacus_tweb.p5_graph_patch_utils import assemble_patch
from workflows.abacus_tweb.p5_build_graph_patch_adapter import chunks, add_degrees, fill_incident

BLOCK = 500_000


class Segments:
    """Index an ordered concatenation of existing tables without copying full radius payloads."""
    def __init__(self, parts):
        self.parts = tuple(parts)
        self.ends = np.cumsum([len(p) for p in parts], dtype=np.int64)
        self.shape = (int(self.ends[-1]), *parts[0].shape[1:])
        self.dtype = parts[0].dtype
        require(all(p.shape[1:] == self.shape[1:] and p.dtype == self.dtype for p in parts),
                "incompatible graph components")

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, ids):
        ids = np.asarray(ids)
        require(ids.ndim == 1 and ids.dtype.kind in "iu" and
                not ((ids < 0) | (ids >= len(self))).any(), "invalid global edge indices")
        out = np.empty((len(ids), *self.shape[1:]), self.dtype)
        start = 0
        for part, stop in zip(self.parts, self.ends):
            mask = (ids >= start) & (ids < stop)
            out[mask] = part[ids[mask] - start]
            start = stop
        return out


def oracle(core, parts, num_nodes, hops):
    """Deliberately no CSR: stream every source pair against the previous frontier."""
    seen = np.zeros(num_nodes, bool)
    seen[core] = True
    for _ in range(hops):
        nxt = seen.copy()
        for part in parts:
            for a, b in chunks(len(part), BLOCK):
                u, v = part[a:b].T
                nxt[u[seen[v]]] = True
                nxt[v[seen[u]]] = True
        seen = nxt
    edge_ids, offset = [], 0
    for part in parts:
        for a, b in chunks(len(part), BLOCK):
            u, v = part[a:b].T
            edge_ids.append(np.flatnonzero(seen[u] & seen[v]) + offset + a)
        offset += len(part)
    return np.flatnonzero(seen), np.concatenate(edge_ids)


def build(phase, union, out):
    require("SLURM_STEP_ID" in os.environ and phase in roles(ROOT), "visible phase compute step required")
    marker = read(union / "CORRECTED_UNION_READY.json")
    p2path = union / "p2b_union_manifest.json"
    require(marker["phase"] == phase and digest(p2path) == marker["output_sha256"], "wrong corrected union")
    p2 = read(p2path)
    pc = read(ROOT / "phases" / phase / "phase_contract.json")
    idx_path = Path(read(Path(pc["inputs"]["p1_manifest"]))["index"])
    require(str(idx_path) == p2["canonical_index"], "wrong canonical index")
    with np.load(idx_path) as idx:
        context = idx["context"].astype(bool)
        cap = idx["cap"]
    parent = len(context)
    totals = p2["counts"]
    require(parent == pc["parent_rows"] == totals["parent_nodes"] and
            int(context.sum()) == totals["context_nodes"], "wrong corrected parent domain")
    src = marker["source"]
    old = Path(read(ROOT.parent / "registry.json")["reuse_source_root"])
    geometry = read(old / phase / "p2_graph/P2_COMPLETE.json")
    require(geometry["pass"] and geometry["phase"] == phase and src == geometry["inputs"],
            "wrong parent graph binding")
    gm_path = Path(src["graph_metadata"])
    src_pairs = array(gm_path.parent / read(gm_path)["files"]["edges"])
    meta = read(Path(src["gnn_metadata"]))
    gnpath = Path(meta["outputs"]["gnn_arrays_npz"])
    require(digest(gnpath) == geometry["hashes"]["gnn_arrays"], "changed GNN geometry")
    out.mkdir(exist_ok=False)
    def table(name, dtype, shape):
        return np.lib.format.open_memmap(out / (name + ".npy"), mode="w+", dtype=dtype, shape=shape)
    edges = totals["delaunay_pairs_context"]
    pairs = table("delaunay_pairs", np.int32, (edges, 2))
    attrs = table("delaunay_attr", np.float32, (edges, 5))
    with np.load(gnpath) as gn:
        x, ei, ea = gn["x"], gn["edge_index"], gn["edge_attr"]
        require(x.shape == (parent,7) and ei.shape == (2,len(src_pairs))
                and ea.shape == (len(src_pairs),5), "parent geometry dimensions")
        np.save(out / "node_features.npy", x.astype(np.float32, copy=False), allow_pickle=False)
        write = 0
        for a, b in chunks(len(src_pairs), BLOCK):
            block = src_pairs[a:b]
            np.testing.assert_array_equal(block, ei[:,a:b].T)
            require(block.min() >= 0 and block.max() < parent, "invalid parent index")
            u,v = block.T
            valid = context[u] & context[v]
            stop = write + int(valid.sum())
            require(np.all(cap[u[valid]] == cap[v[valid]]), "cross-cap Delaunay edge")
            pairs[write:stop], attrs[write:stop] = block[valid], ea[a:b][valid]
            write = stop
        require(write == edges, "Delaunay filter count mismatch")
        del ei, ea, x
    pairs.flush(); attrs.flush()
    pp, aa = [pairs], [attrs]
    for name, cap_id in (("NGC", 1), ("SGC", 0)):
        c = p2["components"][name]
        for key in ("pairs", "edge_attr"):
            require(digest(c[key]) == c[key+"_sha256"], f"{name} {key} hash mismatch")
        p, v = array(c["pairs"]), array(c["edge_attr"])
        require(p.shape == (c["radius_only_pairs"], 2) and v.shape == (len(p),5), "radius dimension mismatch")
        for a,b in chunks(len(p), BLOCK):
            ids = p[a:b]
            require(ids.min() >= 0 and ids.max() < parent and context[ids].all()
                    and (cap[ids] == cap_id).all(), "radius edge outside corrected domain")
        pp.append(p); aa.append(v)
    combined, attributes = Segments(pp), Segments(aa)
    require(len(combined) == totals["union_pairs_context"] < 2**31 and len(attributes) == len(combined),
            "unrepresentable union")
    degree = np.zeros(parent, np.int64)
    for part in pp:
        for a,b in chunks(len(part), BLOCK):
            add_degrees(part[a:b], degree)
    offsets = np.r_[np.int64(0), np.cumsum(degree)]
    require(offsets[-1] == 2*len(combined), "incident count mismatch")
    np.save(out / "incident_offsets.npy", offsets, allow_pickle=False)
    inc = table("incident_edge_id", np.int32, (2*len(combined),))
    cursor = offsets[:-1].copy()
    base = 0
    for part in pp:
        for a,b in chunks(len(part), BLOCK):
            fill_incident(part[a:b], base+a, cursor, inc)
        base += len(part)
    np.testing.assert_array_equal(cursor, offsets[1:])
    inc.flush()
    # Use the exact P12-A core ownership; no fabricated in-phase strict-support flags.
    field = ROOT / "adapters" / phase / "field"
    cores = array(ROOT / "phases" / phase / ("validation_core_id.npy" if phase in DEV else "training_core_id.npy"))
    caps = array(field / "core_cap.npy")
    starts, parents = array(field / "core_active_offsets.npy"), array(field / "core_active_parent.npy")
    raw = array(out / "node_features.npy")
    checks = []
    for cap_id in (0,1):
        pool = cores[caps[cores] == cap_id]
        cid = int(pool[len(pool)//2])
        owned = parents[starts[cid]:starts[cid+1]]
        patch = assemble_patch(core_id=cid, fold=-1, core_parent_ids=owned, loss_parent_ids=owned,
            strict_parent_ids=np.empty(0, np.int64), loss_policy="authoritative", num_passes=2,
            dependency_hops=4, node_features=raw, union_pairs=combined, union_edge_features=attributes,
            offsets=offsets, incident_edge_id=inc)
        nodes, eids = oracle(owned, pp, parent, 4)
        np.testing.assert_array_equal(nodes, patch.parent_node_id)
        np.testing.assert_array_equal(np.tile(eids,2), patch.union_edge_id)
        np.testing.assert_array_equal(owned, nodes[patch.loss_mask])
        require(context[nodes].all() and (cap[nodes] == cap_id).all(), "invalid patch context")
        checks.append(dict(core=cid, cap=cap_id, supervised=len(owned), nodes=len(nodes), directed=patch.n_edge))
    receipt = dict(phase=phase, passed=True, union=str(p2path), union_sha256=marker["output_sha256"],
                   rows=parent, edges=len(combined), checks=checks, node_features="unscaled parent graph metrics",
                   source_pairs=[str(p.filename) for p in pp], source_attrs=[str(p.filename) for p in aa])
    (out / "PATCH_GEOMETRY_READY.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase")
    parser.add_argument("union", type=Path)
    parser.add_argument("output", type=Path)
    a = parser.parse_args()
    build(a.phase, a.union, a.output)
