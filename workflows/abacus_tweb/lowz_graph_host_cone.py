"""Opt-in CPU extractor for exact two-pass owned-row G-PATCH calculations.

On a bidirectional graph one pass needs send/receive sums on requested
nodes, hence all incoming edges at receivers within one hop and node
features within two. Two passes need incoming edges at distance <=3 and
features at distance <=4. Halo rows are context, not qualified outputs.
"""
import numpy as np

from workflows.abacus_tweb.p5_graph_patch_utils import (
    GraphPatch, _incident_ids, reverse_edge_features, reverse_k_hop_from_incident_csr,
)


def owned_patch(adapter, cores, *, block=262144):
    cores = np.asarray(cores,dtype=np.int64)
    if cores.ndim != 1 or not len(cores) or block < 1:
        raise ValueError("nonempty cores and positive block required")
    if len(np.unique(adapter.core_cap[cores])) != 1:
        raise ValueError("cannot combine different caps")
    owned = np.unique(np.concatenate([adapter.core_nodes(int(c))[0] for c in cores]))
    if not len(owned):
        raise ValueError("empty authoritative row set")
    off, inc, table = adapter.offsets, adapter.incident_edge_id, adapter.union_pairs
    inner = reverse_k_hop_from_incident_csr(owned,off,inc,table,3)
    ids = _incident_ids(inner,off,inc)
    in3 = np.zeros(len(off)-1,dtype=bool)
    in3[inner] = True
    seen = in3.copy()
    nf = nr = 0
    for start in range(0,len(ids),block):
        u,v = table[ids[start:start+block]].T
        seen[u] = seen[v] = True
        nf += np.count_nonzero(in3[v])
        nr += np.count_nonzero(in3[u])
    nodes = np.flatnonzero(seen)
    local = np.empty(len(seen),dtype=np.int32)
    local[nodes] = np.arange(len(nodes),dtype=np.int32)
    del seen
    s = np.empty(nf+nr,dtype=np.int32)
    r = np.empty_like(s)
    eid = np.empty(len(s),dtype=np.int64)
    attrs = np.empty((len(s),5),dtype=np.float32)
    f, g = 0, nf
    for start in range(0,len(ids),block):
        ix = ids[start:start+block]
        u,v = table[ix].T
        forward, reverse = in3[v], in3[u]
        a,b = np.count_nonzero(forward), np.count_nonzero(reverse)
        # Preserve original forward-then-reverse edge order and P2 convention.
        w = adapter.union_edge_features[ix]
        dest = slice(f,f+a)
        eid[dest],s[dest],r[dest],attrs[dest] = ix[forward],local[u[forward]],local[v[forward]],w[forward]
        dest = slice(g,g+b)
        eid[dest],s[dest],r[dest] = ix[reverse],local[v[reverse]],local[u[reverse]]
        attrs[dest] = reverse_edge_features(w[reverse])
        f += a
        g += b
    assert f==nf and g==len(eid)
    mask = np.isin(nodes,owned,assume_unique=True)
    return GraphPatch(core_id=int(cores[0]) if len(cores)==1 else -1,fold=-1,
                      num_passes=2,dependency_hops=4,parent_node_id=nodes,
                      node_features=np.asarray(adapter.node_features[nodes],dtype=np.float32),
                      union_edge_id=eid,edge_features=attrs,senders=s,receivers=r,
                      authoritative_core_mask=mask,strict_support_mask=np.zeros_like(mask),
                      loss_mask=mask.copy(),loss_policy="authoritative" if len(cores)==1 else "authoritative_group")
