"""Exact two-pass graph readout for selected nodes without patch-wide edge latents."""
from contextlib import contextmanager

import torch

from workflows.abacus_tweb.lowz_graph_attention import BoundedGraphPatchNet
from workflows.abacus_tweb.p8_train_graph_patch import transformed_patch as original_patch


def owned_patch(patch, edge_spec, cap, device):
    inputs = original_patch(patch, edge_spec, cap, device)
    owned = torch.as_tensor(patch.loss_mask | patch.authoritative_core_mask, device=device)
    return (*inputs, owned)


class _Accumulate(torch.autograd.Function):
    """Adjoint of the two index_adds needs indices, not the weighted edge values."""
    @staticmethod
    def forward(ctx, sent, received, w, s, r):
        ctx.save_for_backward(s, r)
        ctx.mark_dirty(sent, received)
        sent.index_add_(0, s, w)
        received.index_add_(0, r, w)
        return sent, received

    @staticmethod
    def backward(ctx, ds, dr):
        s, r = ctx.saved_tensors
        return ds, dr, ds[s] + dr[r], None, None


def receiver_sets(s, r, owned):
    # A desired sender also aggregates outgoing messages; each such message
    # needs the *whole* incoming softmax neighborhood of its receiver.
    def needed(nodes):
        receivers = nodes.clone()
        receivers[r[nodes[s]]] = True
        return receivers
    last = needed(owned)
    mid = owned | last
    mid[s[last[r]]] = True
    first = needed(mid)
    return first & ~last, last


def receiver_groups(s, r, receiver_mask, width):
    ids = torch.where(receiver_mask[r])[0]
    if not ids.numel():
        return []
    ids = ids[torch.argsort(r[ids], stable=True)]
    dest = r[ids]
    # Move nominal boundaries to the beginning of the receiver. A receiver
    # with degree >width is kept intact. Only O(groups) offsets go to host.
    cuts = torch.searchsorted(dest, dest[width::width].contiguous())
    ends = sorted({0, *cuts.tolist(), len(ids)})
    return [(ids[i:j], s[ids[i:j]], dest[i:j]) for i,j in zip(ends,ends[1:])]


@contextmanager
def replay_seed(device, seed):
    if seed is None:
        yield
        return
    cuda = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda):
        torch.random.default_generator.manual_seed(seed)
        if cuda:
            torch.cuda.default_generators[cuda[0]].manual_seed(seed)
        yield


class StreamingGraphPatchNet(BoundedGraphPatchNet):
    # Leave small patches on the verified eager/chunked path.
    stream_threshold = 262144

    @staticmethod
    def message(layer, nodes, edge, s, r, seed):
        src, dst = nodes[s], nodes[r]
        with replay_seed(nodes.device, seed):
            updated = layer.edge(torch.cat((edge,src,dst),dim=1))
        inp = torch.cat((updated,src,dst),dim=1)
        logits = torch.cat([head(inp) for head in layer.attention],dim=1)
        weights = layer.receiver_softmax(logits,r,len(nodes))
        return (updated.reshape(-1,layer.heads,layer.head_size)*weights[:,:,None]).reshape_as(edge)

    def forward(self, nodes, edges, senders, receivers, owned=None):
        if len(edges) <= self.stream_threshold:
            return super().forward(nodes, edges, senders, receivers)
        if len(self.passes) != 2:
            raise ValueError("receiver-streamed dependency calculation requires two passes")
        if owned is None:
            owned = torch.ones(len(nodes),device=nodes.device,dtype=torch.bool)
        if owned.dtype != torch.bool or owned.shape != (len(nodes),):
            raise ValueError("invalid owned-node mask")
        early, final = receiver_sets(senders,receivers,owned)
        final_groups = receiver_groups(senders,receivers,final,self.chunk_edges)
        groups = final_groups + receiver_groups(senders,receivers,early,self.chunk_edges)
        if len(edges) >= 2 * getattr(self, "_logged_stream_scale", 0):
            print(f"G-PATCH streamed: training={self.training} edges={len(edges)} "
                  f"first={sum(len(g[0]) for g in groups)} final={sum(len(g[0]) for g in final_groups)} "
                  f"max_group={max((len(g[0]) for g in groups),default=0)}",flush=True)
            self._logged_stream_scale = len(edges)
        n0 = self.node_embed(nodes)
        l0,l1 = self.passes
        stochastic = self.training and any(m.p > 0 for m in self.modules()
                                            if isinstance(m, torch.nn.Dropout))
        base = int(torch.randint(2**55,(),device="cpu")) if stochastic else None
        def seed(i):
            return None if base is None else base+i
        def first(n, raw, s, r, key):
            return self.message(l0,n,self.edge_embed(raw),s,r,key)
        def second(n, mid, raw, s, r, k0, k1):
            e0 = self.edge_embed(raw)
            e1 = e0 + self.message(l0,n,e0,s,r,k0)
            return self.message(l1,mid,e1,s,r,k1)
        def accumulate(layer,n,work,fn):
            sent, received = torch.zeros_like(n), torch.zeros_like(n)
            for i,(ids,s,r) in enumerate(work):
                weighted = fn(i,edges[ids],s,r)
                sent,received = _Accumulate.apply(sent,received,weighted,s,r)
            return n + layer.node(torch.cat((n,sent,received),dim=1))
        n1 = accumulate(l0,n0,groups,lambda i,e,s,r: self._run(first,n0,e,s,r,seed(i)))
        n2 = accumulate(l1,n1,final_groups,lambda i,e,s,r: self._run(
            second,n0,n1,e,s,r,seed(i),seed(len(groups)+i)))
        # The trainer gathers only declared rows. Never present an unqualified
        # non-owned row of this partial readout as a scientific prediction.
        return self.output(n2).masked_fill(~owned[:,None],float("nan"))
