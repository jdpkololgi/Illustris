"""Bound dense edge activations without pruning edges or localizing receiver softmax."""
import torch
from torch.utils.checkpoint import checkpoint

from workflows.abacus_tweb.p8_train_graph_patch import GraphPatchNet


class BoundedGraphPatchNet(GraphPatchNet):
    # A single FP32 [E,3*80] concatenation at this size is 60 MiB.
    chunk_edges = 65536
    rematerialize = True

    def _run(self, fn, *args):
        if self.training and torch.is_grad_enabled() and self.rematerialize:
            return checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=True)
        return fn(*args)

    def _pass(self, layer, nodes, edges, senders, receivers):
        # SplitBackward collects edge gradients once, rather than repeatedly
        # materializing a full-patch gradient for each indexed slice.
        width = self.chunk_edges
        e_chunks = edges.split(width)
        s_chunks, r_chunks = senders.split(width), receivers.split(width)

        def messages(n, e, s, r):
            src, dst = n[s], n[r]
            updated = layer.edge(torch.cat((e, src, dst), dim=1))
            inp = torch.cat((updated, src, dst), dim=1)
            logits = torch.cat([head(inp) for head in layer.attention], dim=1)
            return updated, logits

        chunks = [self._run(messages, nodes, e, s, r)
                  for e, s, r in zip(e_chunks, s_chunks, r_chunks)]
        logits = torch.cat([v[1] for v in chunks])
        # Edges for a receiver can be in different chunks. Normalize over
        # all incident directed edges before either message aggregation.
        weights = layer.receiver_softmax(logits, receivers, len(nodes))
        weighted = torch.cat([(u.reshape(-1, layer.heads, layer.head_size)
                               * w.unsqueeze(-1)).reshape(-1, layer.latent_size)
                              for (u, _), w in zip(chunks, weights.split(width))])
        sent = layer.segment_sum(weighted, senders, len(nodes))
        received = layer.segment_sum(weighted, receivers, len(nodes))
        return nodes + layer.node(torch.cat((nodes, sent, received), dim=1)), edges + weighted

    def forward(self, nodes, edges, senders, receivers):
        if len(edges) <= self.chunk_edges:
            return super().forward(nodes, edges, senders, receivers)
        if len(edges) >= 2 * getattr(self, "_logged_edge_scale", 0):
            print(f"G-PATCH dense forward: training={self.training} nodes={len(nodes)} "
                  f"directed_edges={len(edges)} chunk_edges={self.chunk_edges}", flush=True)
            self._logged_edge_scale = len(edges)
        n, e = self.node_embed(nodes), self.edge_embed(edges)
        for layer in self.passes:
            # Bind the layer now so checkpoint recomputation uses the same weights.
            def one_pass(n, e, s, r, layer=layer):
                return self._pass(layer, n, e, s, r)
            n, e = self._run(one_pass, n, e, senders, receivers)
        return self.output(n)
