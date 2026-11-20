import unittest
from types import SimpleNamespace

import numpy as np
import torch

from workflows.abacus_tweb.lowz_graph_host_cone import owned_patch
from workflows.abacus_tweb.lowz_host_cone_gate import compare_arrays, comparison
from workflows.abacus_tweb.p5_graph_patch_utils import assemble_patch
from workflows.abacus_tweb.p8_train_graph_patch import GraphPatchNet


class HostConeTest(unittest.TestCase):
    def setUp(self):
        pairs=np.array([(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                        (4,8),(3,7),(8,10),(10,11)])
        inc=[np.flatnonzero(np.any(pairs==i,axis=1)) for i in range(12)]
        rng=np.random.default_rng(42)
        self.a=SimpleNamespace(union_pairs=pairs,union_edge_features=rng.uniform(.1,2,(len(pairs),5)).astype("f"),
                               offsets=np.cumsum([0]+[len(q) for q in inc]),incident_edge_id=np.concatenate(inc),
                               node_features=rng.normal(size=(12,8)).astype("f"),core_cap=np.array([0,0,1]),
                               core_nodes=lambda c:(np.array([0 if c==0 else 9]),None))

    def test_owned_rows_features_outputs_and_gradients(self):
        a=self.a
        g=owned_patch(a,[0,1],block=2)
        ref=assemble_patch(core_id=-1,fold=-1,core_parent_ids=np.array([0,9]),loss_parent_ids=np.array([0,9]),
                           strict_parent_ids=np.empty(0,int),num_passes=2,dependency_hops=4,
                           node_features=a.node_features,union_pairs=a.union_pairs,
                           union_edge_features=a.union_edge_features,offsets=a.offsets,incident_edge_id=a.incident_edge_id)
        np.testing.assert_equal(g.parent_node_id,ref.parent_node_id)
        np.testing.assert_equal(g.loss_mask,ref.loss_mask)
        compare_arrays(ref,g)
        # Independent expected closure: only 4 and 8 lie at distance four.
        keep=~np.isin(ref.parent_node_id[ref.receivers],[4,8])
        self.assertLess(g.n_edge,ref.n_edge)
        for name in ("union_edge_id","edge_features","senders","receivers"):
            np.testing.assert_equal(getattr(g,name),getattr(ref,name)[keep])
        torch.manual_seed(17)
        m=GraphPatchNet(latent_size=8,heads=2,dropout=0.).double()
        def run(p):
            m.zero_grad(set_to_none=True)
            x,e=[torch.tensor(t,dtype=torch.float64,requires_grad=True) for t in (p.node_features,p.edge_features)]
            s,r=[torch.tensor(t,dtype=torch.long) for t in (p.senders,p.receivers)]
            out=m(x,e,s,r)[p.loss_mask]
            out.square().sum().backward()
            return out.detach(),x.grad,e.grad,[z.grad.clone() for z in m.parameters()]
        y,x,e,w=run(ref); yy,xx,ee,ww=run(g)
        for t,v in zip([y,x,e[keep],*w],[yy,xx,ee,*ww],strict=True):
            torch.testing.assert_close(t,v,rtol=1e-10,atol=1e-12)
        self.assertEqual(torch.count_nonzero(e[~keep]).item(),0)

    def test_isolated_owned_and_cap_guard(self):
        g=owned_patch(self.a,[1])
        self.assertEqual(g.n_edge,0)
        np.testing.assert_equal(g.parent_node_id,[9])
        with self.assertRaisesRegex(ValueError,"caps"):
            owned_patch(self.a,[0,2])

    def test_comparison_rejects_row_substitution(self):
        group=dict(phase="ph006",cap=1,cores=[2,4],rows=5,nodes=10,largest_owned=False,edges=32,seconds=1.)
        ref=dict(seed=42,source_step=256,training_window=[64,256],training_cores_per_epoch=346364,
                 validation_samples=[dict(group) for _ in range(24)],updates_per_second=.4)
        obs=dict(ref,updates_per_second=.8,validation_samples=[dict(group,edges=20) for _ in range(24)])
        self.assertEqual(comparison(obs,ref)["training_speed_ratio"],2.)
        obs["validation_samples"][1]["cores"]=[3,4]
        with self.assertRaisesRegex((RuntimeError,ValueError),"different inference workload"):
            comparison(obs,ref)
