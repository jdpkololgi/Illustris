import unittest
import torch

from workflows.abacus_tweb.lowz_graph_streaming import StreamingGraphPatchNet, receiver_groups
from workflows.abacus_tweb.p8_train_graph_patch import GraphPatchNet


class ReceiverStreamingTest(unittest.TestCase):
    def compare(self, dtype, training, every=False, dropout=0.):
        torch.manual_seed(83)
        a = (StreamingGraphPatchNet if dropout and training else GraphPatchNet)(
            latent_size=8,heads=2,dropout=dropout).to(dtype)
        b = StreamingGraphPatchNet(latent_size=8,heads=2,dropout=dropout).to(dtype)
        a.train(training); b.train(training)
        a.stream_threshold = b.stream_threshold = 0
        a.chunk_edges = b.chunk_edges = 4
        a.rematerialize = False
        b.load_state_dict(a.state_dict(),strict=True)
        # Asymmetric edges, repeated edges, bridge, isolated node and a
        # disconnected component. Receiver 1 alone exceeds the chunk width.
        s = torch.tensor([3,1,3,0,2,4,3,4,0,5,6,7,8,9,10,11,12,13,14])
        r = torch.tensor([1,4,1,1,1,1,0,2,1,2,5,6,7,8,1,12,13,14,11])
        mask = torch.ones(16,dtype=torch.bool) if every else torch.arange(16)%16==0
        mask[-1] = True
        data = (torch.randn(16,8,dtype=dtype),torch.randn(len(s),5,dtype=dtype))
        def run(m):
            x,e=[v.clone().requires_grad_() for v in data]
            torch.manual_seed(312)
            out = (m(x,e,s,r,mask) if isinstance(m,StreamingGraphPatchNet) else m(x,e,s,r))
            if isinstance(m,StreamingGraphPatchNet):
                self.assertTrue(torch.isnan(out[~mask]).all())
            out = out[mask]
            out.square().mean().backward()
            return [out,x.grad,e.grad,*[p.grad for p in m.parameters()]], torch.get_rng_state()
        lhs,ra=run(a); rhs,rb=run(b)
        self.assertEqual(len(lhs),len(rhs))
        for x,y in zip(lhs,rhs):
            self.assertIsNotNone(x); self.assertIsNotNone(y)
            tol=(1e-9,1e-11) if dtype==torch.float64 else (2e-5,2e-6)
            torch.testing.assert_close(x,y,rtol=tol[0],atol=tol[1])
        torch.testing.assert_close(ra,rb)

    def test_partial_training_matches_original(self):
        self.compare(torch.float64,True)

    def test_full_fp32_matches_original(self):
        self.compare(torch.float32,True,every=True)

    def test_partial_eval_matches_original(self):
        self.compare(torch.float64,False,dropout=.1)

    def test_checkpoint_matches_chunk_dropout_and_rng(self):
        self.compare(torch.float64,True,dropout=.1)

    def test_chunk_owns_entire_receiver(self):
        s=torch.arange(19); r=s%3; keep=torch.tensor([True,False,True])
        chunks=receiver_groups(s,r,keep,4)
        ids=torch.cat([g[0] for g in chunks]).sort().values
        torch.testing.assert_close(ids,torch.where(keep[r])[0])
        for _,_,dest in chunks:
            for d in dest.unique():
                self.assertEqual(int((dest==d).sum()),int((r==d).sum()))

    def test_no_patch_wide_edge_latent_in_forward_or_backward(self):
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.utils._pytree import tree_leaves
        class Observe(TorchDispatchMode):
            def __init__(self):
                super().__init__()
                self.shapes=set()
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                result=func(*args,**(kwargs or {}))
                self.shapes.update(tuple(t.shape) for t in tree_leaves(result)
                                   if isinstance(t,torch.Tensor) and t.is_floating_point())
                return result
        torch.manual_seed(17)
        model=StreamingGraphPatchNet(latent_size=8,heads=2,dropout=0.).train()
        model.stream_threshold=0
        model.chunk_edges=8
        s=torch.randint(17,(80,)); r=torch.arange(80)%17
        x=torch.rand(17,8); e=torch.rand(80,5)
        with Observe() as ops:
            model(x,e,s,r).square().mean().backward()
        self.assertIn((17,8),ops.shapes)
        self.assertFalse([sh for sh in ops.shapes if len(sh)>=2 and sh[0]==80
                          and torch.Size(sh[1:]).numel()>=8])
