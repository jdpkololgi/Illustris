import unittest
import torch

from workflows.abacus_tweb.lowz_graph_attention import BoundedGraphPatchNet
from workflows.abacus_tweb.p8_train_graph_patch import GraphPatchNet


class BoundedGraphAttentionTests(unittest.TestCase):
    def compare(self, dropout, training, reference_class):
        torch.manual_seed(42)
        a = reference_class(latent_size=8, heads=2, dropout=dropout).double()
        b = BoundedGraphPatchNet(latent_size=8, heads=2, dropout=dropout).double()
        a.chunk_edges = b.chunk_edges = 4
        a.rematerialize = False
        b.load_state_dict(a.state_dict(), strict=True)
        # Receiver 0 occurs on both sides of chunk boundaries; last chunk is partial.
        s = torch.tensor([1,2,3,0,4,2,0,1,3,2,4])
        r = torch.tensor([0,1,4,3,0,3,1,2,0,4,1])
        data = (torch.randn(5,8,dtype=torch.float64), torch.randn(len(s),5,dtype=torch.float64))
        def run(m):
            m.train(training)
            x, e = [t.detach().clone().requires_grad_() for t in data]
            torch.manual_seed(17)
            out = m(x,e,s,r)
            out.square().mean().backward()
            return [out, x.grad, e.grad, *(p.grad for p in m.parameters())], torch.get_rng_state()
        lhs, la = run(a)
        rhs, lb = run(b)
        for x, y in zip(lhs, rhs, strict=True):
            self.assertIsNotNone(x)
            self.assertIsNotNone(y)
            torch.testing.assert_close(x,y,rtol=1e-9,atol=1e-11)
        torch.testing.assert_close(la,lb)

    def test_reference_training_no_dropout(self):
        self.compare(0., True, GraphPatchNet)

    def test_reference_eval(self):
        self.compare(.1, False, GraphPatchNet)

    def test_checkpoint_dropout_rng_and_gradient_replay(self):
        self.compare(.1, True, BoundedGraphPatchNet)
