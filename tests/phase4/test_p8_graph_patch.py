import unittest

import torch

from workflows.abacus_tweb.p8_train_graph_patch import (
    AttentionPass,
    GraphPatchNet,
)


class TestP8GraphPatch(unittest.TestCase):
    def test_receiver_softmax_sums_to_one(self):
        logits = torch.tensor([[1.0, 2.0], [2.0, 1.0], [-3.0, 4.0]])
        receivers = torch.tensor([1, 1, 2])
        weight = AttentionPass.receiver_softmax(logits, receivers, 3)
        self.assertTrue(torch.allclose(weight[:2].sum(0), torch.ones(2)))
        self.assertTrue(torch.allclose(weight[2], torch.ones(2)))

    def test_output_shape_and_finite_gradients(self):
        torch.manual_seed(5)
        model = GraphPatchNet(latent_size=16, heads=4, dropout=0.0)
        nodes = torch.randn(6, 8)
        senders = torch.tensor([0, 1, 2, 3, 4, 5, 1, 2])
        receivers = torch.tensor([1, 2, 3, 4, 5, 0, 0, 1])
        edges = torch.randn(len(senders), 5)
        output = model(nodes, edges, senders, receivers)
        self.assertEqual(output.shape, (6, 3))
        output.square().mean().backward()
        self.assertTrue(
            all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
