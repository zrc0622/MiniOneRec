import unittest

import torch

from latent.vq import ParallelVQ, balanced_transport


class BalancedVQTests(unittest.TestCase):
    def test_transport_marginals_and_finite_extreme_costs(self):
        torch.manual_seed(42)
        costs = torch.rand(3, 64, 8) * 10000
        q = balanced_transport(costs, .05, 200)
        self.assertTrue(torch.isfinite(q).all())
        torch.testing.assert_close(q.sum(-1), torch.ones(3, 64), atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(q.sum(-2), torch.full((3, 8), 8.), atol=2e-3, rtol=2e-3)

    def test_export_independent_of_batch_and_training_has_gradients(self):
        torch.manual_seed(42)
        model = ParallelVQ(8, 4, 2, [12, 8], balance_epsilon=.05)
        x = torch.randn(16, 8)
        loss, _, _ = model(x)
        loss.backward()
        self.assertGreater(model.codebooks.grad.abs().sum().item(), 0)
        self.assertGreater(next(model.encoder.parameters()).grad.abs().sum().item(), 0)
        model.eval()
        whole = model.quantize(x)[2]
        parts = torch.cat([model.quantize(chunk)[2] for chunk in x.split(3)])
        torch.testing.assert_close(whole, parts, rtol=0, atol=0)
        torch.testing.assert_close(model(x)[2], whole)
        restored = ParallelVQ(**model.options)
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(restored.quantize(x)[2], whole)


if __name__ == '__main__':
    unittest.main()
