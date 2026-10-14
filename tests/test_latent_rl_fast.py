"""Distribution, gradient and 4x4 catalog checks for isolated latent speed code."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from datasets import Dataset
from interest.train import InterestRLArguments

from tests.test_latent import fixture
from latent.generation import constrained_logps, pack_trajectories
from latent.rl_fast import FastLatentGRPOTrainer, cached_rollouts, restricted_logps, vectorized_logps
from latent.trainer import LatentGRPOTrainer


class FastRLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_cached_variable_length_prompts_match_fresh_conditional_distribution(self):
        tk, model, protocol, grammar = fixture()
        model.eval()
        prompts = [[2] * 3, [2] * 8]
        observed = []

        def audit(logits, allowed, users, prefixes):
            for i, (user, prefix, legal) in enumerate(zip(users, prefixes, allowed)):
                seq = prompts[user] + prefix
                ref = model(torch.tensor([seq]), use_cache=False).logits[0, -1]
                expected = ref[legal].float().log_softmax(-1)
                actual = logits[i, legal].float().log_softmax(-1)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
            observed.append(len(users))

        for auxiliary in (False, 'identification', 'title'):
            samples = cached_rollouts(model, prompts, grammar, auxiliary, .7, audit)
            self.assertEqual([len(s) for s in samples], [16, 16])
            for group in samples:
                for seq in group:
                    grammar.item(seq, auxiliary)
                if not auxiliary:
                    for start in range(0, 16, 4):
                        self.assertTrue(all(s[:grammar.k] == group[start][:grammar.k] for s in group[start:start + 4]))
        self.assertIn(8, observed)
        self.assertIn(32, observed)

    def test_constrained_logprobs_and_model_gradients_match_original(self):
        tk, model, protocol, grammar = fixture()
        model.eval()
        prompts = [[2] * 3, [2] * 8]
        samples = cached_rollouts(model, prompts, grammar)
        flat = [s for group in samples for s in group]
        packed = pack_trajectories([p for p in prompts for _ in range(16)], flat,
                                  [3] * 32, [False] * 32, grammar, 'cpu')
        original = constrained_logps(model, packed, .7)
        weights = torch.randn_like(original)
        (original * weights).sum().backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        fast = vectorized_logps(model, packed, .7)
        torch.testing.assert_close(fast, original, rtol=1e-6, atol=1e-6)
        (fast * weights).sum().backward()
        for n, p in model.named_parameters():
            if n in grads:
                torch.testing.assert_close(p.grad, grads[n], rtol=2e-5, atol=3e-6)

    def test_ragged_allowed_logits_value_and_gradient(self):
        logits = torch.randn(2, 3, 11, requires_grad=True)
        sets = [[[1], [2, 4], [1, 3, 5]], [[0, 2, 8], [3], [5, 7]]]
        pos = [[0, 1, 2], [1, 0, 0]]
        packed = dict(allowed_sets=sets, action_positions=pos)
        ref = torch.stack([torch.stack([logits[b, t, s].float()[pos[b][t]] / .8 -
                torch.logsumexp(logits[b, t, s].float() / .8, 0) for t, s in enumerate(row)])
                for b, row in enumerate(sets)])
        fast = restricted_logps(logits, packed, .8)
        torch.testing.assert_close(fast, ref)
        torch.testing.assert_close(torch.autograd.grad(fast.sum(), logits, retain_graph=True)[0],
                                   torch.autograd.grad(ref.sum(), logits)[0])

    def test_identical_rollouts_preserve_rewards_grouping_loss_and_gradients(self):
        tk, model, protocol, grammar = fixture()
        model.eval()
        records = [dict(prompt='history ' * n, target='<a_0><b_0><c_0>', auxiliary=aux, task=task)
                   for n, aux, task in ((3, False, 'history_sid'), (8, False, 'history_sid'),
                                        (4, True, 'item_identification'), (6, True, 'history_title'))]
        encoded = [tk.encode(r['prompt'], add_special_tokens=False) for r in records]
        auxes = [False, False, 'identification', 'title']
        samples = [cached_rollouts(model, [p], grammar, a)[0] for p, a in zip(encoded, auxes)]
        lookup = {(tuple(p), a): s for p, a, s in zip(encoded, auxes, samples)}
        inputs = [r for r in records for _ in range(16)]
        losses, grads, packs = [], [], []
        with tempfile.TemporaryDirectory() as temp:
            for cls in (LatentGRPOTrainer, FastLatentGRPOTrainer):
                args = InterestRLArguments(output_dir=str(Path(temp) / cls.__name__), use_cpu=True,
                    per_device_train_batch_size=64, per_device_eval_batch_size=64,
                    report_to=[], remove_unused_columns=False, prediction_loss_only=True)
                m = copy.deepcopy(model)
                trainer = cls(model=m, args=args, processing_class=tk, grammar=grammar, mode='4x4',
                    reward_kind='sid_dense', train_dataset=Dataset.from_list(records), eval_dataset=Dataset.from_list(records))
                with patch('latent.trainer.rollout', side_effect=lambda model, prompt, grammar, mode, auxiliary, temperature:
                           lookup[tuple(prompt), auxiliary]), patch('latent.rl_fast.cached_rollouts',
                           side_effect=lambda model, prompts, grammar, aux, temperature:
                           [lookup[tuple(p), aux] for p in prompts]):
                    packed = trainer._prepare_inputs(inputs)
                loss = trainer.compute_loss(m, packed)
                loss.backward()
                losses.append(loss.detach())
                grads.append({n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None})
                packs.append(packed)
        for key in ('input_ids', 'attention_mask', 'completion_ids', 'interest_mask', 'item_mask', 'ref_logps'):
            torch.testing.assert_close(packs[0][key], packs[1][key])
        for a, b in zip(packs[0]['group_specs'], packs[1]['group_specs']):
            self.assertEqual(a[:3], b[:3])
            torch.testing.assert_close(a[3], b[3])
            torch.testing.assert_close(a[4], b[4])
        torch.testing.assert_close(losses[0], losses[1], atol=1e-6, rtol=1e-6)
        for name in grads[0]:
            torch.testing.assert_close(grads[0][name], grads[1][name], atol=2e-6, rtol=2e-5)


if __name__ == '__main__':
    unittest.main()
