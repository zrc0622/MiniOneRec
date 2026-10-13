"""Verify evaluation search with a real tiny Qwen and exact score examples."""
import math
import unittest

import torch

from tests.test_latent import fixture
from latent.eval_search import beam_generate, rank_items, search_paths
from latent.generation import full_logps, pack_trajectories


class EvalSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_probability_sum_can_reverse_max_ranking(self):
        paths = [('A', math.log(.2)), ('A', math.log(.2)), ('B', math.log(.3))]
        self.assertEqual(rank_items(paths, 'max'), ['B', 'A'])
        self.assertEqual(rank_items(paths, 'logsumexp'), ['A', 'B'])
        self.assertEqual(rank_items([('A', -1001), ('A', -1001), ('B', -1000.5)], 'logsumexp'), ['A', 'B'])

    def test_model_defaults_and_all_search_modes(self):
        tk, model, _, grammar = fixture()
        model.eval()
        model.generation_config.do_sample = True
        model.generation_config.temperature = .6
        model.generation_config.top_p = .95
        prompt = tk.encode('history')
        results = {}
        for scheme in ('joint_max', 'hierarchical_max', 'hierarchical_sum', 'fixed_latent'):
            samples = search_paths(model, prompt, grammar, scheme, beams=8,
                                   latent_count=2, item_beams=5, latent_beams=4)
            self.assertEqual(len(samples), len({tuple(s) for s in samples}))
            for sample in samples:
                grammar.item(sample)
            packed = pack_trajectories([prompt] * len(samples), samples,
                                      [3] * len(samples), [False] * len(samples), grammar, 'cpu')
            with torch.no_grad():
                logps = full_logps(model, packed)
            self.assertTrue(torch.isfinite(logps).all())
            results[scheme] = samples
        self.assertEqual(results['hierarchical_max'], results['hierarchical_sum'])
        self.assertEqual(len({tuple(s[:3]) for s in results['fixed_latent']}), 1)
        self.assertEqual(len({grammar.item(s) for s in results['fixed_latent']}), 5)
        self.assertEqual(results['joint_max'], search_paths(model, prompt, grammar, 'joint_max', beams=8))

    def test_fixed_latent_uses_best_complete_beam_prefix(self):
        tk, model, _, grammar = fixture()
        model.eval()
        prompt = tk.encode('history')
        prefixes = beam_generate(model, prompt, grammar, 4, phase='interest')
        samples = search_paths(model, prompt, grammar, 'fixed_latent', item_beams=5, latent_beams=4)
        self.assertTrue(all(s[:3] == prefixes[0] for s in samples))


if __name__ == '__main__':
    unittest.main()
