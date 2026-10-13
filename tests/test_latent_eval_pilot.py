import unittest
import torch
from tests.test_latent import fixture
from latent.eval_pilot import PrefixScorer, original_scores
from latent.eval_search import search_paths

class CachedScoringTests(unittest.TestCase):
    def test_cached_full_vocab_scores_match_original_variable_length_paths(self):
        torch.set_num_threads(1)
        tk, model, _, grammar=fixture()
        model.eval()
        prompt=tk.encode('history history history')
        paths=search_paths(model,prompt,grammar,'hierarchical_max',latent_count=2,item_beams=5)
        self.assertGreater(len({len(s) for s in paths}),1)
        expected=original_scores(model,prompt,paths,grammar,batch=3)
        scorer=PrefixScorer(model,prompt)
        for batch in [1,3,8]:
            actual=scorer.score(paths,batch)
            torch.testing.assert_close(torch.tensor(actual),torch.tensor(expected),atol=1e-5,rtol=1e-6)
        # Scoring must not mutate the reusable prefix cache.
        self.assertEqual(scorer.cache.get_seq_length(),len(prompt))
        torch.testing.assert_close(torch.tensor(scorer.score(paths,3)),torch.tensor(expected),atol=1e-5,rtol=1e-6)

    def test_cached_generation_matches_standard_beams_and_full_scores(self):
        from latent.eval_cached_beam import cached_item_beam
        from latent.eval_search import beam_generate
        tk,model,_,grammar=fixture()
        model.eval()
        prompt=tk.encode('history history history')
        prefix=beam_generate(model,prompt,grammar,1,'interest',beam_width=4)[0]
        scorer=PrefixScorer(model,prompt)
        expected=beam_generate(model,prompt+prefix,grammar,5,'item')
        actual,scores=cached_item_beam(model,prompt,prefix,grammar,5,scorer.cache)
        self.assertEqual(actual,expected)
        prefix_score=scorer.score([prefix])[0]
        joint=scorer.score([prefix+item for item in actual])
        torch.testing.assert_close(torch.tensor(scores)+prefix_score,torch.tensor(joint),atol=1e-5,rtol=1e-6)
        self.assertEqual(scorer.cache.get_seq_length(),len(prompt))

    def test_batched_branches_match_single_branch_full_scores(self):
        from latent.eval_cached_beam import cached_item_beam_batch,cached_item_beam
        from latent.eval_search import beam_generate
        tk,model,_,grammar=fixture()
        model.eval()
        prompt=tk.encode('history history')
        prefixes=beam_generate(model,prompt,grammar,2,'interest',beam_width=4)
        scorer=PrefixScorer(model,prompt)
        groups=cached_item_beam_batch(model,prompt,prefixes,grammar,5,scorer.cache)
        self.assertEqual(len(groups),len(prefixes))
        for prefix,(items,scores) in zip(prefixes,groups):
            expected,expected_scores=cached_item_beam(model,prompt,prefix,grammar,5,scorer.cache)
            self.assertEqual(items,expected)
            torch.testing.assert_close(torch.tensor(scores),torch.tensor(expected_scores),atol=1e-5,rtol=1e-6)
        self.assertEqual(scorer.cache.get_seq_length(),len(prompt))

if __name__=='__main__': unittest.main()
