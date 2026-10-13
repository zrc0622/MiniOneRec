import unittest
import torch
from tests.test_latent import fixture
from latent.eval_batched_high_budget import batch_infer
from latent.eval_at20 import infer
from latent.eval_pilot import PrefixScorer
from latent.eval_search import beam_generate

class MultiUserBatchTest(unittest.TestCase):
 def test_leftpad_cache_positions_match_individual_requests(self):
  torch.set_num_threads(1);tk,model,_,grammar=fixture();model.eval()
  prompts=[tk.encode('history history'),tk.encode('history history history history history')]
  together=batch_infer(model,grammar,prompts,2,5)
  individual=[batch_infer(model,grammar,[p],2,5)[0] for p in prompts]
  for a,b in zip(together,individual):
   self.assertEqual(a['predict'],b['predict'])
   self.assertEqual(a['generated_paths'],b['generated_paths'])
   self.assertEqual(a['unique_before_top20'],b['unique_before_top20'])
   self.assertAlmostEqual(a['selected_latent_mass'],b['selected_latent_mass'],places=6)
 def test_batched_cached_sid_scores_match_full_teacher_forcing(self):
  torch.set_num_threads(1);tk,model,_,grammar=fixture();model.eval()
  prompts=[tk.encode('history <a_0> <b_0>'),tk.encode('history <a_3> <b_0> <c_0> history')]
  captured=[];original=model.generate
  def wrapped(*args,**kwargs):
   out=original(*args,**kwargs);captured.append(out);return out
  model.generate=wrapped
  batch_infer(model,grammar,prompts,2,5)
  interest,items=captured;plen=max(map(len,prompts));prefixes=interest.sequences[:,plen:].tolist()
  sequences=items.sequences[:,plen+3:].tolist()
  for user,prompt in enumerate(prompts):
   scorer=PrefixScorer(model,prompt);paths=[];scores=[]
   for branch in range(2):
    idx=user*2+branch
    for j in range(idx*5,(idx+1)*5):
     seq=sequences[j];seq=seq[:seq.index(grammar.eos)+1]
     paths.append(prefixes[idx]+seq);scores.append(interest.sequences_scores[idx]+items.sequences_scores[j])
   torch.testing.assert_close(torch.stack(scores),torch.tensor(scorer.score(paths)),atol=1e-5,rtol=1e-6)
 def test_interest_beam_scores_are_full_vocab_joint_probabilities(self):
  from latent.eval_batched_high_budget import cfg
  tk,model,_,grammar=fixture();model.eval();prompt=tk.encode('history history')
  ids=torch.tensor([prompt]);out=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),generation_config=cfg(grammar,2,3),use_model_defaults=False,do_sample=False,prefix_allowed_tokens_fn=lambda _,t:grammar.allowed(t[len(prompt):].tolist(),'interest'))
  prefixes=out.sequences[:,len(prompt):].tolist();full=PrefixScorer(model,prompt).score(prefixes)
  torch.testing.assert_close(out.sequences_scores,torch.tensor(full),atol=1e-5,rtol=1e-6)
if __name__=='__main__':unittest.main()
