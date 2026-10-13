import unittest
import torch
from tests.test_latent import fixture
from latent.eval_unified_small import BatchPrefixScorer,padded,batch_latent,batch_sid
from latent.eval_pilot import PrefixScorer
from latent.eval_at20 import infer
from latent.eval_search import beam_generate

class UnifiedEvalTests(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(1);self.tk,self.model,_,self.g=fixture();self.model.eval();self.prompts=[self.tk.encode('history <a_0> <b_0>'),self.tk.encode('history <a_3> <b_0> <c_0> history')]
 def test_cross_user_full_likelihood_variable_sid_lengths(self):
  ids,mask=padded(self.prompts,self.g.eos,torch.device('cpu'))
  with torch.no_grad():out=self.model(input_ids=ids,attention_mask=mask,position_ids=(mask.cumsum(-1)-1).clamp_min(0),use_cache=True,logits_to_keep=1)
  first=out.logits[:,-1].float();first=first-torch.logsumexp(first,-1,keepdim=True);scorer=BatchPrefixScorer(self.model,out.past_key_values,mask,first)
  seqs=[];users=[];expected=[]
  for u,p in enumerate(self.prompts):
   prefix=beam_generate(self.model,p,self.g,1,'interest',beam_width=4)[0];items=beam_generate(self.model,p+prefix,self.g,10,'item');paths=[prefix+s for s in items]
   seqs+=paths;users +=[u]*len(paths);expected+=PrefixScorer(self.model,p).score(paths)
  self.assertGreater(len({len(s) for s in seqs}),1)
  torch.testing.assert_close(torch.tensor(scorer.score(seqs,users,3)),torch.tensor(expected),atol=1e-5,rtol=1e-6)
  self.assertEqual(scorer.cache.get_seq_length(),ids.shape[1])
 def test_both_protocols_match_existing_individual_fp32(self):
  for method in ['single20','four20_marginal']:
   rs=batch_latent(self.model,self.g,self.prompts,method)
   for prompt,r in zip(self.prompts,rs):
    text=self.tk.decode(prompt,skip_special_tokens=False);expected=infer(self.model,self.tk,self.g,dict(prompt=text),method)
    self.assertEqual(r['predict'],expected)
 def test_cached_native_sid_matches_individual_beams(self):
  rows=batch_sid(self.model,self.g,self.prompts,20)
  for p,r in zip(self.prompts,rows):
   expected=beam_generate(self.model,p,self.g,20,'item');items=[self.g.sid_by_ids[False][tuple(s)] for s in expected]
   self.assertEqual(r['predict'],items[:20])
if __name__=='__main__':unittest.main()
