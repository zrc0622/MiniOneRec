"""Exploratory decoding budgets and prior calibration. Does not alter training."""
import argparse,json,math,time
from collections import defaultdict
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from .core import history_records,load_index,validate_protocol,write_json,sha
from .generation import Grammar
from .eval_search import beam_generate,rank_items
from .eval_cached_beam import cached_item_beam_batch
from .eval_pilot import PrefixScorer,timed

ALPHAS={'a0':0.,'a05':.5,'a1':1.,'a2':2.}
STAGES={'p2b20':(2,20),'p4b10':(4,10),'p8b5':(8,5),'p8b10':(8,10)}

def pool_rank(grammar,prefixes,groups,priors,alpha=1.,aggregation='logsumexp'):
 pairs=[(grammar.item(prefix+seq),alpha*prior+score) for prefix,(seqs,scores),prior in zip(prefixes,groups,priors) for seq,score in zip(seqs,scores)]
 return rank_items(pairs,aggregation,10)

def rrf_rank(grammar,prefixes,groups,k):
 votes=defaultdict(float)
 for prefix,(seqs,_) in zip(prefixes,groups):
  seen=set()
  for rank,seq in enumerate(seqs,1):
   item=grammar.item(prefix+seq)
   if item not in seen:votes[item]+=1/(k+rank);seen.add(item)
 return sorted(votes,key=lambda x:(-votes[x],x))[:10]

def marginal_rank(grammar,prefixes,cross,scores,priors,alpha):
 # Joint likelihood already includes prior; replace it with alpha * prior.
 adjusted=[score+(alpha-1)*priors[i%len(prefixes)] for i,score in enumerate(scores)]
 return rank_items([(grammar.item(seq),score) for seq,score in zip(cross,adjusted)],'logsumexp',10)

def cross_paths(grammar,prefixes,groups):
 candidates={}
 for prefix,(seqs,_) in zip(prefixes,groups):
  for seq in seqs:candidates.setdefault(grammar.item(prefix+seq),seq)
 return [prefix+seq for seq in candidates.values() for prefix in prefixes]

def trial(model,tk,grammar,record):
 prompt=tk.encode(record['prompt'],add_special_tokens=False)[-512:]
 p4,t4=timed(lambda:beam_generate(model,prompt,grammar,4,'interest'))
 p8,t8=timed(lambda:beam_generate(model,prompt,grammar,8,'interest'))
 scorer,tc=timed(lambda:PrefixScorer(model,prompt))
 ps4,ts4=timed(lambda:scorer.score(p4));ps8,ts8=timed(lambda:scorer.score(p8))
 results={};details={}
 for name,(n,width) in STAGES.items():
  prefixes=p4[:n] if n<=4 else p8;priors=ps4[:n] if n<=4 else ps8
  groups,tg=timed(lambda:cached_item_beam_batch(model,prompt,prefixes,grammar,width,scorer.cache))
  setup=(t4+ts4 if n<=4 else t8+ts8)+tc+tg
  unique=len({grammar.item(p+s) for p,(seqs,_) in zip(prefixes,groups) for s in seqs})
  for tag,a in ALPHAS.items():
   for agg in ['sum','max']:
    pred,tr=timed(lambda:pool_rank(grammar,prefixes,groups,priors,a,'logsumexp' if agg=='sum' else 'max'))
    results[f'{name}_{agg}_{tag}']=dict(predict=pred,elapsed_seconds=setup+tr,unique_proposals=unique)
  for k in [20,60]:
   pred,tr=timed(lambda:rrf_rank(grammar,prefixes,groups,k))
   results[f'{name}_rrf{k}']=dict(predict=pred,elapsed_seconds=setup+tr,unique_proposals=unique)
  details[name]=dict(prefixes=prefixes,prior_logps=priors,groups=groups,unique_proposals=unique)
  if name=='p4b10':
   cross=cross_paths(grammar,prefixes,groups);scores,tx=timed(lambda:scorer.score(cross,batch=32))
   for tag,a in ALPHAS.items():
    pred,tr=timed(lambda:marginal_rank(grammar,prefixes,cross,scores,priors,a))
    results[f'{name}_marg_{tag}']=dict(predict=pred,elapsed_seconds=setup+tx+tr,unique_proposals=unique)
 for width in [50]:
  paths,tg=timed(lambda:beam_generate(model,prompt,grammar,width,'joint'))
  scores,ts=timed(lambda:scorer.score(paths,batch=32))
  for agg in ['max','sum']:
   pred,tr=timed(lambda:rank_items([(grammar.item(s),p) for s,p in zip(paths,scores)],'max' if agg=='max' else 'logsumexp',10))
   results[f'joint{width}_{agg}']=dict(predict=pred,elapsed_seconds=tg+tc+ts+tr,unique_proposals=len(set(grammar.item(s) for s in paths)))
  details[f'joint{width}']=dict(paths=paths,logps=scores)
 return results,details

def infer(model,tk,grammar,record,name):
 prompt=tk.encode(record['prompt'],add_special_tokens=False)[-512:]
 if name.startswith('joint'):
  budget,agg=name.split('_');paths=beam_generate(model,prompt,grammar,int(budget[5:]),'joint');scorer=PrefixScorer(model,prompt)
  return rank_items([(grammar.item(s),p) for s,p in zip(paths,scorer.score(paths,batch=32))],'max' if agg=='max' else 'logsumexp',10)
 stage,agg,*tags=name.split('_');n,width=STAGES[stage]
 prefixes=beam_generate(model,prompt,grammar,4 if n<=4 else 8,'interest')[:n]
 scorer=PrefixScorer(model,prompt);priors=scorer.score(prefixes)
 groups=cached_item_beam_batch(model,prompt,prefixes,grammar,width,scorer.cache)
 if agg.startswith('rrf'):return rrf_rank(grammar,prefixes,groups,int(agg[3:]))
 alpha=ALPHAS[tags[0]]
 if agg=='marg':
  cross=cross_paths(grammar,prefixes,groups)
  return marginal_rank(grammar,prefixes,cross,scorer.score(cross,batch=32),priors,alpha)
 return pool_rank(grammar,prefixes,groups,priors,alpha,'logsumexp' if agg=='sum' else 'max')

def main():
 p=argparse.ArgumentParser()
 for key in ['model','data','index','sample','output']:p.add_argument('--'+key,required=True)
 p.add_argument('--mode',choices=['screen','confirm'],default='screen');p.add_argument('--methods',nargs='+')
 o=p.parse_args();torch.set_num_threads(1)
 tk=AutoTokenizer.from_pretrained(o.model);tk.pad_token=tk.eos_token
 model=AutoModelForCausalLM.from_pretrained(o.model,torch_dtype=torch.bfloat16).cuda().eval()
 index=load_index(o.index);grammar=Grammar(tk,index,validate_protocol(model.config,tk,index));records=history_records(o.data,index,grammar.k)
 ids=json.loads(Path(o.sample).read_text())['row_ids'];rows=defaultdict(list);details={}
 if o.mode=='screen':trial(model,tk,grammar,records[ids[0]])
 else:
  for method in o.methods:infer(model,tk,grammar,records[ids[0]],method)
 start=time.perf_counter()
 for j,rid in enumerate(ids):
  if o.mode=='screen':result,details[str(rid)]=trial(model,tk,grammar,records[rid])
  else:
   result={};order=o.methods[j%len(o.methods):]+o.methods[:j%len(o.methods)]
   for method in order:
    pred,t=timed(lambda:infer(model,tk,grammar,records[rid],method));result[method]=dict(predict=pred,elapsed_seconds=t)
  for method,r in result.items():rows[method].append(dict(row_id=rid,output=records[rid]['target'],**r))
  write_json(o.output,dict(rows=dict(rows),proposal_details=details,metadata=dict(model=o.model,mode=o.mode,examples=len(ids),sample_sha256=sha(o.sample),data_sha256=sha(o.data),index_sha256=sha(o.index),source_sha256=sha(__file__),attention=model.config._attn_implementation,timing='screen composed cached stages + measured ranking; confirm independent fullrequest, CUDA sync, warmup excluded')))
  print(json.dumps(dict(done=j+1,total=len(ids),wall_seconds=time.perf_counter()-start)),flush=True)
if __name__=='__main__':main()
