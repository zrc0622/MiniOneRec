"""Validation-only comparison of decoding budgets and item aggregations."""
import argparse,csv,json,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from data import EvalSidDataset
from interest.evaluate import summarize
from .core import history_records,load_index,validate_protocol,write_json,sha
from .generation import Grammar
from .eval_search import beam_generate,rank_items
from .eval_cached_beam import cached_item_beam_batch
from .eval_pilot import PrefixScorer,timed

METHODS={
 'single10':(1,10,'max'),'single20':(1,20,'max'),'single50':(1,50,'max'),
 'two10_max':(2,10,'max'),'two10_sum':(2,10,'logsumexp'),'two10_marginal':(2,10,'marginal'),
 'four10_max':(4,10,'max'),'four10_sum':(4,10,'logsumexp'),'four10_marginal':(4,10,'marginal'),
 'four20_sum':(4,20,'logsumexp'),'budget80_sum':(4,'asymmetric','logsumexp')}


def flatten(prefixes,groups,latent_scores):
 paths=[];scores=[]
 for prefix,(items,probs),prior in zip(prefixes,groups,latent_scores):
  paths.extend(prefix+item for item in items);scores.extend(prior+s for s in probs)
 return paths,scores


def ranked(grammar,paths,scores,aggregation,top_k=10):
 return rank_items([(grammar.item(s),p) for s,p in zip(paths,scores)],aggregation,limit=top_k)


def cross_scores(scorer,grammar,prefixes,paths):
 candidates={}
 for seq in paths:candidates.setdefault(grammar.item(seq),seq[grammar.k:])
 cross=[prefix+item for item in candidates.values() for prefix in prefixes]
 return cross,scorer.score(cross,batch=32)


def screen(model,tk,grammar,record,top_k=10):
 prompt=tk.encode(record['prompt'],add_special_tokens=False)[-512:]
 prefixes,pt=timed(lambda:beam_generate(model,prompt,grammar,4,'interest'))
 scorer,ct=timed(lambda:PrefixScorer(model,prompt))
 ps,pst=timed(lambda:scorer.score(prefixes))
 setup=pt+ct+pst;stages={}
 for n,width in [(1,10),(1,20),(1,50),(2,10),(4,10),(4,20),(3,10)]:
  which=prefixes[1:] if n==3 else prefixes[:n]
  group,t=timed(lambda:cached_item_beam_batch(model,prompt,which,grammar,width,scorer.cache))
  stages[n,width]=(group,t)
 results={}
 for name,(n,width,aggregation) in METHODS.items():
  if width=='asymmetric':
   first,t1=stages[1,50];tail,t2=stages[3,10]
   paths,scores=flatten(prefixes,first+tail,ps);elapsed=setup+t1+t2
  else:
   groups,t=stages[n,width];paths,scores=flatten(prefixes[:n],groups,ps[:n]);elapsed=setup+t
  if aggregation=='marginal':
   (paths,scores),extra=timed(lambda:cross_scores(scorer,grammar,prefixes[:n],paths));elapsed+=extra
  results[name]=dict(predict=ranked(grammar,paths,scores,'logsumexp' if aggregation=='marginal' else aggregation,top_k),elapsed_seconds=elapsed,
      generated_paths=n*width if isinstance(width,int) else 80,scored_paths=len(paths))
 paths,jt=timed(lambda:beam_generate(model,prompt,grammar,10,'joint'))
 scores,st=timed(lambda:scorer.score(paths))
 results['joint10']=dict(predict=ranked(grammar,paths,scores,'max',top_k),elapsed_seconds=jt+ct+st,generated_paths=len(paths),scored_paths=len(paths))
 return results


def infer(model,tk,grammar,record,name,top_k=10):
 prompt=tk.encode(record['prompt'],add_special_tokens=False)[-512:]
 if name=='joint10':
  paths=beam_generate(model,prompt,grammar,10,'joint');scorer=PrefixScorer(model,prompt)
  return ranked(grammar,paths,scorer.score(paths),'max',top_k)
 n,width,aggregation=METHODS[name]
 prefixes=beam_generate(model,prompt,grammar,4,'interest')
 scorer=PrefixScorer(model,prompt);ps=scorer.score(prefixes)
 if width=='asymmetric':
  groups=cached_item_beam_batch(model,prompt,prefixes[:1],grammar,50,scorer.cache)+cached_item_beam_batch(model,prompt,prefixes[1:],grammar,10,scorer.cache)
 else:groups=cached_item_beam_batch(model,prompt,prefixes[:n],grammar,width,scorer.cache)
 paths,scores=flatten(prefixes[:n],groups,ps[:n])
 if aggregation=='marginal':paths,scores=cross_scores(scorer,grammar,prefixes[:n],paths)
 return ranked(grammar,paths,scores,'logsumexp' if aggregation=='marginal' else aggregation,top_k)


def main():
 p=argparse.ArgumentParser()
 for key in ['model','data','index','sample','output']:p.add_argument('--'+key,required=True)
 p.add_argument('--mode',choices=['screen','confirm'],default='screen')
 p.add_argument('--top-k',type=int,default=10)
 p.add_argument('--methods',nargs='+');p.add_argument('--no-latent',action='store_true')
 o=p.parse_args();torch.set_num_threads(1)
 if o.top_k<1:p.error('--top-k must be positive')
 ids=json.loads(Path(o.sample).read_text())['row_ids'];index=load_index(o.index)
 tk=AutoTokenizer.from_pretrained(o.model);tk.pad_token=tk.eos_token
 model=AutoModelForCausalLM.from_pretrained(o.model,torch_dtype=torch.bfloat16).cuda().eval()
 if o.no_latent:
  grammar=Grammar(tk,index,dict(tokens=[[],[],[]]))
  with open(o.data) as f:source=list(csv.DictReader(f))
  path=Path(o.output).with_suffix('.csv')
  with path.open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(source[0]));w.writeheader();w.writerows(source[i] for i in ids)
  dataset=EvalSidDataset(str(path),tk,max_len=2560,test=True,K=0,seed=42)
  names=o.methods or ['direct10','direct20','direct50']
  def direct(i,name):
   seqs=beam_generate(model,dataset[i]['input_ids'],grammar,int(name.removeprefix('direct')),'item')
   return [grammar.sid_by_ids[False][tuple(s)] for s in seqs][:o.top_k]
  for name in names:direct(0,name)
 else:
  grammar=Grammar(tk,index,validate_protocol(model.config,tk,index));records=history_records(o.data,index,grammar.k)
  names=o.methods or list(METHODS)+['joint10']
  if o.mode=='screen':screen(model,tk,grammar,records[ids[0]],o.top_k)
  else:
   for name in names:infer(model,tk,grammar,records[ids[0]],name,o.top_k)
 rows=defaultdict(list);start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
 for j,rid in enumerate(ids):
  if o.no_latent:
   result={}
   order=names[j%len(names):]+names[:j%len(names)]
   for name in order:
    pred,t=timed(lambda:direct(j,name));result[name]=dict(predict=pred,elapsed_seconds=t)
   target=source[rid]['item_sid']
  else:
   target=records[rid]['target']
   if o.mode=='screen':result=screen(model,tk,grammar,records[rid],o.top_k)
   else:
    result={};order=names[j%len(names):]+names[:j%len(names)]
    for name in order:
     pred,t=timed(lambda:infer(model,tk,grammar,records[rid],name,o.top_k));result[name]=dict(predict=pred,elapsed_seconds=t)
  for name,r in result.items():rows[name].append(dict(row_id=rid,output=target,**r))
  write_json(o.output,dict(rows=dict(rows),metadata=dict(model=o.model,mode=o.mode,examples=len(ids),top_k=o.top_k,sample_sha256=sha(o.sample),data_sha256=sha(o.data),index_sha256=sha(o.index),
    timing='warmup excluded CUDA sync; shared-stage composed screen; independent complete request confirmation with rotated order',attention=model.config._attn_implementation,
    score_normalization='full vocabulary logprob, no legal-vocab renormalization',source_hashes={n:sha(Path(__file__).parent/n) for n in ['eval_search_matrix.py','eval_cached_beam.py','eval_pilot.py']}),peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
  print(json.dumps(dict(done=j+1,total=len(ids),wall_seconds=time.perf_counter()-start)),flush=True)

if __name__=='__main__':main()
