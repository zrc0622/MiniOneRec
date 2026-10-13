"""Top20 budget comparison; preserve all earlier top10 eval artifacts."""
import argparse,json,time
from collections import defaultdict
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from .core import history_records,load_index,validate_protocol,write_json,sha
from .generation import Grammar
from .eval_search import beam_generate,rank_items
from .eval_cached_beam import cached_item_beam_batch
from .eval_pilot import PrefixScorer,timed
from .eval_search_matrix import flatten,cross_scores

STAGES={'single20':(1,20),'single50':(1,50),'two20_sum':(2,20),'four10_sum':(4,10),'four20_sum':(4,20),'eight10_sum':(8,10),'eight20_sum':(8,20),'sixteen20_sum':(16,20),'eight50_sum':(8,50)}

def rank(grammar,paths,scores,aggregation='logsumexp'):
 return rank_items([(grammar.item(s),p) for s,p in zip(paths,scores)],aggregation,20)

def trial(model,tk,grammar,record):
 prompt=tk.encode(record['prompt'],add_special_tokens=False)[-512:]
 p4,t4=timed(lambda:beam_generate(model,prompt,grammar,4,'interest'));p8,t8=timed(lambda:beam_generate(model,prompt,grammar,8,'interest'))
 scorer,tc=timed(lambda:PrefixScorer(model,prompt));ps4,s4=timed(lambda:scorer.score(p4));ps8,s8=timed(lambda:scorer.score(p8))
 results={}
 for name in ['single20','single50','two20_sum','four10_sum','four20_sum','eight10_sum']:
  n,width=STAGES[name]
  prefixes=p4[:n] if n<=4 else p8;priors=ps4[:n] if n<=4 else ps8
  groups,tg=timed(lambda:cached_item_beam_batch(model,prompt,prefixes,grammar,width,scorer.cache))
  paths,scores=flatten(prefixes,groups,priors);unique=len(set(grammar.item(s) for s in paths));setup=tc+tg+(t4+s4 if n<=4 else t8+s8)
  pred,tr=timed(lambda:rank(grammar,paths,scores));results[name]=dict(predict=pred,elapsed_seconds=setup+tr,unique_before_top20=unique,generated_paths=len(paths))
  if name=='four20_sum':
   pred,tr=timed(lambda:rank(grammar,paths,scores,'max'));results['four20_max']=dict(predict=pred,elapsed_seconds=setup+tr,unique_before_top20=unique,generated_paths=len(paths))
   (cross,scores),tx=timed(lambda:cross_scores(scorer,grammar,prefixes,paths))
   pred,tr=timed(lambda:rank(grammar,cross,scores));results['four20_marginal']=dict(predict=pred,elapsed_seconds=setup+tx+tr,unique_before_top20=unique,generated_paths=len(paths),scored_paths=len(cross))
 return results

def infer(model,tk,grammar,record,name,return_details=False):
 prompt=tk.encode(record['prompt'],add_special_tokens=False)[-512:]
 n,width=STAGES['four20_sum' if name in ['four20_max','four20_marginal'] else name]
 prefixes=beam_generate(model,prompt,grammar,max(4,n),'interest')[:n];scorer=PrefixScorer(model,prompt);ps=scorer.score(prefixes)
 groups=cached_item_beam_batch(model,prompt,prefixes,grammar,width,scorer.cache);paths,scores=flatten(prefixes,groups,ps)
 if name=='four20_marginal':paths,scores=cross_scores(scorer,grammar,prefixes,paths)
 prediction=rank(grammar,paths,scores,'max' if name=='four20_max' else 'logsumexp')
 if return_details:
  import math
  return dict(predict=prediction,unique_before_top20=len(set(grammar.item(s) for s in paths)),generated_paths=sum(len(items) for items,_ in groups),selected_latent_mass=sum(math.exp(p) for p in ps))
 return prediction

def main():
 p=argparse.ArgumentParser()
 for key in ['model','data','index','sample','output']:p.add_argument('--'+key,required=True)
 protocol=json.loads((Path(__file__).resolve().parents[1]/'eval_metric_protocol.json').read_text())
 p.add_argument('--mode',choices=['screen','confirm'],default='confirm');p.add_argument('--methods',nargs='+',default=protocol['latent_eval_methods']);o=p.parse_args();torch.set_num_threads(1)
 tk=AutoTokenizer.from_pretrained(o.model);tk.pad_token=tk.eos_token;model=AutoModelForCausalLM.from_pretrained(o.model,torch_dtype=torch.bfloat16).cuda().eval()
 index=load_index(o.index);grammar=Grammar(tk,index,validate_protocol(model.config,tk,index));records=history_records(o.data,index,grammar.k);ids=json.loads(Path(o.sample).read_text())['row_ids']
 if o.mode=='screen':trial(model,tk,grammar,records[ids[0]])
 else:
  for m in o.methods:infer(model,tk,grammar,records[ids[0]],m)
 rows=defaultdict(list);start=time.perf_counter()
 for j,rid in enumerate(ids):
  if o.mode=='screen':result=trial(model,tk,grammar,records[rid])
  else:
   result={};order=o.methods[j%len(o.methods):]+o.methods[:j%len(o.methods)]
   for m in order:
    pred,t=timed(lambda:infer(model,tk,grammar,records[rid],m));result[m]=dict(predict=pred,elapsed_seconds=t)
  for m,r in result.items():rows[m].append(dict(row_id=rid,output=records[rid]['target'],**r))
  write_json(o.output,dict(rows=dict(rows),metadata=dict(model=o.model,mode=o.mode,top_k=20,examples=len(ids),sample_sha256=sha(o.sample),data_sha256=sha(o.data),index_sha256=sha(o.index),source_sha256=sha(__file__),attention=model.config._attn_implementation,timing='screen shared-stage composition inclranking; confirm independent complete request CUDA synced exclude warmup')))
  print(json.dumps(dict(done=j+1,total=len(ids),wall_seconds=time.perf_counter()-start)),flush=True)
if __name__=='__main__':main()
