"""Uniform two-protocol small eval, batched users and exact cross-latent scoring."""
import argparse,copy,csv,gc,json,math,time
from collections import defaultdict
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from data import EvalSidDataset
from .core import load_index,history_records,validate_protocol,write_json,sha
from .generation import Grammar
from .eval_search import rank_items
from .eval_batched_high_budget import cfg,capacity
from .eval_pilot import timed

class BatchPrefixScorer:
 def __init__(self,model,cache,mask,first):
  self.model=model;self.cache=cache;self.mask=mask;self.first=first;self.width=mask.shape[1]
 @torch.no_grad()
 def score(self,sequences,users,batch=32):
  if len(sequences)!=len(users):raise ValueError('Score user mapping mismatch')
  groups=defaultdict(list);result=[None]*len(sequences);device=self.mask.device
  for i,seq in enumerate(sequences):groups[len(seq)].append(i)
  for length,indices in groups.items():
   for start in range(0,len(indices),batch):
    chosen=indices[start:start+batch];uid=torch.tensor([users[i] for i in chosen],device=device);ids=torch.tensor([sequences[i] for i in chosen],device=device)
    if length==1:scores=self.first[uid,ids[:,0]]
    else:
     cache=copy.deepcopy(self.cache);cache.batch_select_indices(uid)
     mask=torch.cat([self.mask[uid],torch.ones((len(chosen),length-1),dtype=self.mask.dtype,device=device)],dim=1)
     positions=(mask.cumsum(-1)-1).clamp_min(0)[:,self.width:]
     out=self.model(input_ids=ids[:,:-1],past_key_values=cache,attention_mask=mask,position_ids=positions,use_cache=True)
     logits=out.logits.float();logps=logits.gather(-1,ids[:,1:,None]).squeeze(-1)-torch.logsumexp(logits,-1)
     scores=logps.sum(1)+self.first[uid,ids[:,0]]
    for i,v in zip(chosen,scores.tolist()):result[i]=v
  return result

def padded(prompts,eos,device):
 width=max(map(len,prompts));ids=torch.tensor([[eos]*(width-len(p))+p for p in prompts],device=device);mask=torch.tensor([[0]*(width-len(p))+[1]*len(p) for p in prompts],device=device)
 return ids,mask

@torch.no_grad()
def batch_latent(model,grammar,prompts,method):
 n=1 if method=='single20' else 4
 if method not in ['single20','four20_marginal']:raise ValueError(method)
 device=next(model.parameters()).device;ids,mask=padded(prompts,grammar.eos,device);size,plen=ids.shape
 interest=model.generate(input_ids=ids,attention_mask=mask,generation_config=cfg(grammar,4,grammar.k),use_model_defaults=False,do_sample=False,prefix_allowed_tokens_fn=lambda _,t:grammar.allowed(t[plen:].tolist(),'interest'),synced_gpus=False)
 codes=interest.sequences[:,plen:].tolist();prior=interest.sequences_scores.tolist()
 prefixes=[codes[u*4+b] for u in range(size) for b in range(n)];ps=[prior[u*4+b] for u in range(size) for b in range(n)]
 if any(len(p)!=grammar.k for p in prefixes):raise ValueError('Incomplete interest code')
 del interest
 out=model(input_ids=ids,attention_mask=mask,position_ids=(mask.cumsum(-1)-1).clamp_min(0),use_cache=True,logits_to_keep=1)
 base=out.past_key_values;logits=out.logits[:,-1].float();first=logits-torch.logsumexp(logits,-1,keepdim=True);del out,logits
 cache=copy.deepcopy(base);cache.batch_repeat_interleave(n*20)
 prefix_ids=torch.tensor(prefixes,device=device);items_ids=torch.cat([ids.repeat_interleave(n,0),prefix_ids],1);items_mask=torch.cat([mask.repeat_interleave(n,0),torch.ones_like(prefix_ids)],1);start=plen+grammar.k
 generated=model.generate(input_ids=items_ids,attention_mask=items_mask,past_key_values=cache,generation_config=cfg(grammar,20,grammar.item_length),use_model_defaults=False,do_sample=False,prefix_allowed_tokens_fn=lambda _,t:grammar.allowed(t[start:].tolist(),'item'),synced_gpus=False)
 sequences=generated.sequences[:,start:].tolist();item_scores=generated.sequences_scores.tolist();del generated,cache,items_ids,items_mask
 proposals=[];pairs_by_user=[];paths_per_user=[]
 for u in range(size):
  candidates={};pairs=[];valid=0
  for b in range(n):
   branch=u*n+b;seen=set()
   for seq,score in zip(sequences[branch*20:(branch+1)*20],item_scores[branch*20:(branch+1)*20]):
    if grammar.eos in seq:seq=seq[:seq.index(grammar.eos)+1]
    key=tuple(seq)
    if key in seen or not math.isfinite(score) or not seq or seq[-1]!=grammar.eos:continue
    if not all(t in grammar.allowed(seq[:i],'item') for i,t in enumerate(seq)):continue
    seen.add(key);item=grammar.item(prefixes[branch]+seq);candidates.setdefault(item,seq);pairs.append((item,ps[branch]+score));valid+=1
  if not candidates:raise ValueError('No SID candidates')
  proposals.append(candidates);pairs_by_user.append(pairs);paths_per_user.append(valid)
 if method=='four20_marginal':
  seqs=[];users=[];items=[]
  for u,candidates in enumerate(proposals):
   for item,seq in candidates.items():
    for b in range(4):seqs.append(prefixes[u*4+b]+seq);users.append(u);items.append(item)
  scores=BatchPrefixScorer(model,base,mask,first).score(seqs,users,32);pairs_by_user=[[] for _ in range(size)]
  for u,item,score in zip(users,items,scores):pairs_by_user[u].append((item,score))
 return [dict(predict=rank_items(pairs_by_user[u],'logsumexp' if n==4 else 'max',20),unique_before_top20=len(proposals[u]),generated_paths=paths_per_user[u],scored_paths=len(pairs_by_user[u]),selected_latent_mass=sum(math.exp(p) for p in ps[u*n:(u+1)*n])) for u in range(size)]

@torch.no_grad()
def batch_sid(model,grammar,prompts,width=50):
 device=next(model.parameters()).device;ids,mask=padded(prompts,grammar.eos,device);size,plen=ids.shape
 if plen<2:raise ValueError('Prompt too short for cached SID generation')
 out=model(input_ids=ids[:,:-1],attention_mask=mask[:,:-1],position_ids=(mask.cumsum(-1)-1).clamp_min(0)[:,:-1],use_cache=True,logits_to_keep=1)
 cache=out.past_key_values;del out;cache.batch_repeat_interleave(width)
 generated=model.generate(input_ids=ids,attention_mask=mask,past_key_values=cache,generation_config=cfg(grammar,width,grammar.item_length),use_model_defaults=False,do_sample=False,prefix_allowed_tokens_fn=lambda _,t:grammar.allowed(t[plen:].tolist(),'item'),synced_gpus=False)
 sequences=generated.sequences[:,plen:].tolist();scores=generated.sequences_scores.tolist();results=[]
 for u in range(size):
  pairs=[];seen=set()
  for seq,score in zip(sequences[u*width:(u+1)*width],scores[u*width:(u+1)*width]):
   if grammar.eos in seq:seq=seq[:seq.index(grammar.eos)+1]
   key=tuple(seq)
   if key in seen or not math.isfinite(score) or not seq or seq[-1]!=grammar.eos:continue
   if not all(t in grammar.allowed(seq[:i],'item') for i,t in enumerate(seq)):continue
   seen.add(key);pairs.append((grammar.sid_by_ids[False][key],score))
  results.append(dict(predict=[item for item,_ in pairs][:20],unique_before_top20=len(pairs),generated_paths=len(pairs)))
 return results

def main():
 p=argparse.ArgumentParser()
 for key in ['model','data','index','sample','output','kind']:p.add_argument('--'+key,required=True)
 p.add_argument('--max-batch',type=int,default=32);o=p.parse_args();torch.set_num_threads(1)
 protocol=json.loads((Path(__file__).resolve().parents[1]/'eval_metric_protocol.json').read_text());methods=protocol['latent_eval_methods'] if o.kind=='latent' else ['direct50']
 tk=AutoTokenizer.from_pretrained(o.model);tk.pad_token=tk.eos_token;model=AutoModelForCausalLM.from_pretrained(o.model,torch_dtype=torch.bfloat16).cuda().eval();index=load_index(o.index);ids=json.loads(Path(o.sample).read_text())['row_ids']
 if o.kind=='latent':
  grammar=Grammar(tk,index,validate_protocol(model.config,tk,index));records=history_records(o.data,index,grammar.k);encoded={rid:tk.encode(records[rid]['prompt'],add_special_tokens=False)[-512:] for rid in ids};targets={rid:records[rid]['target'] for rid in ids}
 else:
  grammar=Grammar(tk,index,dict(tokens=[[],[],[]]))
  with open(o.data) as f:source=list(csv.DictReader(f))
  path=Path(o.output).with_suffix('.csv')
  with path.open('w') as f:w=csv.DictWriter(f,fieldnames=list(source[0]));w.writeheader();w.writerows(source[rid] for rid in ids)
  dataset=EvalSidDataset(str(path),tk,max_len=2560,test=True,K=0,seed=42);encoded={rid:dataset[j]['input_ids'] for j,rid in enumerate(ids)};targets={rid:source[rid]['item_sid'] for rid in ids}
 order=sorted(ids,key=lambda rid:(len(encoded[rid]),rid));rows={m:{} for m in methods};batches={m:[] for m in methods};retries=[];start_time=time.perf_counter();initial_hash=sha(__file__)
 def write():
  write_json(o.output,dict(rows={m:[rows[m][rid] for rid in ids if rid in rows[m]] for m in methods},batches=batches,oom_retries=retries,metadata=dict(model=o.model,kind=o.kind,methods=methods,top_k=20,examples=len(ids),completed={m:len(rows[m]) for m in methods},sample_sha256=sha(o.sample),data_sha256=sha(o.data),index_sha256=sha(o.index),source_sha256=initial_hash,protocol=protocol,attention=model.config._attn_implementation,timing='CUDA synced batchthroughput includinggeneration andscoring/ranking, excludedload/tokenization/warmup; amortizedperrow NOTsingleuserlatency',batching='leftpadRoPEcorrect, sharedpromptKV, lengthbucket adaptivebatchmax32; no BF16bitwiseguarantee')))
 for method in methods:
  fn=batch_sid if o.kind=='sid' else lambda model,grammar,prompts:batch_latent(model,grammar,prompts,method)
  longest=order[-1];_,warm=timed(lambda:fn(model,grammar,[encoded[longest]]));print(json.dumps(dict(warmup=method,seconds=warm)),flush=True);cursor=0
  while cursor<len(order):
   tentative=order[cursor:cursor+o.max_batch];length=max(len(encoded[r]) for r in tentative);n,width=(1,50) if o.kind=='sid' else (1 if method=='single20' else 4,20)
   count=min(len(tentative),capacity(model,grammar,length,n,width,o.max_batch))
   while True:
    selected=order[cursor:cursor+count];torch.cuda.reset_peak_memory_stats()
    try:predictions,elapsed=timed(lambda:fn(model,grammar,[encoded[r] for r in selected]));break
    except torch.cuda.OutOfMemoryError:
     if count==1:raise
     retries.append(dict(method=method,cursor=cursor,failed_batch=count));count=max(1,count//2);gc.collect();torch.cuda.empty_cache()
   peak=torch.cuda.max_memory_allocated()/2**30;batches[method].append(dict(row_ids=selected,size=count,elapsed_seconds=elapsed,peak_allocated_gib=peak))
   for rid,r in zip(selected,predictions):rows[method][rid]=dict(row_id=rid,output=targets[rid],elapsed_seconds=elapsed/count,batch_size=count,**r)
   cursor+=count;write();print(json.dumps(dict(method=method,done=cursor,total=len(ids),batch=count,wall_seconds=time.perf_counter()-start_time,peak_allocated_gib=peak)),flush=True)
if __name__=='__main__':main()
