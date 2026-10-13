"""Length-bucketed multiuser eval with cached prompt KV and adaptive batches."""
import argparse,copy,gc,json,math,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig
from .core import load_index,history_records,validate_protocol,write_json,sha
from .generation import Grammar
from .eval_search import rank_items
from .eval_pilot import timed

RECIPES={'eight20_sum':(8,20),'sixteen20_sum':(16,20),'eight50_sum':(8,50)}

def cfg(grammar,beams,length):
 return GenerationConfig(max_new_tokens=length,do_sample=False,num_beams=beams,num_return_sequences=beams,length_penalty=0.,top_k=None,top_p=None,pad_token_id=grammar.eos,eos_token_id=grammar.eos,forced_eos_token_id=None,renormalize_logits=False,use_cache=True,return_dict_in_generate=True,output_scores=True)

@torch.no_grad()
def batch_infer(model,grammar,prompts,n,width):
 device=next(model.parameters()).device;size=len(prompts);plen=max(map(len,prompts))
 ids=torch.tensor([[grammar.eos]*(plen-len(p))+p for p in prompts],device=device)
 mask=torch.tensor([[0]*(plen-len(p))+[1]*len(p) for p in prompts],device=device)
 interests=model.generate(input_ids=ids,attention_mask=mask,generation_config=cfg(grammar,n,grammar.k),use_model_defaults=False,do_sample=False,prefix_allowed_tokens_fn=lambda _,tokens:grammar.allowed(tokens[plen:].tolist(),'interest'),synced_gpus=False)
 prefixes=interests.sequences[:,plen:].tolist();priors=interests.sequences_scores.tolist()
 if len(prefixes)!=size*n:raise ValueError('Incomplete latent beam batch')
 for prefix,score in zip(prefixes,priors):
  if len(prefix)!=grammar.k or not math.isfinite(score) or not all(t in grammar.allowed(prefix[:i],'interest') for i,t in enumerate(prefix)):raise ValueError('Invalid interest beam')
 del interests
 # RoPE positions must count real tokens, rather than left padding.
 positions=(mask.cumsum(-1)-1).clamp_min(0)
 out=model(input_ids=ids,attention_mask=mask,position_ids=positions,use_cache=True,logits_to_keep=1)
 cache=out.past_key_values;del out
 cache.batch_repeat_interleave(n*width)
 pref=torch.tensor(prefixes,device=device);item_ids=torch.cat([ids.repeat_interleave(n,0),pref],dim=1)
 item_mask=torch.cat([mask.repeat_interleave(n,0),torch.ones_like(pref)],dim=1)
 prompt_length=plen+grammar.k
 generated=model.generate(input_ids=item_ids,attention_mask=item_mask,past_key_values=cache,generation_config=cfg(grammar,width,grammar.item_length),use_model_defaults=False,do_sample=False,prefix_allowed_tokens_fn=lambda _,tokens:grammar.allowed(tokens[prompt_length:].tolist(),'item'),synced_gpus=False)
 sequences=generated.sequences[:,prompt_length:].tolist();scores=generated.sequences_scores.tolist();results=[]
 for user in range(size):
  pairs=[];valid_paths=0
  for branch in range(n):
   idx=user*n+branch;prefix=prefixes[idx];seen=set()
   for seq,score in zip(sequences[idx*width:(idx+1)*width],scores[idx*width:(idx+1)*width]):
    if grammar.eos in seq:seq=seq[:seq.index(grammar.eos)+1]
    key=tuple(seq)
    if key in seen or not math.isfinite(score) or not seq or seq[-1]!=grammar.eos:continue
    if not all(t in grammar.allowed(seq[:i],'item') for i,t in enumerate(seq)):continue
    seen.add(key);pairs.append((grammar.item(prefix+seq),priors[idx]+score));valid_paths+=1
  if not pairs:raise ValueError('No SID completions')
  results.append(dict(predict=rank_items(pairs,'logsumexp',20),unique_before_top20=len(set(item for item,_ in pairs)),generated_paths=valid_paths,selected_latent_mass=sum(math.exp(p) for p in priors[user*n:(user+1)*n])))
 return results

def capacity(model,grammar,length,n,width,maximum):
 free,total=torch.cuda.mem_get_info();reclaim=torch.cuda.memory_reserved()-torch.cuda.memory_allocated();available=.80*(free+reclaim)
 c=model.config;kv=2*2*c.num_hidden_layers*c.num_key_value_heads*c.head_dim*(length+grammar.k+grammar.item_length)*n*width
 score=c.vocab_size*4*grammar.item_length*n*width
 cost=1.4*kv+score+256*2**20
 return max(1,min(maximum,int(available//cost)))

def main():
 p=argparse.ArgumentParser()
 for key in ['model','data','index','sample','output','method']:p.add_argument('--'+key,required=True)
 p.add_argument('--max-batch',type=int,default=8);o=p.parse_args();torch.set_num_threads(1)
 n,width=RECIPES[o.method];tk=AutoTokenizer.from_pretrained(o.model);tk.pad_token=tk.eos_token;model=AutoModelForCausalLM.from_pretrained(o.model,torch_dtype=torch.bfloat16).cuda().eval()
 index=load_index(o.index);grammar=Grammar(tk,index,validate_protocol(model.config,tk,index));records=history_records(o.data,index,grammar.k);ids=json.loads(Path(o.sample).read_text())['row_ids']
 encoded={rid:tk.encode(records[rid]['prompt'],add_special_tokens=False)[-512:] for rid in ids};order=sorted(ids,key=lambda rid:(len(encoded[rid]),rid));rows={};batches=[];retries=[]
 # Check the longest prompt at batch1 before starting sustained eval.
 longest=order[-1];torch.cuda.reset_peak_memory_stats();_,t=timed(lambda:batch_infer(model,grammar,[encoded[longest]],n,width))
 print(json.dumps(dict(warmup=o.method,longest_tokens=len(encoded[longest]),seconds=t,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)),flush=True)
 start=time.perf_counter();cursor=0
 while cursor<len(order):
  tentative=order[cursor:cursor+o.max_batch];maxlen=max(len(encoded[r]) for r in tentative);count=min(len(tentative),capacity(model,grammar,maxlen,n,width,o.max_batch))
  while True:
   selected=order[cursor:cursor+count];torch.cuda.reset_peak_memory_stats()
   try:
    predictions,elapsed=timed(lambda:batch_infer(model,grammar,[encoded[rid] for rid in selected],n,width));break
   except torch.cuda.OutOfMemoryError:
    if count==1:raise
    retries.append(dict(cursor=cursor,failed_batch=count));count=max(1,count//2);gc.collect();torch.cuda.empty_cache()
  peak=torch.cuda.max_memory_allocated()/2**30
  batches.append(dict(row_ids=selected,size=count,max_prompt_tokens=max(len(encoded[r]) for r in selected),elapsed_seconds=elapsed,peak_allocated_gib=peak))
  for rid,r in zip(selected,predictions):rows[rid]=dict(row_id=rid,output=records[rid]['target'],elapsed_seconds=elapsed/count,batch_size=count,**r)
  cursor+=count;sortedrows=[rows[rid] for rid in ids if rid in rows]
  write_json(o.output,dict(rows={o.method:sortedrows},batches=batches,oom_retries=retries,metadata=dict(model=o.model,method=o.method,top_k=20,examples=len(ids),completed=len(rows),max_batch=o.max_batch,sample_sha256=sha(o.sample),data_sha256=sha(o.data),index_sha256=sha(o.index),attention=model.config._attn_implementation,source_sha256=sha(__file__),timing='CUDA synchronized batchgeneration+ranking, excludedmodel/data load/tokenization/warmup; perrow time=batchelapsed/batchsize, throughput amortization NOT single-request latency',probabilities='fullvocab unrenormalized interestbeam and conditionalSID scores, observed-path logsumexp',batching='lengthbucketed leftpadding, promptKV reused, explicitRoPEpositions; BF16batch changes ranks, no bitwise guarantee')))
  print(json.dumps(dict(done=cursor,total=len(order),batch=count,elapsed_seconds=elapsed,peak_allocated_gib=peak,wall_seconds=time.perf_counter()-start)),flush=True)
if __name__=='__main__':main()
