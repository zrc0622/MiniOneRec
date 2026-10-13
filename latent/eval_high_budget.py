"""Independently timed top20 eval for8x20/16x20/8x50, no training changes."""
import argparse,json,time
from collections import defaultdict
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from .core import load_index,history_records,validate_protocol,write_json,sha
from .generation import Grammar
from .eval_at20 import infer
from .eval_pilot import timed

def main():
 p=argparse.ArgumentParser()
 for key in ['model','data','index','sample','output']:p.add_argument('--'+key,required=True)
 p.add_argument('--methods',nargs='+',required=True);o=p.parse_args();torch.set_num_threads(1)
 tk=AutoTokenizer.from_pretrained(o.model);tk.pad_token=tk.eos_token;model=AutoModelForCausalLM.from_pretrained(o.model,torch_dtype=torch.bfloat16).cuda().eval()
 index=load_index(o.index);grammar=Grammar(tk,index,validate_protocol(model.config,tk,index));records=history_records(o.data,index,grammar.k);ids=json.loads(Path(o.sample).read_text())['row_ids']
 for method in o.methods:
  torch.cuda.reset_peak_memory_stats();r,t=timed(lambda:infer(model,tk,grammar,records[ids[0]],method,return_details=True));print(json.dumps(dict(warmup=method,seconds=t,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,unique=r['unique_before_top20'])),flush=True)
 rows=defaultdict(list);start=time.perf_counter()
 for j,rid in enumerate(ids):
  order=o.methods[j%len(o.methods):]+o.methods[:j%len(o.methods)]
  for method in order:
   torch.cuda.reset_peak_memory_stats();r,t=timed(lambda:infer(model,tk,grammar,records[rid],method,return_details=True))
   rows[method].append(dict(row_id=rid,output=records[rid]['target'],elapsed_seconds=t,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,**r))
  write_json(o.output,dict(rows=dict(rows),metadata=dict(model=o.model,top_k=20,examples=len(ids),sample_sha256=sha(o.sample),data_sha256=sha(o.data),index_sha256=sha(o.index),source_hashes={n:sha(Path(__file__).parent/n) for n in ['eval_high_budget.py','eval_at20.py','eval_cached_beam.py']},attention=model.config._attn_implementation,timing='independent complete request, CUDA synchronized, excludedwarmup, rotatedmethods; peakallocated per request includesmodel; samefullvocablikelihood.')))
  print(json.dumps(dict(done=j+1,total=len(ids),wall_seconds=time.perf_counter()-start)),flush=True)
if __name__=='__main__':main()
