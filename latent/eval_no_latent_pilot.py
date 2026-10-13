"""No-latent baseline on identical sampled validation rows, native SFT prompt."""
import argparse,csv,json,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from data import EvalSidDataset
from .core import load_index,write_json,sha
from .generation import Grammar
from .eval_search import beam_generate
from .eval_pilot import timed


def main():
 p=argparse.ArgumentParser()
 for key in ['model','data','index','sample','output']:p.add_argument('--'+key,required=True)
 p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=2)
 o=p.parse_args();torch.set_num_threads(1)
 ids=json.loads(Path(o.sample).read_text())['row_ids'][o.shard::o.shards]
 with open(o.data) as f:source=list(csv.DictReader(f))
 csvpath=Path(o.output).with_suffix('.csv')
 with csvpath.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(source[0]));w.writeheader();w.writerows(source[i] for i in ids)
 tk=AutoTokenizer.from_pretrained(o.model);tk.pad_token=tk.eos_token
 dataset=EvalSidDataset(train_file=str(csvpath),tokenizer=tk,max_len=2560,test=True,K=0,seed=42,
    category='industrial and scientific items')
 model=AutoModelForCausalLM.from_pretrained(o.model,torch_dtype=torch.bfloat16).cuda().eval()
 grammar=Grammar(tk,load_index(o.index),dict(tokens=[[],[],[]]))
 def request(i):
  prompt=dataset[i]['input_ids']
  samples=beam_generate(model,prompt,grammar,50,phase='item')
  return [grammar.sid_by_ids[False][tuple(s)] for s in samples]
 request(0);torch.cuda.reset_peak_memory_stats();rows=[]
 for i,rid in enumerate(ids):
  predict,elapsed=timed(lambda:request(i))
  rows.append(dict(row_id=rid,output=source[rid]['item_sid'],predict=predict,elapsed_seconds=elapsed,
      prompt_tokens=len(dataset[i]['input_ids'])))
  write_json(o.output,dict(rows=rows,shard=o.shard,metadata=dict(model=o.model,data_sha256=sha(o.data),
      index_sha256=sha(o.index),sample_sha256=sha(o.sample),prompt='native EvalSidDataset testTrue K0 untruncated nativehistory',
      search='50beam deterministic fullvocabulary logprob length_penalty0 legalSID grammar',attention=model.config._attn_implementation),
      peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
  print(json.dumps(dict(done=i+1,total=len(ids),seconds=elapsed)),flush=True)

if __name__=='__main__':main()
