"""Inference pilot using shared prompt KV and beam-returned full-vocab scores."""
import argparse,json,time
from pathlib import Path
from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from .core import load_index,history_records,validate_protocol,write_json,sha
from .generation import Grammar
from .eval_search import beam_generate,rank_items
from .eval_pilot import PrefixScorer,timed
from .eval_cached_beam import cached_item_beam


def trial(model,tk,grammar,record,row_id):
    prompt=tk.encode(record['prompt'],add_special_tokens=False)[-512:]
    prefixes,pt=timed(lambda:beam_generate(model,prompt,grammar,4,'interest'))
    scorer,ct=timed(lambda:PrefixScorer(model,prompt))
    ps,pst=timed(lambda:scorer.score(prefixes))
    full=[];small=[];ft=[];st=[];fs=[];ss=[]
    for j,prefix in enumerate(prefixes):
        (items,scores),t=timed(lambda:cached_item_beam(model,prompt,prefix,grammar,50,scorer.cache))
        full.append([prefix+i for i in items]);fs.append([ps[j]+s for s in scores]);ft.append(t)
        if j:
            (items,scores),t=timed(lambda:cached_item_beam(model,prompt,prefix,grammar,10,scorer.cache))
            small.append([prefix+i for i in items]);ss.append([ps[j]+s for s in scores]);st.append(t)
    results={}
    def save(name,paths,scores,elapsed,aggregation='max',**extra):
        items=[grammar.item(s) for s in paths]
        results[name]=dict(row_id=row_id,output=record['target'],predict=rank_items(list(zip(items,scores)),aggregation),
            elapsed_seconds=elapsed,valid_paths=len(paths),unique_before_top50=len(set(items)),**extra)
    setup=pt+ct+pst
    for n in [1,2,4]:
        save(f'fast_{n}x50',sum(full[:n],[]),sum(fs[:n],[]),setup+sum(ft[:n]))
    paths80=full[0]+sum(small,[]);scores80=fs[0]+sum(ss,[])
    generation=setup+ft[0]+sum(st)
    save('fast80_max',paths80,scores80,generation)
    save('fast80_observed_sum',paths80,scores80,generation,'logsumexp')
    candidates={}
    for seq in paths80: candidates.setdefault(grammar.item(seq),seq[3:])
    cross=[prefix+item for item in candidates.values() for prefix in prefixes]
    cross_scores,rt=timed(lambda:scorer.score(cross))
    save('fast80_marginal',cross,cross_scores,generation+rt,'logsumexp',proposal_paths=len(paths80))
    return results


def main():
    p=argparse.ArgumentParser()
    for k in ['model','data','index','sample','output']:p.add_argument('--'+k,required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=2)
    o=p.parse_args();torch.set_num_threads(1)
    index=load_index(o.index);tk=AutoTokenizer.from_pretrained(o.model);tk.pad_token=tk.eos_token
    model=AutoModelForCausalLM.from_pretrained(o.model,torch_dtype=torch.bfloat16).cuda().eval()
    grammar=Grammar(tk,index,validate_protocol(model.config,tk,index))
    records=history_records(o.data,index,grammar.k);selected=json.loads(Path(o.sample).read_text())['row_ids']
    print('warmup',flush=True);trial(model,tk,grammar,records[selected[0]],selected[0])
    rows=defaultdict(list);started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for n,rid in enumerate(selected[o.shard::o.shards]):
        result=trial(model,tk,grammar,records[rid],rid)
        for name,r in result.items():rows[name].append(r)
        elapsed=time.perf_counter()-started
        write_json(o.output,dict(shard=o.shard,rows=dict(rows),elapsed_seconds=elapsed,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            metadata=dict(model=o.model,data_sha256=sha(o.data),index_sha256=sha(o.index),sample_sha256=sha(o.sample),
                attention=model.config._attn_implementation,timing='CUDA synchronized warmup excluded shared stages composed; beam50 branches reuse historical KV, beam-returned full-vocabulary log probabilities; sequential batch1 requests')))
        print(json.dumps(dict(done=n+1,total=len(selected[o.shard::o.shards]),row_id=rid,wall_seconds=elapsed,
            latency={k:round(r['elapsed_seconds'],3) for k,r in result.items()})),flush=True)

if __name__=='__main__':main()
