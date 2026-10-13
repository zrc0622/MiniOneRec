"""Standalone timed replay of inference finalists, without sharing stages."""
import argparse,json,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from interest.evaluate import summarize
from .core import load_index,history_records,validate_protocol,write_json
from .generation import Grammar
from .eval_search import beam_generate,search_paths,rank_items
from .eval_pilot import PrefixScorer,original_scores,timed
from .eval_cached_beam import cached_item_beam


def infer(model,tk,grammar,record,method):
    prompt=tk.encode(record['prompt'],add_special_tokens=False)[-512:]
    if method=='reference_4x50':
        paths=search_paths(model,prompt,grammar,'hierarchical_max',latent_count=4,item_beams=50)
        scores=original_scores(model,prompt,paths,grammar)
        aggregation='max'
    else:
        prefixes=beam_generate(model,prompt,grammar,4,'interest')
        scorer=PrefixScorer(model,prompt);ps=scorer.score(prefixes)
        n={'fast_1x50':1,'fast_2x50':2}.get(method,4)
        paths=[];scores=[]
        for j,prefix in enumerate(prefixes[:n]):
            count=50 if j==0 or method in ['fast_2x50','fast_4x50'] else 10
            items,probs=cached_item_beam(model,prompt,prefix,grammar,count,scorer.cache)
            paths.extend(prefix+i for i in items);scores.extend(ps[j]+s for s in probs)
        aggregation='logsumexp' if 'sum' in method or 'marginal' in method else 'max'
        if 'marginal' in method:
            candidates={}
            for seq in paths:candidates.setdefault(grammar.item(seq),seq[3:])
            paths=[prefix+item for item in candidates.values() for prefix in prefixes]
            scores=scorer.score(paths,batch=32)
    return rank_items([(grammar.item(s),score) for s,score in zip(paths,scores)],aggregation)


def main():
    p=argparse.ArgumentParser()
    for k in ['model','data','index','sample','output']:p.add_argument('--'+k,required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=2)
    p.add_argument('--methods',nargs='+')
    o=p.parse_args();torch.set_num_threads(1)
    index=load_index(o.index);tk=AutoTokenizer.from_pretrained(o.model);tk.pad_token=tk.eos_token
    model=AutoModelForCausalLM.from_pretrained(o.model,torch_dtype=torch.bfloat16).cuda().eval()
    grammar=Grammar(tk,index,validate_protocol(model.config,tk,index));records=history_records(o.data,index,grammar.k)
    ids=json.loads(Path(o.sample).read_text())['row_ids'][o.shard::o.shards]
    methods=o.methods or ['reference_4x50','fast_2x50','fast_4x50','fast80_observed_sum','fast80_marginal_batch32']
    for name in methods:infer(model,tk,grammar,records[ids[0]],name)
    torch.cuda.reset_peak_memory_stats();rows=defaultdict(list);start=time.perf_counter()
    for n,rid in enumerate(ids):
        # Rotate method order to reduce persistent order/warmup bias.
        order=methods[n%len(methods):]+methods[:n%len(methods)]
        for name in order:
            predict,elapsed=timed(lambda:infer(model,tk,grammar,records[rid],name))
            rows[name].append(dict(row_id=rid,output=records[rid]['target'],predict=predict,elapsed_seconds=elapsed))
        write_json(o.output,dict(rows=dict(rows),shard=o.shard,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            metadata=dict(timing='Standalone per-method CUDA synchronized request includes tokenization, full search, scoring, CPU sorting; warmup excluded; method order rotated; no stage sharing',model=o.model)))
        print(json.dumps(dict(done=n+1,total=len(ids),wall_seconds=time.perf_counter()-start)),flush=True)


def report(root):
    parts=[json.loads((root/f'replay_shard{i}.json').read_text()) for i in range(2)]
    metrics={}
    for name in parts[0]['rows']:
        rows=sorted(parts[0]['rows'][name]+parts[1]['rows'][name],key=lambda x:x['row_id'])
        assert [r['row_id'] for r in rows]==json.loads((root/'sample64.json').read_text())['row_ids']
        m=summarize(rows);seconds=np.array([r['elapsed_seconds'] for r in rows])
        m.update(mean_seconds=float(seconds.mean()),p95_seconds=float(np.percentile(seconds,95)),fraction_with50=float(np.mean([len(r['predict'])==50 for r in rows])))
        metrics[name]=m
    for m in metrics.values():m['speedup']=metrics['reference_4x50']['mean_seconds']/m['mean_seconds']
    output=dict(metadata=parts[0]['metadata'],metrics=metrics,peak_reserved_gib=[p['peak_reserved_gib'] for p in parts])
    write_json(root/'replay_comparison.json',output)
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
