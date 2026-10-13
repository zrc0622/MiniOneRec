"""Merge fixed validation pilot and report paired quality/latency."""
import argparse
import json
from pathlib import Path
import numpy as np
from interest.evaluate import summarize
from .core import write_json


def report(root):
    parts=[json.loads((root/f'shard{i}.json').read_text()) for i in range(2)]
    assert parts[0]['metadata']==parts[1]['metadata']
    selected=json.loads((root/'sample64.json').read_text())['row_ids']
    methods={}
    rows={}
    for name in parts[0]['rows']:
        r=sorted(parts[0]['rows'][name]+parts[1]['rows'][name],key=lambda x:x['row_id'])
        assert [x['row_id'] for x in r]==selected
        rows[name]=r
        metrics=summarize(r)
        latency=np.array([x['elapsed_seconds'] for x in r])
        metrics.update(mean_seconds=float(latency.mean()),p50_seconds=float(np.median(latency)),
            p95_seconds=float(np.percentile(latency,95)),fraction_with50=float(np.mean([len(x['predict'])==50 for x in r])),
            mean_unique_before50=float(np.mean([x['unique_before_top50'] for x in r])),
            hits50=sum(x['output'] in x['predict'][:50] for x in r))
        if 'max_original_score_error' in r[0]:
            metrics['max_original_score_error']=max(x['max_original_score_error'] for x in r)
        methods[name]=metrics
    base=rows['reference_4x50']
    rng=np.random.RandomState(42)
    bootstrap=rng.randint(0,len(base),(5000,len(base)))
    for name,r in rows.items():
        delta=np.array([float(x['output'] in x['predict'][:50])-float(y['output'] in y['predict'][:50]) for x,y in zip(r,base)])
        methods[name]['paired_HR50_delta_95ci']=np.percentile(delta[bootstrap].mean(1),[2.5,97.5]).tolist()
        methods[name]['speedup']=methods['reference_4x50']['mean_seconds']/methods[name]['mean_seconds']
        methods[name]['same_ranking_as_reference_fraction']=float(np.mean([x['predict']==y['predict'] for x,y in zip(r,base)]))
    old=Path('outputs/latent_qwen3_0.6b_balanced_vq15/eval/valid/merged.json')
    previous=json.loads(old.read_text())
    by_id={x['row_id']:x for x in previous['rows']}
    reference_check=dict(targets_match=all(x['output']==by_id[x['row_id']]['output'] for x in base),
        identical_top50_fraction=float(np.mean([x['predict']==by_id[x['row_id']]['predict'] for x in base])))
    output=dict(metadata=parts[0]['metadata'],sample_count=len(base),metrics=methods,
        original_fullvalid_reference_check=reference_check,
        mean_selected_latent_mass_full_vocab=float(np.mean([r['selected_latent_mass_full_vocab'] for r in base])),
        peak_reserved_gib_per_worker=[p['peak_reserved_gib'] for p in parts],
        caution='Small validation diagnostic; paired bootstrap is descriptive, not a powered model-selection study. Stage-composed batch1 latency excludes model load, tokenization, file IO and small CPU ranking time; shared stages reused across methods. Actual standalone finalists require replay.')
    write_json(root/'comparison.json',output)
    lines=['# Fixed-sample validation inference pilot','',f'Samples: {len(base)}; seed42.','',
        '| Method | HR10 | HR50 | NDCG50 | Hits50 | Mean s | P95 s | Speedup | Full50 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,m in methods.items():
        lines.append(f"| {name} | {m['HR@10']:.5f} | {m['HR@50']:.5f} | {m['NDCG@50']:.5f} | {m['hits50']} | {m['mean_seconds']:.3f} | {m['p95_seconds']:.3f} | {m['speedup']:.2f} | {m['fraction_with50']:.0%} |")
    lines+=['',output['caution'],'',json.dumps(reference_check)]
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    report(Path(p.parse_args().root))
