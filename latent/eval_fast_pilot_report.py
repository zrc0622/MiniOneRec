"""Paired validation report for cached generation pilot."""
import argparse,json
from pathlib import Path
import numpy as np
from interest.evaluate import summarize
from .core import write_json


def report(root):
    parts=[json.loads((root/f'fast_shard{i}.json').read_text()) for i in range(2)]
    assert parts[0]['metadata']==parts[1]['metadata']
    ids=json.loads((root/'sample256.json').read_text())['row_ids']
    pilot=json.loads((root/'comparison.json').read_text())
    measured=[]
    for i in range(2):measured+=json.loads((root/f'shard{i}.json').read_text())['rows']['reference_4x50']
    measured={r['row_id']:r for r in measured}
    original=json.loads(Path('outputs/latent_qwen3_0.6b_balanced_vq15/eval/valid/merged.json').read_text())
    original_by_id={r['row_id']:r for r in original['rows']}
    baseline=[original_by_id[i] for i in ids]
    base=summarize(baseline);base['hits50']=sum(r['output'] in r['predict'][:50] for r in baseline)
    rows={};metrics={}
    rng=np.random.RandomState(42);indices=rng.randint(0,len(ids),(5000,len(ids)))
    for name in parts[0]['rows']:
        r=sorted(parts[0]['rows'][name]+parts[1]['rows'][name],key=lambda x:x['row_id'])
        assert [x['row_id'] for x in r]==ids
        assert all(x['output']==y['output'] for x,y in zip(r,baseline))
        rows[name]=r;m=summarize(r)
        latency=np.array([x['elapsed_seconds'] for x in r])
        matched=[x for x in r if x['row_id'] in measured]
        delta=np.array([float(x['output'] in x['predict'][:50])-float(y['output'] in y['predict'][:50]) for x,y in zip(r,baseline)])
        m.update(mean_seconds=float(latency.mean()),p95_seconds=float(np.percentile(latency,95)),
            same_ranking_as_reference_fraction=float(np.mean([x['predict']==y['predict'] for x,y in zip(r,baseline)])),
            fraction_with50=float(np.mean([len(x['predict'])==50 for x in r])),
            hits50=sum(x['output'] in x['predict'][:50] for x in r),
            paired_HR50_delta_95ci=np.percentile(delta[indices].mean(1),[2.5,97.5]).tolist(),
            speedup_vs_reference_on_matched64=float(np.mean([measured[x['row_id']]['elapsed_seconds'] for x in matched])/np.mean([x['elapsed_seconds'] for x in matched])))
        metrics[name]=m
    output=dict(metadata=parts[0]['metadata'],sample_count=len(ids),metrics=metrics,reference256_metrics=base,
        measured_reference64_metrics=pilot['metrics']['reference_4x50'],
        peak_reserved_gib_per_worker=[p['peak_reserved_gib'] for p in parts],
        caution='256 fixed validation examples; no test selection. Speedup uses64 paired requests on same GPUs in consecutive runs, not concurrent loads. Shared-stage composed batch1 latency excludes tokenization/IO/small CPU sort, warmup excluded. Tiny sample confidence intervals descriptive. No attention backend changed. Original attention backend recorded.')
    write_json(root/'fast_comparison.json',output)
    lines=['# Cached generation validation pilot','',f'Samples: {len(ids)} seed42; direct speed reference measured on64 nested samples.','',
        '| Method | HR10 | HR50 | NDCG50 | Hits50 | Mean s | P95 s | Speedup64 | Full50 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    lines.append(f"| original4x50 (quality256; time64) | {base['HR@10']:.5f} | {base['HR@50']:.5f} | {base['NDCG@50']:.5f} | {base['hits50']} | {pilot['metrics']['reference_4x50']['mean_seconds']:.3f} | {pilot['metrics']['reference_4x50']['p95_seconds']:.3f} | 1.00 | 100% |")
    for n,m in metrics.items():
        lines.append(f"| {n} | {m['HR@10']:.5f} | {m['HR@50']:.5f} | {m['NDCG@50']:.5f} | {m['hits50']} | {m['mean_seconds']:.3f} | {m['p95_seconds']:.3f} | {m['speedup_vs_reference_on_matched64']:.2f} | {m['fraction_with50']:.0%} |")
    lines+=['',output['caution']]
    (root/'fast_comparison.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);report(Path(p.parse_args().root))
