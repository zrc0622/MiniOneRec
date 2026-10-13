"""Compare frozen VQ variants and no-latent SFT on identical validation rows."""
import argparse,json
from pathlib import Path
import numpy as np
from interest.evaluate import summarize
from .core import write_json


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);root=Path(p.parse_args().root)
 parts=[json.loads((root/f'nonbalanced_shard{i}.json').read_text()) for i in range(2)]
 assert parts[0]['metadata']==parts[1]['metadata']
 rows=sorted(parts[0]['rows']['fast80_observed_sum']+parts[1]['rows']['fast80_observed_sum'],key=lambda x:x['row_id'])
 ids=json.loads((root/'sample256.json').read_text())['row_ids']
 assert [x['row_id'] for x in rows]==ids
 base=json.loads((root/'no_latent_comparison.json').read_text())
 assert all(x['row_id']==y['row_id'] and x['output']==y['output'] for x,y in zip(rows,base['rows']))
 m=summarize(rows);m.update(hits50=sum(x['output'] in x['predict'][:50] for x in rows),fraction_with50=float(np.mean([len(x['predict'])==50 for x in rows])))
 sample64=set(json.loads((root/'sample64.json').read_text())['row_ids'])
 seconds=[x['elapsed_seconds'] for x in rows if x['row_id'] in sample64]
 m.update(mean_seconds64=float(np.mean(seconds)),p95_seconds64=float(np.percentile(seconds,95)))
 baseline=dict(base['no_latent_sft']);baseline.update(mean_seconds64=base['no_latent_latency_matched64']['mean_seconds'],p95_seconds64=base['no_latent_latency_matched64']['p95_seconds'])
 balanced=dict(base['latent_balanced_vq80sum']);balanced.update(mean_seconds64=base['latent80sum_latency64']['mean_seconds'],p95_seconds64=base['latent80sum_latency64']['p95_seconds'])
 results={'no_latent_sft':baseline,'nonbalanced_vq_cached80sum':m,'balanced_vq_cached80sum':balanced}
 rq_paths=[root/f'rq_shard{i}.json' for i in range(2)]
 if all(p.exists() for p in rq_paths):
  parts_rq=[json.loads(p.read_text()) for p in rq_paths]
  rqrows=sorted(parts_rq[0]['rows']['fast80_observed_sum']+parts_rq[1]['rows']['fast80_observed_sum'],key=lambda x:x['row_id'])
  assert [x['row_id'] for x in rqrows]==ids
  assert all(x['output']==y['output'] for x,y in zip(rqrows,base['rows']))
  rm=summarize(rqrows)
  times=[x['elapsed_seconds'] for x in rqrows if x['row_id'] in sample64]
  rm.update(hits50=sum(x['output'] in x['predict'][:50] for x in rqrows),fraction_with50=float(np.mean([len(x['predict'])==50 for x in rqrows])),mean_seconds64=float(np.mean(times)),p95_seconds64=float(np.percentile(times,95)))
  results['balanced_rqkmeans_cached80sum']=rm
 report=dict(examples=256,split='valid',seed=42,matched_targets=True,results=results,nonbalanced_rows=rows,
    metadata=parts[0]['metadata'],caveat='Same256validation rows; samecached80observed-pathsum for VQ variants; native no-latent beam50. Independent timing matched64subset on4/6GPUs. One checkpoint per variant; small sample no causal/statistical claim. Cached BF16 is not bitwise same as uncached evaluation.')
 write_json(root/'vq_variants_comparison.json',report)
 lines=['# Same validation256: no latent, nonbalanced VQ, balanced VQ and RQ-kmeans','', '| Model | HR10 | HR50 | NDCG50 | Hits50 | Full50 | Mean sec(64) | P95 sec(64) |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for name,x in results.items():
  lines.append(f"| {name} | {x['HR@10']:.5f} | {x['HR@50']:.5f} | {x['NDCG@50']:.5f} | {x['hits50']} | {x['fraction_with50']:.0%} | {x['mean_seconds64']:.3f} | {x['p95_seconds64']:.3f} |")
 lines+=['',report['caveat']];(root/'vq_variants_comparison.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))

if __name__=='__main__':main()
