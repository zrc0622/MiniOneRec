"""Consolidate pilot quality and independent request latency with provenance."""
import argparse,json
from pathlib import Path
import numpy as np
from interest.evaluate import summarize
from .core import write_json,sha


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);root=Path(p.parse_args().root)
 fast=json.loads((root/'fast_comparison.json').read_text());replay=json.loads((root/'replay_comparison.json').read_text())
 single=[json.loads((root/f'single_shard{i}.json').read_text()) for i in range(2)]
 rs=sorted(single[0]['rows']['fast_1x50']+single[1]['rows']['fast_1x50'],key=lambda r:r['row_id'])
 assert [r['row_id'] for r in rs]==json.loads((root/'sample64.json').read_text())['row_ids']
 seconds=np.array([r['elapsed_seconds'] for r in rs]);one=summarize(rs)
 one.update(mean_seconds=float(seconds.mean()),p95_seconds=float(np.percentile(seconds,95)))
 replay['metrics']['fast_1x50']=one
 times=replay['metrics'];rows=[]
 rows.append(dict(method='original_4x50',quality256=fast['reference256_metrics'],latency64=times['reference_4x50']))
 for name in ['fast_1x50','fast_2x50','fast_4x50','fast80_observed_sum']:
  rows.append(dict(method=name,quality256=fast['metrics'][name],latency64=times[name]))
 table=['# Recommendation inference evaluation pilot','',
 'Model: balanced parallel VQ-VAE SFT final checkpoint. Validation only, seed42,256/16162 samples(1.58%). Independent complete request latency measured on64 nested samples, GPUs4/6, one worker perGPU. Default SDPA attention unchanged.','',
 '| Method | HR10 (256) | HR50 (256) | NDCG50 (256) | Mean seconds (64) | P95 seconds (64) | Speedup |',
 '|---|---:|---:|---:|---:|---:|---:|']
 for r in rows:
  q,t=r['quality256'],r['latency64'];speed=times['reference_4x50']['mean_seconds']/t['mean_seconds']
  table.append(f"| {r['method']} | {q['HR@10']:.5f} | {q['HR@50']:.5f} | {q['NDCG@50']:.5f} | {t['mean_seconds']:.3f} | {t['p95_seconds']:.3f} | {speed:.2f} |")
 marginal=fast['metrics']['fast80_marginal'];mb=times['fast80_marginal_batch32']
 table+=['',f"Cross-latent marginal scoring with batch8: quality on256 HR50={marginal['HR@50']:.5f}, NDCG50={marginal['NDCG@50']:.5f}, stage-composed latency={marginal['mean_seconds']:.3f}s. Independent batch32 replay on64: mean={mb['mean_seconds']:.3f}s, P95={mb['p95_seconds']:.3f}s, HR50={mb['HR@50']:.5f}, NDCG50={mb['NDCG@50']:.5f}. Batch32 quality has only64 examples; do not combine it with256-row quality as if identical settings.", '',
 'All methods produced50 distinct products for every sampled request. All methods on256 hit the same21 gold items atHR50; differences come from ordering. Cached BF16 generation/scoring changes numerical arithmetic and rankings; full ranking agreement is0 on256, top50 set overlap recorded per method. It is mathematically equivalent search, not bitwise identical implementation.', '',
 'Observed-path sum is a truncated joint-path sum, not complete marginal probability over candidate items. Budget50+10+10+10 was predefined. Exploratory comparisons on one checkpoint, not evidence of universal superiority. Paired bootstrap intervals are descriptive, multiple comparisons uncorrected.', '',
 'Provisional inference choice: fast1x50 for latency. Quality-oriented candidate:80-path observed sum, pending larger fixed validation and second checkpoint confirmation. Exact-ish cross-latent marginal provides no clear quality gain here. Production batching/throughput and tail latency under load remain unmeasured.', '',
 'Correctness:5 tiny-real-Qwen tests passed covering cached scoring, variable SID length, cache immutability, beam score equivalence, search modes; production64 reference exactly matched existing fullvalid predictions. No training or production evaluator changed.']
 output=dict(rows=rows,marginal_batch8_256=marginal,marginal_batch32_64=mb,standalone_replay64=replay,
   quality_source='fast_comparison.json',samples256_sha256=sha(root/'sample256.json'),samples64_sha256=sha(root/'sample64.json'),
   source_hashes={name:sha(Path(__file__).parent/name) for name in ['eval_pilot.py','eval_cached_beam.py','eval_fast_pilot.py','eval_pilot_replay.py']},
   interpretation='Provisional diagnostic on one balancedVQ checkpoint; full50 candidates throughout; one latent fastest, observed80sum best measured NDCG; full marginal not superior. Cache numerics change rankings.')
 write_json(root/'summary.json',output);(root/'summary.md').write_text('\n'.join(table)+'\n');print('\n'.join(table))

if __name__=='__main__':main()
