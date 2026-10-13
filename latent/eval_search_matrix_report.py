"""Fixed-sample decoding comparison and descriptive paired diagnostics."""
import argparse,json,math
from pathlib import Path
import numpy as np
from interest.evaluate import summarize
from .core import write_json


def reduce(rows):
 raw=summarize(rows)
 m={k:raw[k] for k in ['Recall@5','Recall@10','NDCG@5','NDCG@10','examples']}
 s=np.array([r['elapsed_seconds'] for r in rows])
 m.update(mean_seconds=float(s.mean()),p95_seconds=float(np.percentile(s,95)),
    fraction_with10=float(np.mean([len(r['predict'])>=10 for r in rows])),
    mean_candidates=float(np.mean([len(r['predict']) for r in rows])),
    hits5=sum(r['output'] in r['predict'][:5] for r in rows),hits10=sum(r['output'] in r['predict'][:10] for r in rows))
 return m


def report(root,phase='screen'):
 names=['nonbalanced_vq','balanced_vq','balanced_rqkmeans','no_latent_sft'] if phase=='screen' else ['nonbalanced_vq','balanced_vq','balanced_rqkmeans']
 data={name:json.loads((root/f'{phase}_{name}.json').read_text()) for name in names}
 expected=json.loads((root/('sample256.json' if phase=='screen' else 'sample64.json')).read_text())['row_ids']
 results={};reference=None
 for name,d in data.items():
  results[name]={}
  for method,rs in d['rows'].items():
   rows=sorted(rs,key=lambda r:r['row_id']);assert [r['row_id'] for r in rows]==expected
   identity=[(r['row_id'],r['output']) for r in rows]
   if reference is None:reference=identity
   assert identity==reference
   results[name][method]=reduce(rows)
 output=dict(phase=phase,examples=len(expected),split='valid',seed=42,results=results,
    timing='Shared stages composed screen; independently timed complete request confirmation64. One worker per GPU, excluded warmup, CUDA synchronized.',
    scope='All metrics use top10 stored rankings. Matches OneRecThink Recall/NDCG@5/10 cutoff only; beam10 conditional SID and joint latentSID are distinct protocols. Different beam widths/aggregation and BF16batch shapes change predictions.',
    source_metadata={name:d['metadata'] for name,d in data.items()})
 write_json(root/f'{phase}_comparison.json',output)
 lines=[f'# {phase}: eval search comparison on {len(expected)} fixedvalidation rows','',
    '| Model | Method | Recall5 | Recall10 | NDCG5 | NDCG10 | Mean s | P95 s | Full10 |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for name,methods in results.items():
  for method,m in methods.items():lines.append(f"| {name} | {method} | {m['Recall@5']:.5f} | {m['Recall@10']:.5f} | {m['NDCG@5']:.5f} | {m['NDCG@10']:.5f} | {m['mean_seconds']:.3f} | {m['p95_seconds']:.3f} | {m['fraction_with10']:.0%} |")
 lines+=['',output['timing'],output['scope'],'Exploratory fixed256 validation; small hit counts, no testing or training changes.']
 (root/f'{phase}_comparison.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
 return output

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--phase',default='screen');o=p.parse_args();report(Path(o.root),o.phase)
