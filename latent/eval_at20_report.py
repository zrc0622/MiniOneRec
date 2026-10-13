"""Merge complete top20 artifacts, including supplemental BinaryRL."""
import argparse,json
from pathlib import Path
import numpy as np
from interest.evaluate import summarize
KEYS=['Recall@5','Recall@10','Recall@20','NDCG@5','NDCG@10','NDCG@20']
def metrics(rows):
 raw=summarize(rows);result={k:raw[k] for k in KEYS}
 result.update(examples=len(rows),mean_seconds=float(np.mean([r['elapsed_seconds'] for r in rows])),fraction_with20=float(np.mean([len(r['predict'])>=20 for r in rows])),mean_returned_unique=float(np.mean([len(r['predict']) for r in rows])),mean_unique_proposals=float(np.mean([r['unique_before_top20'] for r in rows])) if all('unique_before_top20' in r for r in rows) else None,hits20=sum(r['output'] in r['predict'][:20] for r in rows))
 return result

def render(title,results,note):
 lines=[title,'','| Model | Method | R5 | R10 | R20 | N5 | N10 | N20 | Mean s | Full20 | Mean unique proposals |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for name,ms in results.items():
  for method,m in ms.items():
   cells=' | '.join(f'{m[k]:.5f}' for k in KEYS);unique=m['mean_unique_proposals'];label='—' if unique is None else f'{unique:.2f}'
   lines.append(f"| {name} | {method} | {cells} | {m['mean_seconds']:.3f} | {m['fraction_with20']:.1%} | {label} |")
 return '\n'.join(lines+['',note,'Proposalcount is only shown when recorded beforetop20 truncation; — means not recorded, not zero.'])+'\n'

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);o=p.parse_args();root=Path(o.root)
 ids=json.loads(Path('outputs/eval_search_compare_20261009/sample256.json').read_text())['row_ids'];results={};sources={};identity=None
 for name in ['nonbalanced_vq','dense_2465','binary_final','no_latent_sft','no_latent_rl']:
  d=json.loads((root/f'screen_{name}.json').read_text());assert d['metadata']['top_k']==20;results[name]={};sources[name]=d['metadata']
  for method,rs in d['rows'].items():
   assert [r['row_id'] for r in rs]==ids
   current=[(r['row_id'],r['output']) for r in rs]
   if identity is None:identity=current
   assert identity==current
   assert all(len(r['predict'])<=20 and len(r['predict'])==len(set(r['predict'])) for r in rs)
   results[name][method]=metrics(rs)
 note='Fixed256valid seed42, singlepositive Recall=HR. All rankings newly retainedtop20, no test/training/attention changes. LatentSFT/Dense screen mean times compose sharedstages; Binary and no-latent are independently timed complete requests. Selectedlatentmethods have separate nested64 whole-request confirmation. Dense2465 vs Binary/no-latentRL final4928 are different trainingsteps. Partialcandidatefullness explicitlyreported; diagnostic sample, no significance claim.'
 out=dict(examples=256,results=results,source_metadata=sources,note=note)
 (root/'comparison.json').write_text(json.dumps(out,indent=2));(root/'comparison.md').write_text(render('# SFT/RL eval@5/10/20 on256 matchedvalidation',results,note))
 # Correct legacy screen report columns that used returnedcount as fallback for unrecorded proposalcount.
 for phase in ['screen','confirm']:
  d=json.loads((root/f'{phase}_comparison.json').read_text())
  for name,ms in d['results'].items():
   raw=json.loads((root/f'{phase}_{name}.json').read_text())
   for method,m in ms.items():m['mean_unique_proposals']=metrics(raw['rows'][method])['mean_unique_proposals']
  (root/f'{phase}_comparison.json').write_text(json.dumps(d,indent=2));(root/f'{phase}_comparison.md').write_text(render(f'# {phase} eval@20, {d["examples"]} fixedvalidation rows',d['results'],d['scope']))
 print((root/'comparison.md').read_text())
if __name__=='__main__':main()
