"""Re-evaluate stored fixed-sample rankings with Recall/NDCG at5 and10."""
import argparse,json,math
from pathlib import Path
from .core import sha,write_json


def metrics(rows):
    result={}
    for k in (5,10):
        hits=0;discounted=0.
        for row in rows:
            top=row['predict'][:k]
            if row['output'] in top:
                hits+=1
                discounted+=1/math.log2(top.index(row['output'])+2)
        result[f'Recall@{k}']=hits/len(rows)
        result[f'NDCG@{k}']=discounted/len(rows)
        result[f'hits@{k}']=hits
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);o=p.parse_args()
    root=Path(o.root)
    ids=json.loads((root/'sample256.json').read_text())['row_ids']
    names={'no_latent_sft':('no_latent_shard','ordinary50beam'),
        'nonbalanced_vq':('nonbalanced_shard','cached80observed_sum'),
        'balanced_vq':('fast_shard','cached80observed_sum'),
        'balanced_rqkmeans':('rq_shard','cached80observed_sum')}
    results={}; reference=None;source_hashes={};all_rows={}
    for name,(prefix,scheme) in names.items():
        parts=[]
        for i in range(2):
            path=root/f'{prefix}{i}.json'
            part=json.loads(path.read_text());source_hashes[path.name]=sha(path)
            parts.extend(part['rows'] if isinstance(part['rows'],list) else part['rows']['fast80_observed_sum'])
        rows=sorted(parts,key=lambda r:r['row_id'])
        assert [r['row_id'] for r in rows]==ids, name
        identity=[(r['row_id'],r['output']) for r in rows]
        if reference is None:reference=identity
        assert identity==reference,name
        assert all(len(r['predict'])>=10 and len(set(r['predict']))==len(r['predict']) for r in rows),name
        result=metrics(rows);result.update(examples=len(rows),search=scheme,
            fraction_with10=sum(len(r['predict'])>=10 for r in rows)/len(rows))
        results[name]=result
        all_rows[name]=[dict(row_id=r['row_id'],output=r['output'],predict=r['predict'][:10]) for r in rows]
    output=dict(metadata=dict(split='valid',seed=42,examples=256,source_sample_sha256=sha(root/'sample256.json'),
        source_hashes=source_hashes,metric_definition='Single positive target per history: Recall@K=Hit@K; NDCG discounts rank by1/log2(rank+1), IDCG=1.',
        metric_reference='papers/OneRec-Think.pdf section5 and AppendixA.1, Recall/NDCG at5 and10',
        evaluation='Metrics-only re-evaluation of saved full rankings; no new generation or weights changed.',
        caveat='Matches paper metric cutoffs, not paper dataset or beam10 search protocol. OrdinarySFT uses nativebeam50, latent uses predefinedcached80pathsum. Small fixed256validation diagnostic.'),results=results,rows=all_rows)
    dest=root/'onerecthink_metrics';dest.mkdir(exist_ok=True)
    write_json(dest/'comparison.json',output)
    lines=['# OneRec-Think metric cutoffs on identical256 validation examples','',
        '| Model | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 | Hits@5 | Hits@10 |','|---|---:|---:|---:|---:|---:|---:|']
    for name,m in results.items():lines.append(f"| {name} | {m['Recall@5']:.5f} | {m['Recall@10']:.5f} | {m['NDCG@5']:.5f} | {m['NDCG@10']:.5f} | {m['hits@5']} | {m['hits@10']} |")
    lines+=['',output['metadata']['evaluation'],output['metadata']['caveat']]
    (dest/'comparison.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))

if __name__=='__main__':main()
