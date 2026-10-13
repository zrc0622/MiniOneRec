"""Verify matched samples and report six metrics for both latent protocols."""
import argparse
import json
from pathlib import Path
from interest.evaluate import summarize


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);o=p.parse_args();root=Path(o.root)
    sample=json.loads((root/'sample.json').read_text())['row_ids'];expected=set(sample)
    data={name:json.loads((root/f'{name}.json').read_text()) for name in ['two_slot_vq_sft','three_slot_vq_sft','no_latent_sft']}
    metrics=[f'{m}@{k}' for m in ['Recall','NDCG'] for k in [5,10,20]]
    results={};targetmap=None
    for name,d in data.items():
        results[name]={}
        for method,rows in d['rows'].items():
            if len(rows)!=len(sample) or {r['row_id'] for r in rows}!=expected:raise ValueError('Sample mismatch')
            targets={r['row_id']:r['output'] for r in rows}
            if targetmap is None:targetmap=targets
            elif targets!=targetmap:raise ValueError('Target mismatch')
            if any(len(set(r['predict']))!=len(r['predict']) for r in rows):raise ValueError('Duplicate predictions')
            summary=summarize(rows)
            results[name][method]={k:summary[k] for k in metrics}
            results[name][method].update(fraction_with20=sum(len(r['predict'])>=20 for r in rows)/len(rows),
                mean_amortized_seconds=sum(r['elapsed_seconds'] for r in rows)/len(rows))
    lines=[f'# 两层非平衡 VQ SFT 对照（{len(sample)} 条验证样本）','',
        '两层与三层非平衡 VQ 使用相同数据、全局 batch、学习率和15轮训练。两层量化器重新训练，每槽256码、32维。无latent SFT使用直接SID beam50，作为两种协议的共同参照。',
        '', '训练中的20%验证、每轮一次仅用于监控；两层使用第15轮最终权重。新旧验证loss样本范围不同，不直接比较原日志均值。推理时间为批量吞吐摊销，非单用户延迟。','']
    for method,title in [('single20','1×20'),('four20_marginal','4×20跨兴趣完整求和')]:
        lines.extend([f'## {title}','', '| 模型 | R@5 | R@10 | R@20 | N@5 | N@10 | N@20 | 满20比例 |','|---|---:|---:|---:|---:|---:|---:|---:|'])
        for name,label in [('no_latent_sft','无latent SFT'),('three_slot_vq_sft','三层非balanced VQ SFT'),('two_slot_vq_sft','两层非balanced VQ SFT')]:
            row=results[name]['direct50' if name=='no_latent_sft' else method]
            lines.append('| '+label+' | '+' | '.join(f'{row[k]:.5f}' for k in metrics)+f" | {row['fraction_with20']:.1%} |")
        lines.append('')
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    (root/'comparison.json').write_text(json.dumps(dict(examples=len(sample),results=results,verified_same_ids_and_targets=True),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(results,ensure_ascii=False))

if __name__=='__main__':main()
