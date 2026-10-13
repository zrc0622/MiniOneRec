#!/usr/bin/env python3
"""Compare old/new item SID SFT on fixed validation rows, using canonical item IDs."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from run_two_slot_vq_sft import PROJECT, PYTHON, DATA, INDEX, STEM, NATIVE, gpu_snapshot, save
from latent.core import load_index, sha
from interest.evaluate import summarize

EXPERIMENT = PROJECT / 'outputs/amazon23_industrial_qwen3_0.6b_sft_lasttoken_l2_balanced15_20261011'
NEW_DATA = PROJECT / 'outputs/amazon23_industrial_qwen3_4b_lasttoken_l2_sid_20261010/dataset'
CONTROLS = PROJECT / 'outputs/latent_qwen3_0.6b_vq2_15ep_20261010/eval/valid20pct'


def validate_alignment(old, new):
    if len(old) != len(new):
        raise ValueError('Different validation population')
    for a, b in zip(old, new):
        if ({k: v for k, v in a.items() if k not in ('item_sid', 'history_item_sid')}
                != {k: v for k, v in b.items() if k not in ('item_sid', 'history_item_sid')}):
            raise ValueError('Validation rows differ beyond SID remapping')


def canonical_rows(result, index, source, ids):
    inverse = {''.join(tokens): item for item, tokens in index.items()}
    if len(inverse) != len(index):
        raise ValueError('SID index must uniquely identify real items')
    if set(result['rows']) != {'direct50'}:
        raise ValueError('Expected common native SID beam50 protocol')
    rows = result['rows']['direct50']
    if [r['row_id'] for r in rows] != ids:
        raise ValueError('Missing, reordered or different evaluation rows')
    converted = []
    for row in rows:
        target = inverse[row['output']]
        if target != source[row['row_id']]['item_id']:
            raise ValueError('SID-decoded target differs from the source item ID')
        predictions = [inverse[s] for s in row['predict']]
        if len(predictions) != len(set(predictions)):
            raise ValueError('Duplicate real item predictions')
        converted.append(dict(row, output=target, predict=predictions))
    return converted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', default='5')
    parser.add_argument('--root', default=str(EXPERIMENT / 'rank_eval_20261012'))
    o = parser.parse_args()
    root = Path(o.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'new_sid.json').exists():
        raise ValueError('Existing evaluation output; preserve it')
    new_index_path = NEW_DATA / 'Industrial_and_Scientific/Industrial_and_Scientific.index.json'
    old_file, new_file = [p / 'valid' / f'{STEM}.csv' for p in (DATA, NEW_DATA)]
    with old_file.open() as f:
        old_source = list(csv.DictReader(f))
    with new_file.open() as f:
        new_source = list(csv.DictReader(f))
    validate_alignment(old_source, new_source)
    sample = json.loads((CONTROLS / 'sample.json').read_text())
    shutil.copy2(CONTROLS / 'sample.json', root / 'sample.json')
    old = json.loads((CONTROLS / 'no_latent_sft.json').read_text())
    meta = old['metadata']
    expected = dict(sample_sha256=sha(root / 'sample.json'), data_sha256=sha(old_file),
                    index_sha256=sha(INDEX), source_sha256=sha(PROJECT / 'latent/eval_unified_small.py'),
                    protocol=json.loads((PROJECT / 'eval_metric_protocol.json').read_text()))
    if Path(meta['model']).resolve() != NATIVE.resolve() or meta['kind'] != 'sid' or any(meta[k] != v for k, v in expected.items()):
        raise ValueError('Historical baseline provenance changed')
    baseline = canonical_rows(old, load_index(INDEX), old_source, sample['row_ids'])
    snapshot = root / 'source_snapshot'
    snapshot.mkdir()
    for name in ('latent', 'interest'):
        shutil.copytree(PROJECT / name, snapshot / name, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('data.py', 'eval_metric_protocol.json'):
        shutil.copy2(PROJECT / name, snapshot / name)
    shutil.copy2(Path(__file__), snapshot / Path(__file__).name)
    command = [PYTHON, '-u', '-m', 'latent.eval_unified_small', '--model', str(EXPERIMENT / 'sft/final_checkpoint'),
               '--data', str(new_file), '--index', str(new_index_path), '--sample', str(root / 'sample.json'),
               '--output', str(root / 'new_sid.json'), '--kind', 'sid', '--max-batch', '32']
    save(root / 'manifest.json', dict(gpu=o.gpu, examples=len(sample['row_ids']), split='fixed20pct_valid',
         sample_sha256=sha(root / 'sample.json'), old_data_sha256=sha(old_file), new_data_sha256=sha(new_file),
         old_index_sha256=sha(INDEX), new_index_sha256=sha(new_index_path),
         comparison='canonical real item IDs; same rows/targets, direct50 top20 both models',
         old_control=str(CONTROLS / 'no_latent_sft.json'), baseline_timing='historical, not a same-hardware speed comparison'))
    save(root / 'eval.command.json', command)
    for check in range(3):
        gpu = gpu_snapshot([o.gpu])[o.gpu]
        if gpu['used_mib'] >= 1024 or gpu['utilization'] >= 10 or gpu['pids']:
            raise RuntimeError('Selected GPU became occupied; do not share another task')
        if check < 2:
            time.sleep(5)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=o.gpu, PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', PYTHONPATH=str(snapshot) + ':' + str(PROJECT))
    with (root / 'eval.log').open('w') as log:
        worker = subprocess.Popen(command, cwd=snapshot, env=env, stdin=subprocess.DEVNULL,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save(root / 'status.json', dict(phase='ranking_eval', worker_pid=worker.pid, coordinator_pid=os.getpid(), gpu=o.gpu))
        code = worker.wait()
    (root / 'eval.exit_code').write_text(str(code) + '\n')
    if code:
        raise RuntimeError('Ranking evaluation failed; inspect eval.log')
    new = json.loads((root / 'new_sid.json').read_text())
    for key, value in dict(sample_sha256=sha(root / 'sample.json'), data_sha256=sha(new_file),
                           index_sha256=sha(new_index_path), source_sha256=expected['source_sha256'],
                           protocol=expected['protocol']).items():
        if new['metadata'][key] != value:
            raise ValueError('New evaluation provenance changed')
    ranked = canonical_rows(new, load_index(new_index_path), new_source, sample['row_ids'])
    results = {}
    metrics = [f'{m}@{k}' for m in ('Recall', 'NDCG') for k in (5, 10, 20)]
    for name, rows in [('old_sid_sft', baseline), ('new_sid_sft', ranked)]:
        values = summarize(rows)
        results[name] = {k: values[k] for k in metrics}
        results[name].update(full20=sum(len(r['predict']) >= 20 for r in rows) / len(rows),
                             hits20=sum(r['output'] in r['predict'][:20] for r in rows))
    save(root / 'comparison.json', dict(results=results, examples=len(ranked), canonical_item_ids=True,
         paired_targets_verified=True, new_oom_retries=new['oom_retries']))
    lines = ['# 新旧商品 SID 无 latent SFT 对照', '',
             '相同3232条验证样本，统一beam50后取top20；SID映射回真实商品ID核对。旧模型结果复用已核验的历史评估。', '',
             '| 模型 | R@5 | R@10 | R@20 | N@5 | N@10 | N@20 | 满20比例 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, label in [('old_sid_sft', '旧SID'), ('new_sid_sft', '新SID')]:
        r = results[name]
        lines.append('| ' + label + ' | ' + ' | '.join(f'{r[k]:.5f}' for k in metrics) + f" | {r['full20']:.1%} |")
    (root / 'comparison.md').write_text('\n'.join(lines) + '\n')
    save(root / 'status.json', dict(phase='complete', completed_at=time.time(), gpu=o.gpu, results=results))
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
