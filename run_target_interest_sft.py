#!/usr/bin/env python3
"""Target-conditioned mean VQ labels -> matched native latent SFT -> two evals."""
import argparse
import fcntl
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from run_two_slot_vq_sft import PROJECT, PYTHON, DATA, INDEX, ITEMS, STEM, REFERENCE, NATIVE, gpu_snapshot, save
from latent.core import load_index, load_labels, sha
from interest.evaluate import summarize

LABELS = PROJECT / 'outputs/latent_qwen3_0.6b_target_vq3_20261010/labels'
CONTROLS = PROJECT / 'outputs/latent_qwen3_0.6b_vq2_15ep_20261010/eval'
METHODS = ['single20', 'four20_marginal']


def training_command(root, snapshot, batch=32):
    if batch not in (16, 32):
        raise ValueError('Supported batches preserve effective batch 1024 on two ranks')
    return [PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
            '-m', 'latent.train', 'sft', '--model', str(PROJECT / 'models/Qwen3-0.6B'),
            '--output_dir', str(root / 'sft'), '--labels', str(LABELS),
            '--train_file', str(DATA / 'train' / f'{STEM}.csv'),
            '--eval_file', str(DATA / 'valid' / f'{STEM}.csv'), '--items', str(ITEMS),
            '--index', str(INDEX), '--batch', str(batch),
            '--gradient_accumulation', str(1024 // (2 * batch)), '--eval_batch', '16',
            '--epochs', '15', '--lr', '.0003', '--seed', '42', '--latent_weight', '1',
            '--sft_eval_fraction', '1', '--sft_eval_every_epochs', '.5',
            '--sft_save_every_epochs', '.5', '--sft_early_stopping_patience', '3',
            '--deepspeed', str(snapshot / 'config/sft_zero2.json')]


def check_labels():
    index = load_index(INDEX)
    for split in ('train', 'valid', 'test'):
        _, m = load_labels(LABELS, split, DATA / split / f'{STEM}.csv', index)
    recipe = m['embedding_recipe']
    if (m['k'] != 3 or m['codebook_size'] != 256 or m['vq_options'].get('balance_epsilon', 0)
            or recipe.get('embedding_input') != 'history_target' or not recipe.get('contains_target')
            or recipe['pooling'] != 'masked_mean' or recipe['normalization'] != 'none'):
        raise ValueError('Experiment2 must use old-SID target-conditioned mean/noL2 labels')
    return m


def validate_result(result, sample, expected_model=None, kind=None):
    expected = sample['row_ids']
    if len(expected) != len(set(expected)):
        raise ValueError('Repeated sample IDs')
    if expected_model is not None and Path(result['metadata']['model']).resolve() != Path(expected_model).resolve():
        raise ValueError('Wrong evaluated checkpoint')
    if kind is not None and result['metadata']['kind'] != kind:
        raise ValueError('Wrong model kind')
    methods = METHODS if result['metadata']['kind'] == 'latent' else ['direct50']
    if set(result['rows']) != set(methods):
        raise ValueError('Missing required evaluation method')
    targets = None
    for rows in result['rows'].values():
        if len(rows) != len(expected) or {r['row_id'] for r in rows} != set(expected):
            raise ValueError('Incomplete or different evaluation sample')
        target = {r['row_id']: r['output'] for r in rows}
        if targets is not None and target != targets:
            raise ValueError('Methods disagree on targets')
        targets = target
        if any(len(r['predict']) != len(set(r['predict'])) for r in rows):
            raise ValueError('Duplicate item predictions')
    return targets


def merge_shards(parts, samples, full_sample):
    from itertools import chain
    if len(parts) != len(samples):
        raise ValueError('Shard count mismatch')
    for part, sample in zip(parts, samples):
        validate_result(part, sample)
    seen = list(chain.from_iterable(s['row_ids'] for s in samples))
    if len(seen) != len(set(seen)) or set(seen) != set(full_sample['row_ids']):
        raise ValueError('Overlapping or incomplete shards')
    keys = ('model', 'kind', 'methods', 'data_sha256', 'index_sha256', 'source_sha256', 'protocol')
    if any(any(p['metadata'][k] != parts[0]['metadata'][k] for k in keys) for p in parts[1:]):
        raise ValueError('Shard provenance mismatch')
    result = dict(rows={}, batches={}, oom_retries=[], metadata=dict(parts[0]['metadata']))
    for method in parts[0]['rows']:
        rowmap = {r['row_id']: r for p in parts for r in p['rows'][method]}
        result['rows'][method] = [rowmap[rid] for rid in full_sample['row_ids']]
        result['batches'][method] = list(chain.from_iterable(p['batches'][method] for p in parts))
    result['metadata'].update(examples=len(seen), completed={m: len(seen) for m in result['rows']},
                              shards=len(parts), shard_sample_sha256=[p['metadata']['sample_sha256'] for p in parts])
    result['oom_retries'] = list(chain.from_iterable(p.get('oom_retries', []) for p in parts))
    validate_result(result, full_sample)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--gpus', default='5,6')
    p.add_argument('--eval-only', action='store_true', help='Evaluate an already completed SFT on a new GPU pair')
    args = p.parse_args()
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpus = args.gpus.split(',')
    if len(gpus) != 2 or len(set(gpus)) != 2:
        p.error('Two distinct GPUs required')
    if args.eval_only and not (root / 'sft/final_checkpoint/tokenizer_config.json').is_file():
        raise ValueError('Evaluation-only recovery requires a completed SFT checkpoint')
    m = check_labels()
    config = dict(gpus=gpus, labels=str(LABELS), labels_sha256=sha(LABELS / 'labels.json'),
                  data=str(DATA), pretrained_model=str(PROJECT / 'models/Qwen3-0.6B'),
                  epochs=15, global_batch=1024, initial_batch=32, accumulation=16, lr=.0003,
                  latent_weight=1, seed=42, validation_fraction=1, validation_every_epochs=.5,
                  saving_every_epochs=.5, checkpoint_selection='best full valid loss, patience3',
                  label_recipe=m['embedding_recipe'], attention='unchanged', methods=METHODS,
                  ranking_samples=[256, 3232], baseline_controls='checked historical outputs; timings historical',
                  no_new_vq=True, rl_authorized=False)
    if (root / 'pipeline_config.json').exists():
        previous = json.loads((root / 'pipeline_config.json').read_text())
        compared = dict(config)
        if args.eval_only:
            compared['gpus'] = previous['gpus']
        if previous != compared:
            raise ValueError('Experiment configuration changed')
    if not args.eval_only:
        save(root / 'pipeline_config.json', config)
    (root / 'pipeline.pid').write_text(str(os.getpid()) + '\n')
    snapshot = root / 'source_snapshot'
    if not snapshot.exists():
        snapshot.mkdir()
        for folder in ('latent', 'interest', 'rq', 'config'):
            shutil.copytree(PROJECT / folder, snapshot / folder, ignore=shutil.ignore_patterns('__pycache__'))
        for file in ('data.py', 'deepspeed_cleanup.py', 'eval_metric_protocol.json', Path(__file__).name):
            shutil.copy2(PROJECT / file, snapshot / file)
        for folder in ('models', 'data'):
            (snapshot / folder).symlink_to(PROJECT / folder, target_is_directory=True)
        save(snapshot / 'hashes.json', {str(f.relative_to(snapshot)): sha(f)
             for f in snapshot.rglob('*') if f.is_file() and not f.is_symlink()})
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, PYTHONUNBUFFERED='1',
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', NCCL_IB_DISABLE='1',
               PYTHONPATH=str(snapshot) + ':' + str(PROJECT),
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    env['PATH'] = str(Path(PYTHON).parent) + ':' + env.get('PATH', '')
    if not Path(env.get('CUDA_HOME', '') + '/bin/nvcc').is_file():
        env['CUDA_HOME'] = str(Path(PYTHON).parent.parent)

    def status(phase, **extra):
        payload = dict(phase=phase, updated_at=time.time(), coordinator_pid=os.getpid(), gpus=gpus, **extra)
        save(root / 'pipeline_status.json', payload)
        print(json.dumps(payload, ensure_ascii=False), flush=True)

    def wait_idle():
        good = 0
        while good < 3:
            snap = gpu_snapshot(gpus)
            good = good + 1 if all(v['used_mib'] < 1024 and v['utilization'] < 10 and not v['pids'] for v in snap.values()) else 0
            status('waiting_for_idle_gpus', measurements=snap, consecutive_idle_checks=good)
            if good < 3:
                time.sleep(10 if good else 30)

    def start(command, log, gpu=None):
        save(log.with_suffix('.command.json'), command)
        stream = log.open('w')
        child_env = dict(env, CUDA_VISIBLE_DEVICES=gpu) if gpu else env
        worker = subprocess.Popen(command, cwd=snapshot, env=child_env, stdin=subprocess.DEVNULL,
                                  stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        log.with_suffix('.pid').write_text(str(worker.pid) + '\n')
        return worker, stream, log

    def finish(worker):
        proc, stream, log = worker
        code = proc.wait()
        stream.close()
        log.with_suffix('.exit_code').write_text(str(code) + '\n')
        return code

    if not (root / 'sft/final_checkpoint/config.json').exists():
        if (root / 'sft').exists():
            raise ValueError('Partial SFT output exists; preserve it and choose a fresh root')
        wait_idle()
        if sha(LABELS / 'labels.json') != config['labels_sha256']:
            raise ValueError('Labels changed while waiting')
        worker = start(training_command(root, snapshot), root / 'sft.log')
        status('training_sft', worker_pid=worker[0].pid, batch_per_device=32,
               accumulation=16, global_batch=1024, validation_rows=16162)
        code = finish(worker)
        if code:
            log = (root / 'sft.log').read_text(errors='replace')
            if 'out of memory' not in log.lower() or re.search(r'[1-9]\d*/4140', log):
                raise RuntimeError('SFT failed; inspect sft.log')
            # Startup-only fallback; never discard optimized weights.
            archive = root / 'startup_oom_batch32'
            archive.mkdir()
            if (root / 'sft').exists():
                (root / 'sft').rename(archive / 'sft')
            for f in root.glob('sft.*'):
                if f.is_file():
                    f.rename(archive / f.name)
            save(root / 'batch_fallback.json', dict(reason='OOM before first completed optimizer step',
                                                    batch=16, accumulation=32, global_batch=1024))
            wait_idle()
            worker = start(training_command(root, snapshot, batch=16), root / 'sft.log')
            status('training_sft', worker_pid=worker[0].pid, batch_per_device=16,
                   accumulation=32, global_batch=1024, validation_rows=16162)
            if finish(worker):
                raise RuntimeError('SFT startup fallback failed; inspect sft.log')
        if not (root / 'sft/final_checkpoint/tokenizer_config.json').exists():
            raise ValueError('Final checkpoint incomplete')
        save(root / 'sft_complete.json', dict(checkpoint=str(root / 'sft/final_checkpoint'), completed_at=time.time()))

    protocol = json.loads((snapshot / 'eval_metric_protocol.json').read_text())
    for scope in ('sample256', 'valid20pct'):
        source = CONTROLS / scope
        dest = root / 'eval' / scope
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / 'sample.json', dest / 'sample.json')
        sample = json.loads((dest / 'sample.json').read_text())
        results = {}
        for name, model, kind in [('history_only_latent_sft', REFERENCE, 'latent'), ('no_latent_sft', NATIVE, 'sid')]:
            filename = 'three_slot_vq_sft.json' if kind == 'latent' else 'no_latent_sft.json'
            cached = json.loads((source / filename).read_text())
            validate_result(cached, sample, model, kind)
            meta = cached['metadata']
            expected = dict(sample_sha256=sha(dest / 'sample.json'), data_sha256=sha(DATA / 'valid' / f'{STEM}.csv'),
                            index_sha256=sha(INDEX), source_sha256=sha(snapshot / 'latent/eval_unified_small.py'), protocol=protocol)
            if any(meta[k] != v for k, v in expected.items()):
                raise ValueError('Historical control evaluation provenance differs; rerun controls before comparison')
            shutil.copy2(source / filename, dest / (name + '.json'))
            results[name] = cached
        wait_idle()
        parts, samples, workers = [], [], []
        for shard, gpu in enumerate(gpus):
            sub = dict(sample, row_ids=sample['row_ids'][shard::len(gpus)])
            sub.update(examples=len(sub['row_ids']), shard=shard, shards=len(gpus))
            path = dest / f'sample.part{shard}.json'
            save(path, sub)
            samples.append(sub)
            command = [PYTHON, '-u', '-m', 'latent.eval_unified_small', '--model', str(root / 'sft/final_checkpoint'),
                       '--kind', 'latent', '--data', str(DATA / 'valid' / f'{STEM}.csv'), '--index', str(INDEX),
                       '--sample', str(path), '--output', str(dest / f'target_latent_sft.part{shard}.json'), '--max-batch', '32']
            workers.append(start(command, dest / f'eval.part{shard}.log', gpu))
        status('ranking_eval', scope=scope, worker_pids=[w[0].pid for w in workers], methods=METHODS)
        codes = [finish(w) for w in workers]
        if any(codes):
            raise RuntimeError('Target-interest ranking evaluation failed')
        parts = [json.loads((dest / f'target_latent_sft.part{i}.json').read_text()) for i in range(len(gpus))]
        merged = merge_shards(parts, samples, sample)
        merged['metadata']['sample_sha256'] = sha(dest / 'sample.json')
        save(dest / 'target_latent_sft.json', merged)
        results['target_latent_sft'] = merged
        write_report(dest, sample, results)
    status('complete', comparison=str(root / 'eval/valid20pct/comparison.md'))


def write_report(dest, sample, results):
    metrics = [f'{m}@{k}' for m in ('Recall', 'NDCG') for k in (5, 10, 20)]
    summaries, targets = {}, None
    for name, result in results.items():
        t = validate_result(result, sample)
        if targets is not None and t != targets:
            raise ValueError('Comparison uses different targets')
        targets = t
        summaries[name] = {}
        for method, rows in result['rows'].items():
            total = summarize(rows)
            summaries[name][method] = {k: total[k] for k in metrics}
            summaries[name][method]['fraction_with20'] = sum(len(r['predict']) >= 20 for r in rows) / len(rows)
    lines = [f'# 目标条件兴趣标签 SFT 对照（{len(sample["row_ids"])} 条验证样本）', '',
             '旧商品SID、mean pooling/无L2、三槽非balanced VQ；目标标签使用历史＋真实下一商品，推理只用历史生成兴趣。',
             'SFT沿用原基线15轮上限、全量验证每半轮、最佳loss选择/提前停止。两卡保持有效batch1024。',
             '基线排名来自相同样本、数据、SID、评估源码与协议的历史结果。历史吞吐不作为同场速度对照。', '']
    for method, title in zip(METHODS, ('1×20', '4×20跨兴趣完整求和')):
        lines += [f'## {title}', '', '| 模型 | R@5 | R@10 | R@20 | N@5 | N@10 | N@20 | 满20比例 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for name, title in [('no_latent_sft', '无latent'), ('history_only_latent_sft', '仅历史兴趣'), ('target_latent_sft', '历史＋目标兴趣')]:
            row = summaries[name]['direct50' if name == 'no_latent_sft' else method]
            lines.append('| ' + title + ' | ' + ' | '.join(f'{row[k]:.5f}' for k in metrics) + f" | {row['fraction_with20']:.1%} |")
        lines.append('')
    (dest / 'comparison.md').write_text('\n'.join(lines) + '\n')
    save(dest / 'comparison.json', dict(examples=len(sample['row_ids']), results=summaries,
         verified_same_ids_and_targets=True, controls_reused=True, control_timing='historical'))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        import sys
        if '--root' in sys.argv:
            root = Path(sys.argv[sys.argv.index('--root') + 1]).resolve()
            root.mkdir(parents=True, exist_ok=True)
            save(root / 'pipeline_status.json', dict(phase='failed', updated_at=time.time(), error=traceback.format_exc()))
        raise
