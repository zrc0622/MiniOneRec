"""New item SID + official history-only VQ interests -> causal serial SFT and eval."""
import argparse
import copy
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import time
import traceback

from run_two_slot_vq_sft import PROJECT, PYTHON, STEM, gpu_snapshot, save
from latent.core import load_index, load_labels, sha

DATA = PROJECT / 'outputs/amazon23_industrial_qwen3_4b_lasttoken_l2_sid_20261010/dataset'
INDEX = DATA / 'Industrial_and_Scientific/Industrial_and_Scientific.index.json'
ITEMS = DATA / 'Industrial_and_Scientific/Industrial_and_Scientific.item.json'
LABELS = PROJECT / 'outputs/latent_qwen3_0.6b_history_lasttoken_l2_sqrtd_vq3_20261012/labels'
NATIVE = PROJECT / 'outputs/amazon23_industrial_qwen3_0.6b_sft_lasttoken_l2_balanced15_20261011/sft/final_checkpoint'
NATIVE_CACHE = NATIVE.parents[1] / 'rank_eval_20261012/new_sid.json'
OLD_DATA = PROJECT / 'data/Amazon23/variants/rqkmeans_balanced'
OLD_INDEX = OLD_DATA / 'Industrial_and_Scientific/Industrial_and_Scientific.index.json'
OLD_NATIVE = PROJECT / 'outputs/amazon23_industrial_qwen3_0.6b_sft_balanced_15ep/final_checkpoint'
OLD_LATENT = PROJECT / 'outputs/latent_qwen3_0.6b_balanced15/sft/final_checkpoint'
CONTROLS = PROJECT / 'outputs/latent_qwen3_0.6b_vq2_15ep_20261010/eval'
METHODS = ['single20', 'four20_marginal']
OWNED_WORKERS = []


def check_recipe(manifest):
    r = manifest['embedding_recipe']
    if (manifest['k'] != 3 or manifest['codebook_size'] != 256
            or manifest['vq_options'].get('balance_epsilon', 0)
            or r.get('embedding_input') != 'history' or r.get('contains_target') is not False
            or r['pooling'] != 'last_valid_token' or r['normalization'] != 'l2'
            or manifest['vq_input_transform']['kind'] != 'sqrt_dim'
            or manifest['vq_options'].get('input_scale') != math.sqrt(r['dim'])
            or manifest.get('collapsed_all_slots')):
        raise ValueError('Require new scaled, nonbalanced, history-only three-slot interest labels')


def training_command(root, snapshot, port, batch=32, latent_weight=1.0):
    if batch not in (16, 32):
        raise ValueError('Supported two-card batches are 16 and 32')
    if not math.isfinite(latent_weight) or latent_weight <= 0:
        raise ValueError('Interest weight must be finite and positive')
    return [PYTHON, '-m', 'torch.distributed.run', '--nnodes=1', '--nproc_per_node=2',
        '--rdzv_backend=static', '--master_addr=127.0.0.1', f'--master_port={port}',
        '-m', 'latent.train', 'sft', '--model', str(PROJECT / 'models/Qwen3-0.6B'),
        '--labels', str(LABELS), '--train_file', str(DATA / 'train' / f'{STEM}.csv'),
        '--eval_file', str(DATA / 'valid' / f'{STEM}.csv'), '--items', str(ITEMS), '--index', str(INDEX),
        '--output_dir', str(root / 'sft'), '--batch', str(batch),
        '--gradient_accumulation', str(1024 // (2 * batch)), '--eval_batch', '16',
        '--epochs', '15', '--lr', '.0003', '--seed', '42', '--latent_weight', str(latent_weight),
        '--sft_eval_fraction', '1', '--sft_eval_every_epochs', '.5', '--sft_save_every_epochs', '.5',
        '--sft_early_stopping_patience', '3', '--deepspeed', str(snapshot / 'config/sft_zero2.json')]


def resume_command(root, checkpoint, port):
    from run_sft_recovery import checkpoint_state
    checkpoint_state(root, checkpoint)
    command = json.loads((root / 'sft.command.json').read_text())
    if 'latent.train' not in command or command[command.index('latent.train') + 1] != 'sft':
        raise ValueError('Not the original serial SFT command')
    if '--resume_from_checkpoint' in command:
        raise ValueError('Original command already contains a resume option')
    for rank in (0, 1):
        if not (checkpoint / f'rng_state_{rank}.pth').is_file():
            raise ValueError('Both original RNG states are required')
    command = [f'--master_port={port}' if s.startswith('--master_port=') else s for s in command]
    command += ['--resume_from_checkpoint', str(checkpoint)]
    from latent.train import parser, validate_sft_resume
    options = parser().parse_args(command[command.index('latent.train') + 1:])
    validate_sft_resume(options, root / 'sft')
    snapshot = root / 'source_snapshot'
    for name, digest in json.loads((snapshot / 'hashes.json').read_text()).items():
        if sha(snapshot / name) != digest:
            raise ValueError(f'Original source snapshot changed: {name}')
    return command


def canonical_result(result, index, source, ids):
    inverse = {''.join(tokens): item for item, tokens in index.items()}
    if len(inverse) != len(index):
        raise ValueError('SID must uniquely identify real items')
    converted = copy.deepcopy(result)
    expected_methods = METHODS if result['metadata']['kind'] == 'latent' else ['direct50']
    if set(result['rows']) != set(expected_methods):
        raise ValueError('Missing evaluation method')
    for method, rows in converted['rows'].items():
        if [r['row_id'] for r in rows] != ids:
            raise ValueError('Missing, reordered or different evaluation rows')
        for row in rows:
            target = inverse[row['output']]
            if target != source[row['row_id']]['item_id']:
                raise ValueError('Decoded target differs from the real target item')
            row['output'] = target
            row['predict'] = [inverse[p] for p in row['predict']]
            if len(row['predict']) != len(set(row['predict'])):
                raise ValueError('Duplicate real item predictions')
    # Keep original metadata truthful: conversion never claims a different dataset/model was evaluated.
    converted['canonical_item_ids'] = True
    return converted


def validate_cache(result, model, kind, sample_file, data_file, index_file, snapshot):
    expected = dict(model=str(model), kind=kind, sample_sha256=sha(sample_file),
        data_sha256=sha(data_file), index_sha256=sha(index_file),
        source_sha256=sha(snapshot / 'latent/eval_unified_small.py'),
        protocol=json.loads((snapshot / 'eval_metric_protocol.json').read_text()))
    if any(result['metadata'].get(k) != v for k, v in expected.items()):
        raise ValueError('Cached control model/data/sample/source/protocol does not match')


def report(dest, sample, results):
    metrics = [f'{m}@{k}' for m in ('Recall', 'NDCG') for k in (5, 10, 20)]
    summaries = {}
    targets = None
    for name, result in results.items():
        summaries[name] = {}
        for method, rows in result['rows'].items():
            if [r['row_id'] for r in rows] != sample['row_ids']:
                raise ValueError('Different comparison rows')
            current = [r['output'] for r in rows]
            if targets is not None and targets != current:
                raise ValueError('Different real target items')
            targets = current
            ranks = [r['predict'].index(r['output']) + 1 if r['output'] in r['predict'] else None for r in rows]
            summaries[name][method] = {
                **{f'Recall@{k}': sum(r is not None and r <= k for r in ranks) / len(rows) for k in (5, 10, 20)},
                **{f'NDCG@{k}': sum(1 / math.log2(r + 1) for r in ranks if r is not None and r <= k) / len(rows)
                   for k in (5, 10, 20)},
                'fraction_with20': sum(len(set(r['predict'])) >= 20 for r in rows) / len(rows)}
    save(dest / 'comparison.json', dict(examples=len(sample['row_ids']), results=summaries,
         canonical_item_ids=True, verified_same_ids_and_targets=True,
         note='Historical control timing is not a matched inference speed comparison'))
    lines = [f'# 新商品SID＋新仅历史兴趣：串行SFT对照（{len(sample["row_ids"])}条验证样本）', '',
             '新兴趣使用last-valid-token/L2与尺度修复的非平衡三槽VQ；仅历史，没有目标商品。标准因果串行预测。',
             '相同样本映射回真实商品ID比较；新版无latent为主要参照，旧SID模型提供历史背景。', '']
    names = [('new_sid_no_latent', '新SID，无latent'), ('new_history_latent', '新SID＋新历史兴趣'),
             ('old_sid_no_latent', '旧SID，无latent'), ('old_history_latent', '旧SID＋原历史兴趣')]
    if 'reference_history_latent' in summaries:
        names.insert(2, ('reference_history_latent', '新SID＋新历史兴趣，权重1对照'))
    for method, title in zip(METHODS, ['1×20', '4×20跨兴趣完整求和']):
        lines += [f'## {title}', '', '| 模型 | R@5 | R@10 | R@20 | N@5 | N@10 | N@20 | 满20比例 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for name, label in names:
            row = summaries[name]['direct50' if 'no_latent' in name else method]
            lines.append('| ' + label + ' | ' + ' | '.join(f'{row[k]:.5f}' for k in metrics)
                         + f' | {row["fraction_with20"]:.1%} |')
        lines.append('')
    (dest / 'comparison.md').write_text('\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--gpus', default='2,5')
    p.add_argument('--resume-checkpoint', help='Recover this experiment with its original source and options')
    p.add_argument('--latent-weight', type=float, default=1.0)
    p.add_argument('--reference-run', help='Same-data history-interest SFT to compare after training')
    o = p.parse_args()
    if not math.isfinite(o.latent_weight) or o.latent_weight <= 0:
        raise ValueError('Interest weight must be finite and positive')
    gpus = o.gpus.split(',')
    if len(gpus) != 2 or len(set(gpus)) != 2 or not all(g.isdigit() for g in gpus):
        raise ValueError('Require two distinct GPU indices')
    root = Path(o.root).resolve()
    if not o.resume_checkpoint:
        root.mkdir(exist_ok=False)
    lock = (root / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    recovery = None
    if o.resume_checkpoint:
        checkpoint = Path(o.resume_checkpoint).resolve()
        resume_command(root, checkpoint, 12345)
        recovery = root / ('recovery_' + time.strftime('%Y%m%d_%H%M%S'))
        recovery.mkdir()
        for name in ('pipeline_status.json', 'pipeline_config.json', 'sft.command.json'):
            shutil.copy2(root / name, recovery / name)
        shutil.copy2(root / 'sft/latent_run.json', recovery / 'latent_run.original.json')

    def status(phase, **extra):
        save(root / 'pipeline_status.json', dict(phase=phase, updated_at=time.time(),
             coordinator_pid=os.getpid(), gpus=gpus, **extra))

    status('verifying_new_history_labels')
    index = load_index(INDEX)
    shapes = {}
    for split in ('train', 'valid', 'test'):
        codes, manifest = load_labels(LABELS, split, DATA / split / f'{STEM}.csv', index)
        shapes[split] = list(codes.shape)
    check_recipe(manifest)
    save(root / 'label_binding_verified.json', dict(shapes=shapes, sid_sha256=manifest['sid_sha256'],
         contains_target=False, labels_sha256=sha(LABELS / 'labels.json')))
    config = dict(gpus=gpus, data=str(DATA), labels=str(LABELS), labels_sha256=sha(LABELS / 'labels.json'),
         prediction='autoregressive', attention='original causal Qwen SDPA', num_slots=3, codebook_size=256,
         global_batch=1024, initial_batch=32, accumulation=16, epochs=15, lr=.0003, seed=42,
         latent_weight=o.latent_weight, validation_fraction=1, validation_every_epochs=.5, save_every_epochs=.5,
         checkpoint_selection='best full validation loss, early stopping patience3',
         contains_target=False, methods=METHODS, eval_samples=[256, 3232], rl_authorized=False,
         primary_control=str(NATIVE), periodic_disk_check=False, launcher='single-node static rendezvous')
    reference = Path(o.reference_run).resolve() if o.reference_run else None
    if reference:
        reference_config = json.loads((reference / 'pipeline_config.json').read_text())
        compared = ('data', 'labels', 'labels_sha256', 'prediction', 'attention', 'num_slots',
                    'codebook_size', 'global_batch', 'initial_batch', 'accumulation', 'epochs',
                    'lr', 'seed', 'validation_fraction', 'validation_every_epochs', 'save_every_epochs',
                    'checkpoint_selection', 'contains_target', 'methods', 'eval_samples')
        if any(reference_config[k] != config[k] for k in compared):
            raise ValueError('Tuning reference differs beyond interest weight and GPU placement')
        config['tuning'] = dict(reference_run=str(reference), changed_parameter='latent_weight',
             reference_value=reference_config['latent_weight'], current_value=o.latent_weight,
             comparison='ranking and item CE; combined weighted loss not comparable across weights')
    snapshot = root / 'source_snapshot'
    if recovery:
        config = json.loads((root / 'pipeline_config.json').read_text())
        save(recovery / 'recovery_config.json', dict(gpus=gpus, checkpoint=str(checkpoint),
             original_config=config, original_source_preserved=True, periodic_disk_check=False))
    else:
        save(root / 'pipeline_config.json', config)
        snapshot.mkdir()
        for folder in ('latent', 'interest', 'config'):
            shutil.copytree(PROJECT / folder, snapshot / folder, ignore=shutil.ignore_patterns('__pycache__'))
        for name in ('data.py', 'deepspeed_cleanup.py', 'eval_metric_protocol.json', Path(__file__).name):
            shutil.copy2(PROJECT / name, snapshot / name)
        save(snapshot / 'hashes.json', {str(f.relative_to(snapshot)): sha(f) for f in snapshot.rglob('*') if f.is_file()})
    temp = root / 'tmp'
    temp.mkdir(exist_ok=bool(recovery))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=o.gpus, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false', NCCL_IB_DISABLE='1', TMPDIR=str(temp),
        PYTHONPATH=str(snapshot) + ':' + str(PROJECT), PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    env['PATH'] = str(Path(PYTHON).parent) + ':' + env.get('PATH', '')
    env['CUDA_HOME'] = str(Path(PYTHON).parent.parent)
    (root / 'pipeline.pid').write_text(str(os.getpid()) + '\n')
    # One startup capacity check, as requested; no recurring disk polling.
    if shutil.disk_usage(root).free < 100 * 2**30:
        raise RuntimeError('Insufficient storage before training')

    def wait_idle():
        good = 0
        while good < 3:
            values = gpu_snapshot(gpus)
            good = good + 1 if all(v['used_mib'] < 1024 and v['utilization'] < 10 and not v['pids']
                                 for v in values.values()) else 0
            status('waiting_for_idle_gpus', measurements=values, consecutive_idle_checks=good)
            if good < 3:
                time.sleep(5 if good else 30)

    def start(command, log, gpu=None):
        save(log.with_suffix('.command.json'), command)
        stream = log.open('w')
        worker = subprocess.Popen(command, cwd=snapshot, env=dict(env, CUDA_VISIBLE_DEVICES=gpu) if gpu else env,
                                  stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                  start_new_session=True)
        OWNED_WORKERS.append(worker)
        log.with_suffix('.pid').write_text(str(worker.pid) + '\n')
        return worker, stream, log

    def finish(job):
        worker, stream, log = job
        code = worker.wait()
        OWNED_WORKERS.remove(worker)
        stream.close()
        log.with_suffix('.exit_code').write_text(str(code) + '\n')
        return code

    wait_idle()
    if sha(LABELS / 'labels.json') != config['labels_sha256']:
        raise ValueError('Interest labels changed while waiting for GPUs')
    for batch in ((config['initial_batch'],) if recovery else (32, 16)):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        command = resume_command(root, checkpoint, port) if recovery else training_command(
            root, snapshot, port, batch, config['latent_weight'])
        log_path = (recovery or root) / 'sft.log'
        job = start(command, log_path)
        status('training_serial_sft', worker_pid=job[0].pid, batch_per_device=batch,
               accumulation=1024 // (2 * batch), global_batch=1024,
               resume_from_checkpoint=str(checkpoint) if recovery else None, log=str(log_path))
        code = finish(job)
        if not code:
            break
        if recovery:
            raise RuntimeError('Resumed SFT failed; preserve checkpoint and inspect recovery log')
        log = (root / 'sft.log').read_text(errors='replace')
        if batch != 32 or 'out of memory' not in log.lower() or re.search(r'[1-9]\d*/4140', log):
            raise RuntimeError('Serial SFT failed; preserve state and inspect sft.log')
        archive = root / 'startup_oom_batch32'
        archive.mkdir()
        if (root / 'sft').exists():
            (root / 'sft').rename(archive / 'sft')
        for f in root.glob('sft.*'):
            if f.is_file():
                f.rename(archive / f.name)
        save(root / 'batch_fallback.json', dict(batch=16, accumulation=32, global_batch=1024,
             reason='OOM before first completed optimizer update'))
        wait_idle()
    final = root / 'sft/final_checkpoint'
    if not all((final / f).is_file() for f in ('model.safetensors', 'config.json', 'tokenizer_config.json')):
        raise ValueError('Final model export incomplete')
    save(root / 'sft_complete.json', dict(checkpoint=str(final), completed_at=time.time(), exit_code=0))
    if reference:
        # The reference usually finishes first; never silently omit it if it is still evaluating.
        while True:
            reference_status = json.loads((reference / 'pipeline_status.json').read_text())
            if reference_status['phase'] == 'complete':
                break
            if reference_status['phase'].startswith(('failed', 'interrupted')):
                raise RuntimeError('Reference run did not complete; retain tuned model for later comparison')
            status('waiting_for_reference_eval', reference_run=str(reference))
            time.sleep(30)
    with (DATA / 'valid' / f'{STEM}.csv').open() as stream:
        source = list(csv.DictReader(stream))
    with (OLD_DATA / 'valid' / f'{STEM}.csv').open() as stream:
        old_source = list(csv.DictReader(stream))
    from run_new_sid_ranking_eval import validate_alignment
    validate_alignment(old_source, source)
    old_index = load_index(OLD_INDEX)
    for scope in ('sample256', 'valid20pct'):
        dest = root / 'eval' / scope
        dest.mkdir(parents=True)
        sample_file = dest / 'sample.json'
        shutil.copy2(CONTROLS / scope / 'sample.json', sample_file)
        sample = json.loads(sample_file.read_text())
        results = {}
        for name, filename, model, kind in [('old_history_latent', 'three_slot_vq_sft.json', OLD_LATENT, 'latent'),
                                           ('old_sid_no_latent', 'no_latent_sft.json', OLD_NATIVE, 'sid')]:
            cached = json.loads((CONTROLS / scope / filename).read_text())
            validate_cache(cached, model, kind, sample_file, OLD_DATA / 'valid' / f'{STEM}.csv', OLD_INDEX, snapshot)
            results[name] = canonical_result(cached, old_index, source, sample['row_ids'])
            save(dest / (name + '.json'), results[name])
        wait_idle()

        def eval_command(model, kind, part_sample, output):
            return [PYTHON, '-u', '-m', 'latent.eval_unified_small', '--model', str(model), '--kind', kind,
                    '--data', str(DATA / 'valid' / f'{STEM}.csv'), '--index', str(INDEX),
                    '--sample', str(part_sample), '--output', str(output), '--max-batch', '32']

        if scope == 'sample256':
            jobs = [start(eval_command(final, 'latent', sample_file, dest / 'new_history_latent.json'),
                          dest / 'latent_eval.log', gpus[0]),
                    start(eval_command(NATIVE, 'sid', sample_file, dest / 'new_sid_no_latent.json'),
                          dest / 'native_eval.log', gpus[1])]
            status('ranking_eval', scope=scope, worker_pids=[w[0].pid for w in jobs])
            codes = [finish(w) for w in jobs]
            if any(codes):
                raise RuntimeError('Small-sample evaluation failed')
            fresh = json.loads((dest / 'new_history_latent.json').read_text())
            native = json.loads((dest / 'new_sid_no_latent.json').read_text())
        else:
            native = json.loads(NATIVE_CACHE.read_text())
            validate_cache(native, NATIVE, 'sid', sample_file, DATA / 'valid' / f'{STEM}.csv', INDEX, snapshot)
            save(dest / 'new_sid_no_latent.json', native)
            jobs, parts, samples = [], [], []
            for shard, gpu in enumerate(gpus):
                sub = dict(sample, row_ids=sample['row_ids'][shard::2], shard=shard, shards=2)
                samples.append(sub)
                part_sample = dest / f'sample.part{shard}.json'
                save(part_sample, sub)
                output = dest / f'new_history_latent.part{shard}.json'
                parts.append(output)
                jobs.append(start(eval_command(final, 'latent', part_sample, output), dest / f'eval.part{shard}.log', gpu))
            status('ranking_eval', scope=scope, worker_pids=[w[0].pid for w in jobs])
            codes = [finish(w) for w in jobs]
            if any(codes):
                raise RuntimeError('20percent validation evaluation failed')
            from run_target_interest_sft import merge_shards
            fresh = merge_shards([json.loads(f.read_text()) for f in parts], samples, sample)
            fresh['metadata']['sample_sha256'] = sha(sample_file)
            save(dest / 'new_history_latent.json', fresh)
        validate_cache(fresh, final, 'latent', sample_file, DATA / 'valid' / f'{STEM}.csv', INDEX, snapshot)
        validate_cache(native, NATIVE, 'sid', sample_file, DATA / 'valid' / f'{STEM}.csv', INDEX, snapshot)
        results['new_history_latent'] = canonical_result(fresh, index, source, sample['row_ids'])
        results['new_sid_no_latent'] = canonical_result(native, index, source, sample['row_ids'])
        if reference:
            cached = json.loads((reference / 'eval' / scope / 'new_history_latent.json').read_text())
            validate_cache(cached, reference / 'sft/final_checkpoint', 'latent', sample_file,
                           DATA / 'valid' / f'{STEM}.csv', INDEX, snapshot)
            results['reference_history_latent'] = canonical_result(cached, index, source, sample['row_ids'])
        report(dest, sample, results)
        if reference:
            path = dest / 'comparison.md'
            path.write_text(path.read_text() + f'\n本轮兴趣 loss 权重 {config["latent_weight"]}；'
                 f'对照为 {reference_config["latent_weight"]}。比较推荐指标，不跨权重比较总 loss。\n')
    status('complete', checkpoint=str(final), comparison=str(root / 'eval/valid20pct/comparison.md'))


if __name__ == '__main__':
    def stopped(signum, frame):
        raise RuntimeError(f'Coordinator received signal {signum}')
    signal.signal(signal.SIGTERM, stopped)
    try:
        main()
    except Exception:
        traceback.print_exc()
        import sys
        if '--root' in sys.argv:
            root = Path(sys.argv[sys.argv.index('--root') + 1]).resolve()
            if root.exists():
                save(root / 'pipeline_status.json', dict(phase='failed', updated_at=time.time(), error=traceback.format_exc()))
        raise
    finally:
        for worker in OWNED_WORKERS:
            if worker.poll() is None:
                os.killpg(worker.pid, signal.SIGTERM)
                try:
                    worker.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(worker.pid, signal.SIGKILL)
                    worker.wait()
