#!/usr/bin/env python3
"""Explicit history or history+target recipe -> nonbalanced VQ; STOP after labels."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from run_two_slot_vq_sft import gpu_snapshot, save, PROJECT, PYTHON, DATA, STEM, INDEX


def validate_embedding_recipe(recipe, embedding_input):
    if embedding_input == 'history_target':
        if not recipe.get('contains_target') or recipe.get('intended_use') != 'latent_supervision_only':
            raise ValueError('Wrong embedding provenance for target-conditioned teacher labels')
    elif embedding_input == 'history':
        if (recipe.get('contains_target') or recipe.get('embedding_input', 'history') != 'history'
                or recipe.get('text') != 'chronological_numbered_titles_v1'):
            raise ValueError('History-only preparation must never reuse target-conditioned embeddings')
    else:
        raise ValueError('Unknown embedding input recipe')


def gpu_ready(info, allow_shared=False):
    if allow_shared:
        # Explicit user authorization is required for sharing an occupied card.
        return info['total_mib'] - info['used_mib'] >= 24 * 1024 and info['utilization'] < 10
    return info['used_mib'] < 1024 and info['utilization'] < 10 and not info['pids']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--gpus',default='4,7')
    p.add_argument('--batch',type=int,default=32)
    p.add_argument('--embedding-input',choices=('history','history_target'),default='history_target')
    p.add_argument('--allow-shared-gpu',action='store_true',help='User-authorized single-card sharing; require >=24GiB free and <10%% util before launch')
    p.add_argument('--data',default=str(DATA))
    p.add_argument('--pooling',choices=('masked_mean','last_token'),default='masked_mean')
    p.add_argument('--l2-normalize',action='store_true')
    p.add_argument('--vq-input-scaling',choices=('auto','none','sqrt_dim'),default='auto')
    p.add_argument('--reuse-embeddings',help='Read completed embeddings from this directory; no encoding/merge')
    args=p.parse_args();os.chdir(PROJECT)
    gpus=args.gpus.split(',')
    if len(gpus) not in (1,2) or len(set(gpus))!=len(gpus) or args.batch<1:p.error('One or two distinct GPUs and a positive embedding batch required')
    if args.allow_shared_gpu and len(gpus) != 1:p.error('Explicit sharing is restricted to one selected GPU')
    data=Path(args.data).resolve()
    index=data/'Industrial_and_Scientific/Industrial_and_Scientific.index.json'
    root=Path(args.root).resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (root/'pipeline.pid').write_text(str(os.getpid())+'\n')
    config=dict(embedding_input=args.embedding_input,target_source='CSV.item_title' if args.embedding_input=='history_target' else None,data=str(data),
        embedding_model=str(PROJECT/'models/Qwen3-Embedding-4B'),gpus=gpus,embedding_batch=args.batch,
        dim=2560,max_length=2048,pooling=args.pooling,normalization='l2' if args.l2_normalize else 'none',num_slots=3,
        codebook_size=256,slot_dim=32,balance_epsilon=0,fit_split='train',selection_split='valid',
        epochs=50,patience=5,vq_batch=512,vq_lr=.001,seed=42,stop_after='vq_and_label_export',
        sft_authorized=False,ranking_eval_authorized=False)
    if args.embedding_input=='history':config['contains_target']=False
    if args.allow_shared_gpu:config['allow_shared_gpu']=True
    scaling='sqrt_dim' if args.vq_input_scaling=='auto' and args.l2_normalize else args.vq_input_scaling
    if scaling=='auto':scaling='none'
    if scaling=='sqrt_dim' and not args.l2_normalize:p.error('sqrt_dim requires --l2-normalize')
    if scaling!='none':config['vq_input_scaling']=scaling
    external=Path(args.reuse_embeddings).resolve() if args.reuse_embeddings else None
    if external:
        if not (external/'embeddings.json').is_file():p.error('Reuse requires a completed embedding manifest')
        validate_embedding_recipe(json.loads((external/'embeddings.json').read_text())['recipe'], args.embedding_input)
        config['reused_embeddings']=str(external)
        config['reused_embedding_manifest_sha256']=hashlib.sha256((external/'embeddings.json').read_bytes()).hexdigest()
    if (root/'pipeline_config.json').exists() and json.loads((root/'pipeline_config.json').read_text())!=config:
        raise ValueError('Refusing to reuse output with changed data/embedding configuration')
    save(root/'pipeline_config.json',config)
    snapshot=root/'source_snapshot'
    if not snapshot.exists():
        snapshot.mkdir();shutil.copytree(PROJECT/'latent',snapshot/'latent',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(PROJECT/'rq',snapshot/'rq',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(__file__,snapshot/Path(__file__).name)
        save(snapshot/'hashes.json',{str(f.relative_to(snapshot)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in snapshot.rglob('*.py')})
    env=dict(os.environ,PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',
        PYTHONPATH=str(snapshot)+':'+str(PROJECT),PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    env['PATH']=str(Path(PYTHON).parent)+':'+env.get('PATH','')
    def status(phase,**extra):
        obj=dict(phase=phase,updated_at=time.time(),coordinator_pid=os.getpid(),gpus=gpus,
            stop_after='vq_and_label_export',**extra)
        save(root/'pipeline_status.json',obj);print(json.dumps(obj,ensure_ascii=False),flush=True)
    def wait_idle(selected):
        good=0
        while good<3:
            snap=gpu_snapshot(selected)
            idle=all(gpu_ready(v,args.allow_shared_gpu) for v in snap.values())
            good=good+1 if idle else 0;status('waiting_for_idle_gpus',selected_gpus=selected,measurements=snap,consecutive_idle_checks=good)
            if good<3:time.sleep(10 if idle else 30)
    def start(cmd,log,gpu=None):
        log.parent.mkdir(parents=True,exist_ok=True);save(log.with_suffix('.command.json'),cmd)
        e=dict(env,CUDA_VISIBLE_DEVICES='' if gpu is None else gpu)
        stream=log.open('w');proc=subprocess.Popen(cmd,env=e,cwd=snapshot,stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        log.with_suffix('.pid').write_text(str(proc.pid)+'\n');return proc,stream,log
    def finish(worker):
        proc,stream,log=worker;code=proc.wait();stream.close();log.with_suffix('.exit_code').write_text(str(code)+'\n');return code
    emb=external if external else root/'embeddings'
    if not external:emb.mkdir(exist_ok=True)
    active={};batches={};done=set(range(len(gpus))) if external else set()
    for shard,gpu in enumerate(gpus):
        if (emb/f'shard{shard}.json').exists():done.add(shard)
    def encode_command(shard,batch):
        cmd=[PYTHON,'-u','-m','latent.prepare','encode','--model',str(PROJECT/'models/Qwen3-Embedding-4B'),
            '--index',str(index),'--embedding_input',args.embedding_input,'--dim','2560','--max_length','2048',
            '--pooling',args.pooling,'--shards',str(len(gpus)),'--shard',str(shard),'--batch',str(batch),'--output',str(emb)]
        if args.l2_normalize:cmd.append('--l2_normalize')
        for split in ('train','valid','test'):cmd.extend(['--'+split,str(data/split/f'{STEM}.csv')])
        return cmd
    pending=[s for s in range(len(gpus)) if s not in done]
    for shard in pending:batches[shard]=args.batch
    assignments={};idle_checks={gpu:0 for gpu in gpus};last_report=0
    try:
        while pending or active:
            for shard,worker in list(active.items()):
                if worker[0].poll() is None:continue
                code=finish(worker);del active[shard]
                gpu=assignments.pop(shard);idle_checks[gpu]=0
                if code==0:
                    if not (emb/f'shard{shard}.json').exists():raise RuntimeError('Encoder exited without shard manifest')
                    done.add(shard)
                elif 'out of memory' in worker[2].read_text(errors='replace').lower() and batches[shard]>1:
                    batches[shard]//=2;pending.insert(0,shard)
                else:
                    raise RuntimeError(f'Embedding shard{shard} failed exit{code}; inspect {worker[2]}')
            measurements={}
            if pending:
                available=[gpu for gpu in gpus if gpu not in assignments.values()]
                if available:
                    measurements=gpu_snapshot(available)
                    for gpu in available:
                        v=measurements[gpu]
                        idle=gpu_ready(v,args.allow_shared_gpu)
                        idle_checks[gpu]=idle_checks[gpu]+1 if idle else 0
                        if idle_checks[gpu]>=3 and pending:
                            shard=pending.pop(0);assignments[shard]=gpu;idle_checks[gpu]=0
                            active[shard]=start(encode_command(shard,batches[shard]),root/f'encode{shard}_batch{batches[shard]}.log',gpu)
                            status('encoding_'+args.embedding_input,worker_pids={s:w[0].pid for s,w in active.items()},worker_gpus=assignments,
                                batch_per_shard=batches,completed_shards=sorted(done),pending_shards=pending)
            if time.monotonic()-last_report>60:
                status('encoding_'+args.embedding_input if active else 'waiting_for_idle_gpus',
                    worker_pids={s:w[0].pid for s,w in active.items()},worker_gpus=assignments,
                    batch_per_shard=batches,completed_shards=sorted(done),pending_shards=pending,measurements=measurements)
                last_report=time.monotonic()
            if pending or active:time.sleep(10)
    finally:
        # Failed preparation must not leave another owned encoder orphaned.
        for worker in active.values():
            proc,stream,log=worker
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            stream.close()
    status('merging_embeddings')
    if not (emb/'embeddings.json').exists():
        worker=start([PYTHON,'-u','-m','latent.prepare','merge','--output',str(emb),'--shards',str(len(gpus))],root/'merge.log')
        if finish(worker):raise RuntimeError('Embedding merge failed; inspect merge.log')
    manifest=json.loads((emb/'embeddings.json').read_text())
    validate_embedding_recipe(manifest['recipe'],args.embedding_input)
    from latent.core import index_hash,load_index,sha,load_labels
    if manifest['sid_sha256']!=index_hash(load_index(index)) or any(
        manifest['source_hashes'][split]!=sha(data/split/f'{STEM}.csv') for split in ('train','valid','test')):
        raise ValueError('Embeddings are bound to a different SID/data version')
    expected_pool='last_valid_token' if args.pooling=='last_token' else 'masked_mean'
    if manifest['recipe']['pooling']!=expected_pool or manifest['recipe']['normalization']!=('l2' if args.l2_normalize else 'none'):
        raise ValueError('Wrong interest embedding pooling/normalization')
    if args.l2_normalize:
        import numpy as np
        norm_stats={}
        for split in ('train','valid','test'):
            vectors=np.load(emb/f'{split}.npy',mmap_mode='r',allow_pickle=False)
            low,high=float('inf'),0.
            for chunk_start in range(0,len(vectors),4096):
                norms=np.linalg.norm(vectors[chunk_start:chunk_start+4096],axis=1)
                if not np.allclose(norms,1,atol=1e-5,rtol=0):raise ValueError('Interest embedding is not L2-normalized')
                low=min(low,float(norms.min()));high=max(high,float(norms.max()))
            norm_stats[split]=dict(min_norm=low,max_norm=high,shape=list(vectors.shape))
        save(root/'embedding_norms.json',norm_stats)
    labels=root/'labels'
    if not (labels/'labels.json').exists():
        gpu=gpus[-1];wait_idle([gpu]);status('training_vq',gpu=gpu,num_slots=3,balance_epsilon=0,embedding_counts=manifest['counts'])
        worker=start([PYTHON,'-u','-m','latent.prepare','vq','--embeddings',str(emb),'--output',str(labels),
            '--num_slots','3','--codebook_size','256','--latent_dim','32','--balance_epsilon','0',
            '--epochs','50','--patience','5','--batch','512','--lr','.001','--seed','42',
            '--input_scaling',scaling],root/'vq.log',gpu)
        if finish(worker):raise RuntimeError('VQ fitting failed; inspect vq.log')
    result=json.loads((labels/'labels.json').read_text())
    if result['fit_split']!='train' or result['k']!=3:
        raise ValueError('Wrong VQ labels/provenance')
    validate_embedding_recipe(result['embedding_recipe'],args.embedding_input)
    import math
    expected_scale=math.sqrt(manifest['recipe']['dim']) if scaling=='sqrt_dim' else 1.
    if result['vq_options'].get('input_scale',1.)!=expected_scale:
        raise ValueError('VQ checkpoint uses the wrong input scale')
    for split in ('train','valid','test'):
        load_labels(labels,split,data/split/f'{STEM}.csv',load_index(index))
    status('complete_vq_only',labels=str(labels),best_epoch=result['best_epoch'],
        validation_reconstruction=result['validation_reconstruction'],code_statistics=result['code_statistics'],
        vq_input_transform=result.get('vq_input_transform'),collapsed_all_slots=result.get('collapsed_all_slots'),
        next_action='Wait for user to request SFT and ranking evaluation; neither launched')


if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc()
        import sys
        if '--root' in sys.argv:
            root=Path(sys.argv[sys.argv.index('--root')+1]).resolve();root.mkdir(parents=True,exist_ok=True)
            save(root/'pipeline_error.json',dict(time=time.time(),error=traceback.format_exc()))
            save(root/'pipeline_status.json',dict(phase='failed',updated_at=time.time(),error=traceback.format_exc(),stop_after='vq_and_label_export'))
        raise
