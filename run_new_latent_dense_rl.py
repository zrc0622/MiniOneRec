"""New item/history SID three-slot latent4x4 dense RL from SFT weight0.2 on two GPUs."""
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
import traceback

from transformers import AutoConfig,AutoTokenizer
from prepare_qwen_official_sid import check_experiment
from run_two_slot_vq_sft import PROJECT,PYTHON,STEM,gpu_snapshot,save
from run_new_history_interest_sft import (DATA,INDEX,ITEMS,NATIVE,OLD_DATA,OLD_INDEX,
    OLD_NATIVE,OLD_LATENT,CONTROLS,validate_cache,canonical_result,report,sha)
from latent.rl_launch import check_gpu_capacity
from run_target_interest_sft import merge_shards,validate_result
from run_new_sid_ranking_eval import validate_alignment
from latent.core import load_index,validate_protocol

SFT_SOURCE=PROJECT/'outputs/latent_qwen3_0.6b_history_lasttoken_l2_newsid_serial15_lw02_20261013'
MODEL=SFT_SOURCE/'sft/final_checkpoint'


def training_command(snapshot,root,batch,smoke=False,port=None,fast=False):
    if batch not in (16,32):raise ValueError('Use16or32 trajectories per card')
    if port is None:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    output=root/(f'smoke_batch{batch}' if smoke else 'rl')
    accumulation=1 if smoke else 1024//(2*batch)
    command=[PYTHON,'-u','-m','torch.distributed.run','--nnodes=1','--nproc_per_node=2',
        '--rdzv_backend=static','--master_addr=127.0.0.1',f'--master_port={port}',
        '-m','latent.train_fast' if fast else 'latent.train','rl','--model',str(MODEL),'--mode','4x4','--reward_kind','sid_dense',
        '--train_file',str(DATA/'train'/f'{STEM}.csv'),'--eval_file',str(DATA/'valid'/f'{STEM}.csv'),
        '--items',str(ITEMS),'--index',str(INDEX),'--batch',str(batch),'--eval_batch','32',
        '--gradient_accumulation',str(accumulation),'--epochs','2','--lr','1e-5','--seed','42',
        '--beta','.001','--temperature','1.0','--latent_weight','1.0','--partial_weight','.2',
        '--eval_steps','1' if smoke else '.25','--save_steps','3' if smoke else '.1',
        '--rl_eval_fraction','.2','--rl_save_total_limit','3',
        '--deepspeed',str(snapshot/'config/sft_zero2.json'),'--output_dir',str(output)]
    if smoke:command+=['--max_steps','2','--eval_limit','32']
    return command,output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--gpus',default='4,5')
    p.add_argument('--fast',action='store_true',help='Use verified latent-only cached4x4 and vectorized scoring')
    p.add_argument('--verified-fast-pilot',default=str(PROJECT/'outputs/latent_rl_speed_pilot_20261014_v2'))
    o=p.parse_args();gpus=o.gpus.split(',')
    if len(gpus)!=2 or len(set(gpus))!=2 or not all(g.isdigit() for g in gpus):
        raise ValueError('Two distinct GPUs required')
    root=Path(o.root).resolve();root.mkdir(parents=True,exist_ok=False)
    lock=(root/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (root/'pipeline.pid').write_text(str(os.getpid())+'\n')
    active=[];streams=[]
    def status(phase,**extra):
        save(root/'pipeline_status.json',dict(phase=phase,updated_at=time.time(),coordinator_pid=os.getpid(),gpus=gpus,**extra))
    def stopped(signum,frame):raise RuntimeError(f'DenseRL coordinator receivedsignal{signum}')
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    try:
        status('verifying_dense_rl_inputs')
        fast_evidence=None
        if o.fast:
            pilot=Path(o.verified_fast_pilot).resolve()
            proof=json.loads((pilot/'verification.json').read_text())
            if proof['status']!='pass' or Path(proof['model']).resolve()!=MODEL.resolve():
                raise ValueError('Fast rollout requires passed checks on the same initial SFT')
            if json.loads((pilot/'pipeline_status.json').read_text())['phase']!='complete':
                raise ValueError('Fast pilot must have completed real optimizer updates')
            expected=json.loads((pilot/'source_hashes.json').read_text())['latent/rl_fast.py']
            if sha(PROJECT/'latent/rl_fast.py')!=expected or sha(pilot/'source_snapshot/latent/rl_fast.py')!=expected:
                raise ValueError('Fast sampler/scoring changed since verification')
            fast_evidence=dict(pilot=str(pilot),sampler_scoring_sha256=expected,
                conditional_distribution_checks=proof['checks'],
                sampling_note='same constrained law; batch/cache/RNG order and BF16 numerics differ; not bitwise identical')
            save(root/'fast_implementation_verified.json',fast_evidence)
        check_experiment(DATA)
        if json.loads((SFT_SOURCE/'pipeline_status.json').read_text())['phase']!='complete':
            raise ValueError('Initial SFT must have completed evaluation')
        source_config=json.loads((SFT_SOURCE/'pipeline_config.json').read_text())
        if source_config['latent_weight']!=.2 or source_config['contains_target']:
            raise ValueError('Require the completed history-only SFT weight0.2 model')
        index=load_index(INDEX)
        protocol=validate_protocol(AutoConfig.from_pretrained(MODEL),AutoTokenizer.from_pretrained(MODEL),index)
        if protocol['k']!=3:raise ValueError('Require3serialinterestslots')
        check_gpu_capacity(gpu_snapshot(gpus),gpus,False,35)
        if shutil.disk_usage(root).free<100*2**30:raise RuntimeError('Less than100GiBfree atstartup')
        input_paths=[DATA/split/f'{STEM}.csv' for split in ('train','valid','test')]+[ITEMS,INDEX]
        save(root/'pipeline_config.json',dict(gpus=gpus,model=str(MODEL),data=str(DATA),
            sft_source=str(SFT_SOURCE),sft_latent_weight=.2,rl_latent_weight=1.,mode='4x4',reward_kind='sid_dense',
            reward_formula='I(full_sid_hit)+0.2*matching_a_b_c_positions/3; exclude_d_newline_eos',
            interest_reward='max_of4_item_rewards_per_interest; across4_interest_advantages',
            item_advantage='within4_items_per_interest',auxiliary_reward='original_binary',
            epochs=2,lr=1e-5,beta=.001,temperature=1.,sampling='independent4interestsX4items; use_model_defaultsFalse',
            num_generations=16,global_candidate_batch=1024,independent_prompts_per_update=64,
            initial_per_device_batch=32,initial_accumulation=16,fallback_per_device_batch=16,fallback_accumulation=32,
            training_eval_fraction=.2,training_eval_every_epochs=.5,save_fraction=.1,save_total_limit=3,
            final_checkpoint_selection='final_rl_epoch',ranking_methods=['single20','four20_marginal'],
            ranking_scopes=['sample256','valid20pct'],attention='original_math_SDPunchanged',periodic_disk_check=False,
            implementation='cached4x4_vectorized' if o.fast else 'original',
            training_entrypoint='latent.train_fast' if o.fast else 'latent.train',fast_evidence=fast_evidence,
            source_hashes={str(path):sha(path) for path in input_paths},sft_model_sha256=sha(MODEL/'model.safetensors')))
        snapshot=root/'source_snapshot';snapshot.mkdir()
        for folder in ('latent','interest','config'):
            shutil.copytree(PROJECT/folder,snapshot/folder,ignore=shutil.ignore_patterns('__pycache__'))
        for source in PROJECT.glob('*.py'):shutil.copy2(source,snapshot/source.name)
        shutil.copy2(PROJECT/'eval_metric_protocol.json',snapshot/'eval_metric_protocol.json')
        save(snapshot/'hashes.json',{str(path.relative_to(snapshot)):sha(path) for path in snapshot.rglob('*') if path.is_file()})
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=o.gpus,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',
            TOKENIZERS_PARALLELISM='false',NCCL_IB_DISABLE='1',PYTHONPATH=str(snapshot),
            PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',CUDA_HOME=str(Path(PYTHON).parent.parent))
        env['PATH']=str(Path(PYTHON).parent)+':'+env.get('PATH','')
        def start(command,log,gpu=None):
            save(log.with_suffix('.command.json'),command)
            stream=log.open('w');streams.append(stream)
            worker=subprocess.Popen(command,cwd=snapshot,env=dict(env,CUDA_VISIBLE_DEVICES=gpu) if gpu else env,
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            active.append(worker);log.with_suffix('.pid').write_text(str(worker.pid)+'\n')
            return worker,stream,log
        def finish(job):
            worker,stream,log=job;code=worker.wait();stream.close();active.remove(worker)
            log.with_suffix('.exit_code').write_text(str(code)+'\n');return code
        chosen=None
        for batch in (32,16):
            check_gpu_capacity(gpu_snapshot(gpus),gpus,False,35)
            command,out=training_command(snapshot,root,batch,smoke=True,fast=o.fast)
            job=start(command,root/f'smoke_batch{batch}.log')
            status('smoke_dense_rl',batch_per_device=batch,worker_pid=job[0].pid)
            code=finish(job)
            if not code:
                states=[json.loads(path.read_text()) for path in out.glob('checkpoint-*/trainer_state.json')]
                if not states or max(v['global_step'] for v in states)!=2:
                    raise ValueError('Smoke did not complete2updates')
                last=max(states,key=lambda v:v['global_step'])
                losses=[v['loss'] for v in last['log_history'] if 'loss' in v]
                if not losses or not all(math.isfinite(x) for x in losses):raise ValueError('Nonfinite_smoke_loss')
                chosen=batch;save(root/'smoke_verified.json',dict(batch=batch,steps=2,losses=losses,eval_prompts=32));break
            if batch==16 or 'out of memory' not in (root/f'smoke_batch{batch}.log').read_text().lower():
                raise RuntimeError('DenseRLsmokefailed; inspectlog')
        command,out=training_command(snapshot,root,chosen,fast=o.fast)
        save(root/'formal_training_choice.json',dict(batch_per_device=chosen,accumulation=1024//(2*chosen),
            global_candidate_batch=1024,initial_model=str(MODEL),smoke_weights_not_used=True))
        check_gpu_capacity(gpu_snapshot(gpus),gpus,False,35)
        job=start(command,root/'rl.log')
        status('training_dense_rl',worker_pid=job[0].pid,batch_per_device=chosen,accumulation=1024//(2*chosen))
        if finish(job):raise RuntimeError('Formal denseRLfailed; preservecheckpoints')
        final=out/'final_checkpoint'
        if not all((final/name).is_file() for name in ('model.safetensors','config.json','tokenizer_config.json')):
            raise ValueError('FinaldenseRLmodel incomplete')
        save(root/'rl_complete.json',dict(checkpoint=str(final),exit_code=0,completed_at=time.time()))
        with (DATA/'valid'/f'{STEM}.csv').open() as f:source=list(csv.DictReader(f))
        with (OLD_DATA/'valid'/f'{STEM}.csv').open() as f:old_source=list(csv.DictReader(f))
        validate_alignment(old_source,source);old_index=load_index(OLD_INDEX)
        for scope in ('sample256','valid20pct'):
            dest=root/'eval'/scope;dest.mkdir(parents=True)
            sample_file=dest/'sample.json';shutil.copy2(CONTROLS/scope/'sample.json',sample_file)
            sample=json.loads(sample_file.read_text());results={}
            for name,path,model,kind,dataset,sid_index,mapping in (
                ('new_sid_no_latent',SFT_SOURCE/'eval'/scope/'new_sid_no_latent.json',NATIVE,'sid',DATA,INDEX,index),
                ('reference_history_latent',SFT_SOURCE/'eval'/scope/'new_history_latent.json',MODEL,'latent',DATA,INDEX,index),
                ('old_sid_no_latent',CONTROLS/scope/'no_latent_sft.json',OLD_NATIVE,'sid',OLD_DATA,OLD_INDEX,old_index),
                ('old_history_latent',CONTROLS/scope/'three_slot_vq_sft.json',OLD_LATENT,'latent',OLD_DATA,OLD_INDEX,old_index)):
                cache=json.loads(path.read_text());validate_cache(cache,model,kind,sample_file,dataset/'valid'/f'{STEM}.csv',sid_index,snapshot)
                validate_result(cache,sample,model,kind);save(dest/(name+'.json'),cache)
                results[name]=canonical_result(cache,mapping,source,sample['row_ids'])
            check_gpu_capacity(gpu_snapshot(gpus),gpus,False,20)
            jobs,parts,samples=[],[],[]
            for shard,gpu in enumerate(gpus):
                sub=dict(sample,row_ids=sample['row_ids'][shard::2],shard=shard,shards=2)
                ps=dest/f'sample.part{shard}.json';save(ps,sub);samples.append(sub)
                output=dest/f'dense_rl.part{shard}.json';parts.append(output)
                command=[PYTHON,'-u','-m','latent.eval_unified_small','--model',str(final),
                    '--kind','latent','--data',str(DATA/'valid'/f'{STEM}.csv'),'--index',str(INDEX),
                    '--sample',str(ps),'--output',str(output),'--max-batch','32']
                jobs.append(start(command,dest/f'eval.part{shard}.log',gpu))
            status('ranking_eval',scope=scope,worker_pids=[job[0].pid for job in jobs])
            if any([finish(job) for job in jobs]):raise RuntimeError('DenseRLrankingevalfailed')
            fresh=merge_shards([json.loads(path.read_text()) for path in parts],samples,sample)
            fresh['metadata']['sample_sha256']=sha(sample_file)
            validate_cache(fresh,final,'latent',sample_file,DATA/'valid'/f'{STEM}.csv',INDEX,snapshot)
            save(dest/'new_history_latent.json',fresh)
            results['new_history_latent']=canonical_result(fresh,index,source,sample['row_ids'])
            report(dest,sample,results)
            text=(dest/'comparison.md').read_text().replace('| 新SID＋新历史兴趣 |','| 新SID＋新历史兴趣，4×4 dense RL |').replace('| 新SID＋新历史兴趣，权重1对照 |','| 新SID＋新历史兴趣，SFT权重0.2 |')
            (dest/'comparison.md').write_text(text+'\nRL兴趣项权重1，dense部分匹配系数0.2；0.2SFT只指定初始checkpoint。\n')
        status('complete',checkpoint=str(final),comparison=str(root/'eval/valid20pct/comparison.md'))
    except Exception:
        status('failed',error=traceback.format_exc());raise
    finally:
        for worker in active:
            if worker.poll() is None:
                os.killpg(worker.pid,signal.SIGTERM)
                try:worker.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(worker.pid,signal.SIGKILL);worker.wait()
        for stream in streams:stream.close()

if __name__=='__main__':main()
