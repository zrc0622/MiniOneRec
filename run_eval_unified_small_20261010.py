"""Eightcheckpoint uniform14evals, idleGPU4/5/6, batchedthroughput."""
import json,os,subprocess,time
from pathlib import Path
import numpy as np
from interest.evaluate import summarize
ROOT=Path('/data/zuorongchang/project/MiniOneRec');os.chdir(ROOT)
OUT=ROOT/'outputs/eval_unified_small_20261010';OUT.mkdir(exist_ok=True);REF=ROOT/'outputs/eval_search_compare_20261009'
PYTHON='/data/zuorongchang/miniconda3/envs/minionerec/bin/python';RL='outputs/latent_qwen3_0.6b_balanced15/rl_two/train.Trz3R6'
MODELS={
 'no_latent_sft':dict(label='无latent SFT',kind='sid',model='outputs/amazon23_industrial_qwen3_0.6b_sft_balanced_15ep/final_checkpoint'),
 'no_latent_rl':dict(label='无latent RL',kind='sid',model='outputs/amazon23_industrial_qwen3_0.6b_rl_balanced_15ep/final_checkpoint',steps=4928),
 'vq_sft':dict(label='原VQ-VAE latent SFT',kind='latent',model='outputs/latent_qwen3_0.6b_balanced15/sft/final_checkpoint',quantizer='parallel_VQ_VAE'),
 'vq_rl_binary_final':dict(label='VQ-VAE RL Binary最终',kind='latent',model=RL+'/binary/final_checkpoint',steps=4928,quantizer='parallel_VQ_VAE'),
 'vq_rl_binary_2465':dict(label='VQ-VAE RL Binary2465',kind='latent',model=RL+'/binary/checkpoint-2465',steps=2465,quantizer='parallel_VQ_VAE'),
 'vq_rl_dense_2465':dict(label='VQ-VAE RL Dense2465',kind='latent',model=RL+'/sid_dense/checkpoint-2465',steps=2465,quantizer='parallel_VQ_VAE'),
 'balanced_vq_sft':dict(label='平衡VQ-VAE latent SFT',kind='latent',model='outputs/latent_qwen3_0.6b_balanced_vq15/sft/final_checkpoint',steps=4140,quantizer='parallel_balanced_VQ_VAE'),
 'balanced_rqkmeans_sft':dict(label='平衡RQ-kmeans latent SFT',kind='latent',model='outputs/latent_qwen3_0.6b_balanced_rqkmeans15/sft/final_checkpoint',steps=4140,quantizer='balanced_residual_kmeans')}
for name,m in MODELS.items():
 assert (ROOT/m['model']/'config.json').exists(),name
 assert (ROOT/m['model']/'model.safetensors').exists(),name
(OUT/'manifest.json').write_text(json.dumps(dict(models=MODELS,methods=['single20','four20_marginal'],sid_reference='direct50',sample=str(REF/'sample256.json'),note='User callsVAE RQ-VAE; actualimplementationParallelVQ no residual. RL3 interpreted as priorcomparedBinaryfinal/Binary2465/Dense2465.'),ensure_ascii=False,indent=2))
GPUS=[4,5,6];queue=[n for n,m in MODELS.items() if m['kind']=='latent']+[n for n,m in MODELS.items() if m['kind']=='sid'];active={};completed=[];records=[]
DATA='data/Amazon23/variants/rqkmeans_balanced/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv';INDEX='data/Amazon23/variants/rqkmeans_balanced/Industrial_and_Scientific/Industrial_and_Scientific.index.json'
KEYS=['Recall@5','Recall@10','Recall@20','NDCG@5','NDCG@10','NDCG@20']

def valid(name):
 d=json.loads((OUT/f'{name}.json').read_text());ids=json.loads((REF/'sample256.json').read_text())['row_ids'];expected=['direct50'] if MODELS[name]['kind']=='sid' else ['single20','four20_marginal'];assert set(d['rows'])==set(expected)
 for method in expected:assert [r['row_id'] for r in d['rows'][method]]==ids
 return d

def report():
 results={};sources={};identity=None
 for name,m in MODELS.items():
  d=valid(name);sources[name]=d['metadata'];results[name]={}
  for method,rs in d['rows'].items():
   current=[(r['row_id'],r['output']) for r in rs]
   if identity is None:identity=current
   assert current==identity
   assert all(len(r['predict'])==20 and len(set(r['predict']))==20 for r in rs)
   raw=summarize(rs);ms={k:raw[k] for k in KEYS};elapsed=sum(b['elapsed_seconds'] for b in d['batches'][method]);ms.update(mean_amortized_seconds=elapsed/256,throughput_rows_per_second=256/elapsed,batch_sizes=sorted(set(b['size'] for b in d['batches'][method])),fraction_with20=float(np.mean([len(r['predict'])>=20 for r in rs])),mean_unique_proposals=float(np.mean([r['unique_before_top20'] for r in rs])),max_peak_allocated_gib=max(b['peak_allocated_gib'] for b in d['batches'][method]),hits20=sum(r['output'] in r['predict'][:20] for r in rs))
   results[name][method]=ms
 scope='Same256valid seed42(1.58%16162), actualquantizerParallelVQ-VAE not residualRQ-VAE. Bothlatentprotocols reportedfor all6latentcheckpoints; nativeSIDbeam50 sharedreference. ThreeRLcheckpoints Binaryfinal4928/Binary2465/Dense2465. No training/attention/testchanges. Newbatched BF16results may differ frompreviousindividualruns; exacttinyFP32 protocol/cache/probabilitychecks passed. Fullmarginal scores every unique proposedSID underall4latentcodes usingunrenormalized fullvocabjointlikelihood(scorechunk32). Single20 chooses MAP completecode withininterestbeam4. Time is amortizedbatchthroughput, not singleuserlatency, excludingone-time tokenization/load/warmup. Small sample diagnostic.'
 (OUT/'comparison.json').write_text(json.dumps(dict(results=results,models=MODELS,examples=256,scope=scope,source_metadata=sources),ensure_ascii=False,indent=2))
 lines=['# 八份checkpoint统一小样本评估','',scope]
 for method,label in [('single20','1×20'),('four20_marginal','4×20跨兴趣完整打分求和')]:
  lines+=['',f'## {label}','','| 模型 | Recall@5 | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 |','|---|---:|---:|---:|---:|---:|---:|']
  for name,m in MODELS.items():
   x=results[name]['direct50' if m['kind']=='sid' else method];cells=' | '.join(f'{x[k]:.5f}' for k in KEYS);lines.append(f'| {m["label"]} | {cells} |')
 lines+=['','## 批量推理与候选覆盖','','| 模型 | 协议 | 摊销秒/样本 | Batch | Full20 | 平均候选并集 | 峰值已分配显存GiB |','|---|---|---:|---|---:|---:|---:|']
 for name,ms in results.items():
  for method,m in ms.items():lines.append(f'| {MODELS[name]["label"]} | {method} | {m["mean_amortized_seconds"]:.3f} | {m["batch_sizes"]} | {m["fraction_with20"]:.1%} | {m["mean_unique_proposals"]:.2f} | {m["max_peak_allocated_gib"]:.2f} |')
 (OUT/'comparison.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
try:
 while active or queue:
  for gpu,(p,name) in list(active.items()):
   if p.poll() is not None:
    if p.returncode:raise RuntimeError((name,p.returncode))
    valid(name);completed.append(name);del active[gpu];print('completed',name,flush=True)
  for gpu in GPUS:
   if not queue or gpu in active:continue
   mem=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
   if mem>2048:continue
   name=queue.pop(0);m=MODELS[name];argv=[PYTHON,'-u','-m','latent.eval_unified_small','--model',m['model'],'--kind',m['kind'],'--data',DATA,'--index',INDEX,'--sample',str(REF/'sample256.json'),'--output',str(OUT/f'{name}.json'),'--max-batch','32']
   with (OUT/f'{name}.log').open('w') as log:p=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false'))
   active[gpu]=(p,name);records.append(dict(pid=p.pid,gpu=gpu,name=name));(OUT/'workers.json').write_text(json.dumps(records,indent=2));print('started',gpu,name,p.pid,flush=True)
  (OUT/'status.json').write_text(json.dumps(dict(phase='running',active={g:dict(pid=p.pid,name=n) for g,(p,n) in active.items()},queued=queue,completed=completed,updated_at=time.time()),indent=2))
  if active or queue:time.sleep(10)
 report();(OUT/'status.json').write_text(json.dumps(dict(phase='completed',completed=completed,updated_at=time.time()),indent=2));print('allcompleted',flush=True)
except Exception as e:
 (OUT/'status.json').write_text(json.dumps(dict(phase='failed',error=repr(e),updated_at=time.time())));raise
