"""One high-budget throughput worker per idle GPU, SFT thenDense."""
import json,os,subprocess,time
from pathlib import Path
import numpy as np
from interest.evaluate import summarize
ROOT=Path('/data/zuorongchang/project/MiniOneRec');os.chdir(ROOT)
OUT=ROOT/'outputs/eval_high_budget_batched_20261010';OUT.mkdir(exist_ok=True);REF=ROOT/'outputs/eval_search_compare_20261009';OLD=ROOT/'outputs/eval_at20_20261009'
PYTHON='/data/zuorongchang/miniconda3/envs/minionerec/bin/python'
MODELS={'nonbalanced_vq':'outputs/latent_qwen3_0.6b_balanced15/sft/final_checkpoint','dense_2465':'outputs/latent_qwen3_0.6b_balanced15/rl_two/train.Trz3R6/sid_dense/checkpoint-2465'}
QUEUES={4:[(n,'eight20_sum') for n in MODELS],5:[(n,'sixteen20_sum') for n in MODELS],6:[(n,'eight50_sum') for n in MODELS]};active={};records=[];completed=[]
DATA='data/Amazon23/variants/rqkmeans_balanced/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv';INDEX='data/Amazon23/variants/rqkmeans_balanced/Industrial_and_Scientific/Industrial_and_Scientific.index.json'
KEYS=['Recall@5','Recall@10','Recall@20','NDCG@5','NDCG@10','NDCG@20']
def read(name,method):
 d=json.loads((OUT/f'{name}_{method}.json').read_text());rs=d['rows'][method];ids=json.loads((REF/'sample256.json').read_text())['row_ids'];assert [r['row_id'] for r in rs]==ids;assert d['metadata']['completed']==256;return d

def report():
 results={};sources={};reference=None
 for name in MODELS:
  results[name]={}
  for method in ['eight20_sum','sixteen20_sum','eight50_sum']:
   d=read(name,method);rs=d['rows'][method];identity=[(r['row_id'],r['output']) for r in rs]
   if reference is None:reference=identity
   assert identity==reference
   raw=summarize(rs);m={k:raw[k] for k in KEYS};time=sum(b['elapsed_seconds'] for b in d['batches'])
   m.update(mean_amortized_seconds=time/256,throughput_rows_per_second=256/time,mean_unique_proposals=float(np.mean([r['unique_before_top20'] for r in rs])),fraction_with20=float(np.mean([len(r['predict'])>=20 for r in rs])),batch_sizes=sorted(set(b['size'] for b in d['batches'])),max_peak_allocated_gib=max(b['peak_allocated_gib'] for b in d['batches']),mean_selected_latent_mass=float(np.mean([r['selected_latent_mass'] for r in rs])),oom_retries=d['oom_retries'],hits20=sum(r['output'] in r['predict'][:20] for r in rs))
   results[name][method]=m;sources[f'{name}_{method}']=d['metadata']
 old=json.loads((OLD/'comparison.json').read_text())
 for name in ['nonbalanced_vq','dense_2465','no_latent_rl']:
  results.setdefault(name,{})
  for method in (['single20','four20_sum'] if name!='no_latent_rl' else ['direct50']):
   d=json.loads((OLD/f'screen_{name}.json').read_text());assert [(r['row_id'],r['output']) for r in d['rows'][method]]==reference
   results[name][method+'_previous']={k:old['results'][name][method][k] for k in KEYS}
 scope='Fixed256valid, originalVQ SFT andDense2465. Allsixmetrics primary. Newtimes amortized batchthroughput, NOT singleuserlatency, exclude upfronttokenization/load/warmup. Lengthbucketed leftpad andcache may alter BF16ranking; interestprior comesfromfullvocab unrenormalized beam likelihood. Previousindividualreference probabilities usecachedteacherforce, tinyFP32 mathchecked but no productionbitwiseclaim. Oneworker/GPU4/5/6, maxbatch8 automaticallycapped; no attention/training/testchanges. Small sample, no significance.'
 (OUT/'comparison.json').write_text(json.dumps(dict(results=results,source_metadata=sources,examples=256,scope=scope),indent=2))
 lines=['# High-budget batched eval, matched256validation','','| Model | Method | R5 | R10 | R20 | N5 | N10 | N20 | Batch-amortized s | Unique proposals | Full20 | Batches | Peak allocated GiB |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|']
 for name,ms in results.items():
  for method,m in ms.items():
   cells=' | '.join(f'{m[k]:.5f}' for k in KEYS)
   if 'mean_amortized_seconds' in m:extra=f"{m['mean_amortized_seconds']:.3f} | {m['mean_unique_proposals']:.2f} | {m['fraction_with20']:.1%} | {m['batch_sizes']} | {m['max_peak_allocated_gib']:.2f}"
   else:extra='— | — | — | — | —'
   lines.append(f'| {name} | {method} | {cells} | {extra} |')
 lines+=['',scope];(OUT/'comparison.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
try:
 while active or any(QUEUES.values()):
  for gpu,(p,name,method) in list(active.items()):
   if p.poll() is not None:
    if p.returncode:raise RuntimeError((name,method,p.returncode))
    read(name,method);completed.append((name,method));del active[gpu];print('complete',name,method,flush=True)
  for gpu,queue in QUEUES.items():
   if not queue or gpu in active:continue
   mem=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
   if mem>2048:continue
   name,method=queue.pop(0);argv=[PYTHON,'-u','-m','latent.eval_batched_high_budget','--model',MODELS[name],'--data',DATA,'--index',INDEX,'--sample',str(REF/'sample256.json'),'--output',str(OUT/f'{name}_{method}.json'),'--method',method,'--max-batch','8']
   with (OUT/f'{name}_{method}.log').open('w') as log:p=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false'))
   active[gpu]=(p,name,method);records.append(dict(pid=p.pid,gpu=gpu,name=name,method=method));(OUT/'workers.json').write_text(json.dumps(records,indent=2));print('started',gpu,name,method,p.pid,flush=True)
  (OUT/'status.json').write_text(json.dumps(dict(phase='running',active={g:dict(pid=p.pid,name=n,method=m) for g,(p,n,m) in active.items()},queued=QUEUES,completed=completed,updated_at=time.time()),indent=2))
  if active or any(QUEUES.values()):time.sleep(10)
 report();(OUT/'status.json').write_text(json.dumps(dict(phase='completed',updated_at=time.time(),completed=completed),indent=2));print('allcompleted',flush=True)
except Exception as e:
 (OUT/'status.json').write_text(json.dumps(dict(phase='failed',error=repr(e),updated_at=time.time())));raise
