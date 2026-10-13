"""Bounded durable queue: only idle authorized GPUs, one process per GPU."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path('/data/zuorongchang/project/MiniOneRec');OUT=ROOT/'outputs/eval_search_compare_20261009';PYTHON='/data/zuorongchang/miniconda3/envs/minionerec/bin/python'
os.chdir(ROOT)
MODELS={
 'nonbalanced_vq':'outputs/latent_qwen3_0.6b_balanced15/sft/final_checkpoint',
 'balanced_vq':'outputs/latent_qwen3_0.6b_balanced_vq15/sft/final_checkpoint',
 'balanced_rqkmeans':'outputs/latent_qwen3_0.6b_balanced_rqkmeans15/sft/final_checkpoint',
 'no_latent_sft':'outputs/amazon23_industrial_qwen3_0.6b_sft_balanced_15ep/final_checkpoint'}
GPUS=[2,4,6]
cmd=[PYTHON,'-u','-m','latent.eval_search_matrix','--data','data/Amazon23/variants/rqkmeans_balanced/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv','--index','data/Amazon23/variants/rqkmeans_balanced/Industrial_and_Scientific/Industrial_and_Scientific.index.json']
def state(phase,**kw):
 x=dict(phase=phase,updated_at=time.time(),**kw);(OUT/'status.json').write_text(json.dumps(x,indent=2)+'\n');print(x,flush=True)
def free(gpu):
 rows=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
 # GPU6 has a small persistent noncompute allocation; user authorized this card.
 return int(rows.strip()) < (2048 if gpu==6 else 1024)
def queue(jobs,phase,methods=None):
 pending=list(jobs);active={};records=[]
 while pending or active:
  for gpu,(proc,name) in list(active.items()):
   code=proc.poll()
   if code is not None:
    if code!=0:raise RuntimeError(f'{phase} {name} onGPU{gpu} failed:{code}')
    del active[gpu];print('completed',phase,name,gpu,flush=True)
  for gpu in GPUS:
   if pending and gpu not in active and free(gpu):
    name=pending.pop(0);path=OUT/f'{phase}_{name}.json'
    argv=cmd+['--model',MODELS[name],'--sample',str(OUT/('sample256.json' if phase=='screen' else 'sample64.json')),'--output',str(path),'--mode','screen' if phase=='screen' else 'confirm']
    if name=='no_latent_sft':argv+=['--no-latent']
    if methods:argv+=['--methods']+methods
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
    with (OUT/f'{phase}_{name}.log').open('w') as log:
     proc=subprocess.Popen(argv,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    active[gpu]=(proc,name);records.append(dict(gpu=gpu,pid=proc.pid,name=name,phase=phase))
    (OUT/f'{phase}_workers.json').write_text(json.dumps(records,indent=2)+'\n')
    print('started',records[-1],flush=True)
  state(phase,active={g:dict(pid=p.pid,name=n) for g,(p,n) in active.items()},pending=pending)
  if active or pending:time.sleep(10)
from latent.eval_search_matrix_report import report
try:
 queue(list(MODELS),'screen')
 result=report(OUT,'screen')
 # Use same confirmation methods for all latent checkpoints. Predeclare main
 # mathematically distinct approximations; add best observed-sum variant on validation.
 variants=['two10_sum','four10_sum','four20_sum','budget80_sum']
 best=max(variants,key=lambda name:(sum(result['results'][m][name]['NDCG@10'] for m in list(MODELS)[:3]),-sum(result['results'][m][name]['mean_seconds'] for m in list(MODELS)[:3])))
 methods=list(dict.fromkeys(['single10','four10_sum','four10_marginal',best]))
 (OUT/'confirmation_methods.json').write_text(json.dumps(dict(methods=methods,selected_observed_sum=best,selection='Best meanNDCG10 across3checkpoints on samevalidation256, tiebreak latency; confirmation64 is nested, not independent qualityvalidation.'),indent=2)+'\n')
 queue(list(MODELS)[:3],'confirm',methods)
 report(OUT,'confirm')
 state('completed',methods=methods)
 (OUT/'completed.json').write_text(json.dumps(dict(completed_at=time.time()))+'\n')
except Exception as exc:
 state('failed',error=repr(exc));raise
