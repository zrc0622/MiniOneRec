"""Adopt existing eval workers and overlap baseline with timing confirmations."""
import json,os,subprocess,time
from pathlib import Path
from latent.eval_search_matrix_report import reduce,report
ROOT=Path('/data/zuorongchang/project/MiniOneRec');OUT=ROOT/'outputs/eval_search_compare_20261009';PYTHON='/data/zuorongchang/miniconda3/envs/minionerec/bin/python'
os.chdir(ROOT)
MODELS={'nonbalanced_vq':'outputs/latent_qwen3_0.6b_balanced15/sft/final_checkpoint','balanced_vq':'outputs/latent_qwen3_0.6b_balanced_vq15/sft/final_checkpoint','balanced_rqkmeans':'outputs/latent_qwen3_0.6b_balanced_rqkmeans15/sft/final_checkpoint','no_latent_sft':'outputs/amazon23_industrial_qwen3_0.6b_sft_balanced_15ep/final_checkpoint'}
GPUS=[2,4,6];LATENT=list(MODELS)[:3]
cmd=[PYTHON,'-u','-m','latent.eval_search_matrix','--data','data/Amazon23/variants/rqkmeans_balanced/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv','--index','data/Amazon23/variants/rqkmeans_balanced/Industrial_and_Scientific/Industrial_and_Scientific.index.json']
def live(pid,name):
 try:c=(Path('/proc')/str(pid)/'cmdline').read_bytes()
 except OSError:return False
 return b'latent.eval_search_matrix' in c and MODELS[name].encode() in c
def free(gpu):
 x=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
 return int(x.strip())<(2048 if gpu==6 else 1024)
def valid(phase,name):
 d=json.loads((OUT/f'{phase}_{name}.json').read_text())
 expected=json.loads((OUT/('sample256.json' if phase=='screen' else 'sample64.json')).read_text())['row_ids']
 assert all([r['row_id'] for r in rs]==expected for rs in d['rows'].values()),(phase,name)
 return d
records=json.loads((OUT/'screen_workers.json').read_text());active={};completed=set();pending=[]
for r in records:
 if live(r['pid'],r['name']):active[r['gpu']]=dict(record=r,proc=None)
 else:valid(r['phase'],r['name']);completed.add((r['phase'],r['name']))
if not any(r['name']=='no_latent_sft' for r in records):pending.append(('screen','no_latent_sft'))
methods=None
try:
 while active or pending or methods is None:
  for gpu,item in list(active.items()):
   r=item['record'];proc=item['proc']
   done=(proc.poll() is not None) if proc is not None else not live(r['pid'],r['name'])
   if done:
    if proc is not None and proc.returncode!=0:raise RuntimeError((r,proc.returncode))
    valid(r['phase'],r['name']);completed.add((r['phase'],r['name']));del active[gpu]
    print('complete',r,'exit_code' if proc is not None else 'adopted_artifact_verified',flush=True)
  if methods is None and all(('screen',n) in completed for n in LATENT):
   ds={n:valid('screen',n) for n in LATENT}
   variants=['two10_sum','four10_sum','four20_sum','budget80_sum']
   best=max(variants,key=lambda m:(sum(reduce(ds[n]['rows'][m])['NDCG@10'] for n in LATENT),-sum(reduce(ds[n]['rows'][m])['mean_seconds'] for n in LATENT)))
   methods=list(dict.fromkeys(['single10','four10_sum','four10_marginal',best]))
   (OUT/'confirmation_methods.json').write_text(json.dumps(dict(methods=methods,selected_observed_sum=best,selection='BestglobalmeanNDCG10 across3checkpoints onvalidation256;64nestedconfirmation measures standalone latency, not independent qualityholdout.'),indent=2)+'\n')
   pending.extend(('confirm',n) for n in LATENT)
  for gpu in GPUS:
   if pending and gpu not in active and free(gpu):
    phase,name=pending.pop(0);argv=cmd+['--model',MODELS[name],'--sample',str(OUT/('sample256.json' if phase=='screen' else 'sample64.json')),'--output',str(OUT/f'{phase}_{name}.json'),'--mode','screen' if phase=='screen' else 'confirm']
    if name=='no_latent_sft':argv+=['--no-latent']
    if phase=='confirm':argv+=['--methods']+methods
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
    with (OUT/f'{phase}_{name}.log').open('w') as log:
     proc=subprocess.Popen(argv,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    r=dict(gpu=gpu,pid=proc.pid,name=name,phase=phase);active[gpu]=dict(record=r,proc=proc);records.append(r)
    (OUT/'all_workers.json').write_text(json.dumps(records,indent=2)+'\n');print('started',r,flush=True)
  status=dict(phase='screen_and_confirmation',updated_at=time.time(),active={g:x['record'] for g,x in active.items()},pending=pending,completed=sorted(completed))
  (OUT/'status.json').write_text(json.dumps(status,indent=2)+'\n')
  if active or pending or methods is None:time.sleep(10)
 report(OUT,'screen');report(OUT,'confirm')
 (OUT/'status.json').write_text(json.dumps(dict(phase='completed',methods=methods,updated_at=time.time()),indent=2)+'\n')
 (OUT/'completed.json').write_text(json.dumps(dict(completed_at=time.time(),adopted_workers_verified=True))+'\n')
except Exception as exc:
 (OUT/'status.json').write_text(json.dumps(dict(phase='failed',error=repr(exc),updated_at=time.time()),indent=2)+'\n');raise
