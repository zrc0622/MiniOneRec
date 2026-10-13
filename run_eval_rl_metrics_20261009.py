import json,os,subprocess,time
from pathlib import Path
from latent.eval_search_matrix_report import reduce
ROOT=Path('/data/zuorongchang/project/MiniOneRec');os.chdir(ROOT)
OUT=ROOT/'outputs/eval_rl_metrics_20261009';OUT.mkdir(exist_ok=True)
REF=ROOT/'outputs/eval_search_compare_20261009';BASE=ROOT/'outputs/latent_qwen3_0.6b_balanced15/rl_two/train.Trz3R6'
PYTHON='/data/zuorongchang/miniconda3/envs/minionerec/bin/python'
MODELS={'binary_final':BASE/'binary/final_checkpoint','binary_2465':BASE/'binary/checkpoint-2465','dense_2465':BASE/'sid_dense/checkpoint-2465'}
methods=['single10','four10_sum'];workers=[]
try:
 for gpu,(name,model) in zip([2,4,6],MODELS.items()):
  memory=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
  if memory>2048:raise RuntimeError(f'GPU{gpu} no longer idle: {memory}MiB')
  argv=[PYTHON,'-u','-m','latent.eval_search_matrix','--model',str(model),'--data','data/Amazon23/variants/rqkmeans_balanced/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv','--index','data/Amazon23/variants/rqkmeans_balanced/Industrial_and_Scientific/Industrial_and_Scientific.index.json','--sample',str(REF/'sample256.json'),'--output',str(OUT/f'{name}.json'),'--mode','confirm','--methods',*methods]
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
  with (OUT/f'{name}.log').open('w') as log:p=subprocess.Popen(argv,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  workers.append((name,gpu,p));print('started',name,gpu,p.pid,flush=True)
 (OUT/'workers.json').write_text(json.dumps([dict(name=n,gpu=g,pid=p.pid,model=str(MODELS[n])) for n,g,p in workers],indent=2))
 while any(p.poll() is None for _,_,p in workers):
  (OUT/'status.json').write_text(json.dumps(dict(phase='running',workers=[dict(name=n,pid=p.pid,exit_code=p.poll()) for n,g,p in workers],updated_at=time.time()),indent=2));time.sleep(10)
 for n,g,p in workers:
  if p.returncode:raise RuntimeError((n,p.returncode))
 ref=json.loads((REF/'screen_nonbalanced_vq.json').read_text());identity=[(r['row_id'],r['output']) for r in ref['rows']['single10']]
 results={};sources={}
 for name in ['no_latent_sft','nonbalanced_vq']:
  d=json.loads((REF/f'screen_{name}.json').read_text());selected=['direct10','direct50'] if name=='no_latent_sft' else methods
  results[name]={m:reduce(d['rows'][m]) for m in selected}
  sources[name]=d['metadata']
 for name in MODELS:
  d=json.loads((OUT/f'{name}.json').read_text());results[name]={}
  for m in methods:
   assert [(r['row_id'],r['output']) for r in d['rows'][m]]==identity
   results[name][m]=reduce(d['rows'][m])
  sources[name]=d['metadata']
 (OUT/'comparison.json').write_text(json.dumps(dict(results=results,source_metadata=sources,examples=256,split='valid',note='Same fixed256 rows. RL complete-request timing; SFT latent reference screen shared-stage timing. Binary final4928 vs Dense2465; Binary2465 included for matched training steps.'),indent=2))
 lines=['# RL and SFT on fixed256 validation rows','','| Model | Eval | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 | Mean seconds | Full10 |','|---|---|---:|---:|---:|---:|---:|---:|']
 for name,rs in results.items():
  for method,m in rs.items():lines.append(f"| {name} | {method} | {m['Recall@5']:.5f} | {m['Recall@10']:.5f} | {m['NDCG@5']:.5f} | {m['NDCG@10']:.5f} | {m['mean_seconds']:.3f} | {m['fraction_with10']:.0%} |")
 lines+=['','Singlepositive target: Recall=HR. Same256valid seed42; no test selection. Binary final4928 vs Dense2465; Binary2465 included for matchedsteps. RL times are independent full requests; latentSFT table uses prior composed shared-stage screen timings, so refer to prior confirmation for fairer speed comparison. Small sample diagnostic, not significance. Training and attention unchanged.']
 (OUT/'comparison.md').write_text('\n'.join(lines)+'\n')
 (OUT/'status.json').write_text(json.dumps(dict(phase='completed',updated_at=time.time())))
 print('\n'.join(lines),flush=True)
except Exception as e:
 (OUT/'status.json').write_text(json.dumps(dict(phase='failed',error=repr(e),updated_at=time.time())));raise
