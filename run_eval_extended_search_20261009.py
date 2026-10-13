"""Durable queue for broader validation-only eval exploration."""
import json,os,subprocess,time
from pathlib import Path
from latent.eval_search_matrix_report import reduce
ROOT=Path('/data/zuorongchang/project/MiniOneRec');os.chdir(ROOT)
OUT=ROOT/'outputs/eval_extended_search_20261009';OUT.mkdir(exist_ok=True)
REF=ROOT/'outputs/eval_search_compare_20261009';RLREF=ROOT/'outputs/eval_rl_metrics_20261009'
PYTHON='/data/zuorongchang/miniconda3/envs/minionerec/bin/python'
MODELS={'nonbalanced_vq':ROOT/'outputs/latent_qwen3_0.6b_balanced15/sft/final_checkpoint','dense_2465':ROOT/'outputs/latent_qwen3_0.6b_balanced15/rl_two/train.Trz3R6/sid_dense/checkpoint-2465','no_latent_rl':ROOT/'outputs/amazon23_industrial_qwen3_0.6b_rl_balanced_15ep/final_checkpoint'}
DATA='data/Amazon23/variants/rqkmeans_balanced/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv';INDEX='data/Amazon23/variants/rqkmeans_balanced/Industrial_and_Scientific/Industrial_and_Scientific.index.json'
GPUS=[2,4,6];records=[];active={};completed=set();selected=None;baseline_selected=None
pending=[('screen',name) for name in MODELS]

def free(gpu):
 mem=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
 return mem<2048

def valid(phase,name):
 d=json.loads((OUT/f'{phase}_{name}.json').read_text());ids=json.loads((REF/('sample64.json' if phase=='screen' else 'sample256.json')).read_text())['row_ids']
 for rows in d['rows'].values():assert [r['row_id'] for r in rows]==ids
 return d

def save_report(phase):
 names=list(MODELS);ids=json.loads((REF/('sample64.json' if phase=='screen' else 'sample256.json')).read_text())['row_ids'];results={};sources={};identity=None
 for name in names:
  d=valid(phase,name);results[name]={};sources[name]=d['metadata']
  for m,rs in d['rows'].items():
   current=[(r['row_id'],r['output']) for r in rs]
   if identity is None:identity=current
   assert current==identity
   results[name][m]=reduce(rs)
 for name,path in [('nonbalanced_vq',REF/'screen_nonbalanced_vq.json'),('dense_2465',RLREF/'dense_2465.json'),('no_latent_rl',RLREF/'no_latent_rl.json'),('no_latent_sft',REF/'screen_no_latent_sft.json')]:
  d=json.loads(path.read_text());sources[name+'_reference']=d['metadata'];results.setdefault(name,{})
  wanted=['single10','four10_sum'] if name in ['nonbalanced_vq','dense_2465'] else ['direct10','direct50']
  for method in wanted:
   rows=[r for r in d['rows'][method] if r['row_id'] in set(ids)];assert [(r['row_id'],r['output']) for r in rows]==identity
   results[name][method+'_previous']=reduce(rows)
 output=dict(examples=len(ids),split='valid',results=results,source_metadata=sources,selected=selected,baseline_selected=baseline_selected,note='Fixedseed42 diagnostic. screen64 nested in confirm256, not independent holdout. Screen composed timings; confirm standalone. Prior powers/RRF are ranking diagnostics, alpha1 probabilitysum is model marginal approximation. Existing reference latent SFT screen has composed timing; RL refs independent. Fullvocab probabilities, unchanged attention, deterministic SID constrained generation. Top10 only.')
 (OUT/f'{phase}_comparison.json').write_text(json.dumps(output,indent=2))
 lines=[f'# Extended eval: {phase}, {len(ids)} fixedvalidation rows','','| Model | Method | R5 | R10 | N5 | N10 | Mean s | Full10 |','|---|---|---:|---:|---:|---:|---:|---:|']
 for name,ms in results.items():
  for m,x in sorted(ms.items(),key=lambda pair:-pair[1]['NDCG@10']):lines.append(f"| {name} | {m} | {x['Recall@5']:.5f} | {x['Recall@10']:.5f} | {x['NDCG@5']:.5f} | {x['NDCG@10']:.5f} | {x['mean_seconds']:.3f} | {x['fraction_with10']:.0%} |")
 lines+=['',output['note']];(OUT/f'{phase}_comparison.md').write_text('\n'.join(lines)+'\n')
 return output
try:
 while pending or active:
  for gpu,item in list(active.items()):
   proc,phase,name=item
   if proc.poll() is not None:
    if proc.returncode:raise RuntimeError((phase,name,proc.returncode))
    valid(phase,name);completed.add((phase,name));del active[gpu];print('completed',phase,name,flush=True)
  if baseline_selected is None and ('screen','no_latent_rl') in completed:
   d=valid('screen','no_latent_rl');rs={m:reduce(rows) for m,rows in d['rows'].items()};old=json.loads((RLREF/'no_latent_rl.json').read_text());ids=set(json.loads((REF/'sample64.json').read_text())['row_ids'])
   for m,rows in old['rows'].items():rs[m]=reduce([r for r in rows if r['row_id'] in ids])
   # Always expand widest beam to256 regardless of tiny64 ties; plus the best new width.
   best=max(rs,key=lambda m:(rs[m]['NDCG@10'],-rs[m]['mean_seconds']))
   baseline_selected=list(dict.fromkeys(['direct200',best]));pending.append(('confirm','no_latent_rl'))
  if selected is None and all(('screen',name) in completed for name in ['nonbalanced_vq','dense_2465']):
   ds={n:valid('screen',n) for n in ['nonbalanced_vq','dense_2465']};methods=list(ds['nonbalanced_vq']['rows'])
   def merit(m):return (sum(reduce(d['rows'][m])['NDCG@10'] for d in ds.values()),-sum(reduce(d['rows'][m])['mean_seconds'] for d in ds.values()))
   faithful=[m for m in methods if m.endswith('_sum_a1') or m.endswith('_marg_a1') or m.startswith('joint')]
   faithful=[m for m in faithful if m not in ['p4b10_sum_a1','p4b10_marg_a1']]
   best_prob=max(faithful,key=merit)
   # Predeclared diagnostic: equal-prior sum on the original4x10 proposals.
   selected=list(dict.fromkeys([best_prob,'p4b10_sum_a0']))
   (OUT/'selection.json').write_text(json.dumps(dict(methods=selected,best_probability_variant=best_prob,criterion='MeanNDCG10 across SFT and Dense on64; second diagnostic equalprior4x10 predeclared.256 overlaps64, diagnostic only.'),indent=2))
   pending.extend(('confirm',name) for name in ds)
  for gpu in GPUS:
   if not pending or gpu in active or not free(gpu):continue
   phase,name=pending.pop(0);no_latent=name=='no_latent_rl'
   module='latent.eval_search_matrix' if no_latent else 'latent.eval_extended_search'
   argv=[PYTHON,'-u','-m',module,'--model',str(MODELS[name]),'--data',DATA,'--index',INDEX,'--sample',str(REF/('sample64.json' if phase=='screen' else 'sample256.json')),'--output',str(OUT/f'{phase}_{name}.json'),'--mode','confirm' if no_latent or phase=='confirm' else 'screen']
   if no_latent:argv+=['--no-latent','--methods']+(['direct5','direct20','direct100','direct200'] if phase=='screen' else baseline_selected)
   elif phase=='confirm':argv+=['--methods']+selected
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
   with (OUT/f'{phase}_{name}.log').open('w') as log:p=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
   active[gpu]=(p,phase,name);records.append(dict(gpu=gpu,pid=p.pid,phase=phase,name=name,argv=argv));(OUT/'workers.json').write_text(json.dumps(records,indent=2));print('started',phase,name,gpu,p.pid,flush=True)
  if all(('screen',n) in completed for n in MODELS) and not (OUT/'screen_comparison.json').exists():save_report('screen')
  (OUT/'status.json').write_text(json.dumps(dict(phase='running',active={g:dict(pid=p.pid,phase=ph,name=n) for g,(p,ph,n) in active.items()},pending=pending,completed=sorted(completed),updated_at=time.time()),indent=2))
  if active or pending:time.sleep(10)
 save_report('confirm');(OUT/'status.json').write_text(json.dumps(dict(phase='completed',methods=selected,baseline_methods=baseline_selected,updated_at=time.time()),indent=2));print('all completed',flush=True)
except Exception as e:
 (OUT/'status.json').write_text(json.dumps(dict(phase='failed',error=repr(e),updated_at=time.time())));raise
