import json,os,subprocess,time
from pathlib import Path
import numpy as np
from interest.evaluate import summarize
ROOT=Path('/data/zuorongchang/project/MiniOneRec');os.chdir(ROOT)
OUT=ROOT/'outputs/eval_at20_20261009';OUT.mkdir(exist_ok=True);REF=ROOT/'outputs/eval_search_compare_20261009'
PYTHON='/data/zuorongchang/miniconda3/envs/minionerec/bin/python'
MODELS={'nonbalanced_vq':'outputs/latent_qwen3_0.6b_balanced15/sft/final_checkpoint','dense_2465':'outputs/latent_qwen3_0.6b_balanced15/rl_two/train.Trz3R6/sid_dense/checkpoint-2465','no_latent_sft':'outputs/amazon23_industrial_qwen3_0.6b_sft_balanced_15ep/final_checkpoint','no_latent_rl':'outputs/amazon23_industrial_qwen3_0.6b_rl_balanced_15ep/final_checkpoint'}
GPUS=[4,6,2];active={};records=[];completed=set();selected=None;pending=[('screen',n) for n in MODELS]
DATA='data/Amazon23/variants/rqkmeans_balanced/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv';INDEX='data/Amazon23/variants/rqkmeans_balanced/Industrial_and_Scientific/Industrial_and_Scientific.index.json'
def metrics(rows):
 raw=summarize(rows);m={k:raw[k] for k in ['Recall@5','Recall@10','Recall@20','NDCG@5','NDCG@10','NDCG@20','examples']};times=[r['elapsed_seconds'] for r in rows]
 m.update(mean_seconds=float(np.mean(times)),p95_seconds=float(np.percentile(times,95)),fraction_with20=float(np.mean([len(r['predict'])>=20 for r in rows])),mean_returned_unique=float(np.mean([len(r['predict']) for r in rows])),mean_unique_proposals=float(np.mean([r['unique_before_top20'] for r in rows])) if all('unique_before_top20' in r for r in rows) else None,hits20=sum(r['output'] in r['predict'][:20] for r in rows))
 return m

def valid(phase,name):
 d=json.loads((OUT/f'{phase}_{name}.json').read_text());ids=json.loads((REF/('sample256.json' if phase=='screen' else 'sample64.json')).read_text())['row_ids']
 assert d['metadata']['top_k']==20
 for rows in d['rows'].values():assert [r['row_id'] for r in rows]==ids
 return d

def report(phase):
 names=list(MODELS) if phase=='screen' else ['nonbalanced_vq','dense_2465'];results={};sources={};identity=None
 for name in names:
  d=valid(phase,name);sources[name]=d['metadata'];results[name]={}
  for method,rs in d['rows'].items():
   current=[(r['row_id'],r['output']) for r in rs]
   if identity is None:identity=current
   assert current==identity
   results[name][method]=metrics(rs)
 output=dict(examples=len(identity),results=results,source_metadata=sources,selected=selected,scope='Same seed42 fixedvalidation, onepositive target. Top20 retained, Recall=HR. Oldtop10artifacts not used for@20. screen256 shared-stage timing forlatent, standalone fornativebaselines. confirm64 nested standalone timing, not heldout quality confirmation. Partialcoverage20 explicitly included, no padding or targetoracle. Attention unchanged. Nativebaseline has own trainedprompt.')
 (OUT/f'{phase}_comparison.json').write_text(json.dumps(output,indent=2))
 lines=[f'# Eval@20 {phase}: {len(identity)} fixedvalidation rows','','| Model | Method | R5 | R10 | R20 | N5 | N10 | N20 | Mean s | Full20 | Mean unique proposals |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for name,ms in results.items():
  for method,m in ms.items():lines.append(f"| {name} | {method} | {m['Recall@5']:.5f} | {m['Recall@10']:.5f} | {m['Recall@20']:.5f} | {m['NDCG@5']:.5f} | {m['NDCG@10']:.5f} | {m['NDCG@20']:.5f} | {m['mean_seconds']:.3f} | {m['fraction_with20']:.1%} | {format(m['mean_unique_proposals'], '.2f') if m['mean_unique_proposals'] is not None else '—'} |")
 lines+=['',output['scope']];(OUT/f'{phase}_comparison.md').write_text('\n'.join(lines)+'\n');return output
try:
 while pending or active:
  for gpu,(p,phase,name) in list(active.items()):
   if p.poll() is not None:
    if p.returncode:raise RuntimeError((phase,name,p.returncode))
    valid(phase,name);completed.add((phase,name));del active[gpu];print('completed',phase,name,flush=True)
  if selected is None and all(('screen',n) in completed for n in ['nonbalanced_vq','dense_2465']):
   ds={n:valid('screen',n) for n in ['nonbalanced_vq','dense_2465']}
   eligible=['single20','single50','two20_sum','four20_sum','four20_max','four20_marginal']
   best=max(eligible,key=lambda method:(sum(metrics(d['rows'][method])['NDCG@20'] for d in ds.values()),-sum(metrics(d['rows'][method])['mean_seconds'] for d in ds.values())))
   selected=list(dict.fromkeys(['single20','four20_sum',best]));pending.extend(('confirm',n) for n in ds)
   (OUT/'selection.json').write_text(json.dumps(dict(methods=selected,best_mean_NDCG20=best,criterion='Mean256NDCG20 across2latentmodels amongfull20recipes; nested64 replays only confirm latency.'),indent=2))
  for gpu in GPUS:
   if not pending or gpu in active:continue
   memory=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
   if memory>2048:continue
   phase,name=pending.pop(0);native=name.startswith('no_latent');module='latent.eval_search_matrix' if native else 'latent.eval_at20'
   argv=[PYTHON,'-u','-m',module,'--model',MODELS[name],'--data',DATA,'--index',INDEX,'--sample',str(REF/('sample256.json' if phase=='screen' else 'sample64.json')),'--output',str(OUT/f'{phase}_{name}.json'),'--mode','confirm' if native or phase=='confirm' else 'screen']
   if native:argv+=['--no-latent','--top-k','20','--methods','direct20','direct50']
   elif phase=='confirm':argv+=['--methods']+selected
   with (OUT/f'{phase}_{name}.log').open('w') as log:p=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false'))
   active[gpu]=(p,phase,name);records.append(dict(pid=p.pid,gpu=gpu,phase=phase,name=name,argv=argv));(OUT/'workers.json').write_text(json.dumps(records,indent=2));print('started',phase,name,gpu,p.pid,flush=True)
  if all(('screen',n) in completed for n in MODELS) and not (OUT/'screen_comparison.json').exists():report('screen')
  (OUT/'status.json').write_text(json.dumps(dict(phase='running',active={g:dict(pid=p.pid,phase=ph,name=n) for g,(p,ph,n) in active.items()},pending=pending,completed=sorted(completed),updated_at=time.time()),indent=2))
  if pending or active:time.sleep(10)
 report('screen');report('confirm');(OUT/'status.json').write_text(json.dumps(dict(phase='completed',methods=selected,updated_at=time.time()),indent=2));print('allcompleted',flush=True)
except Exception as e:
 (OUT/'status.json').write_text(json.dumps(dict(phase='failed',error=repr(e),updated_at=time.time())));raise
