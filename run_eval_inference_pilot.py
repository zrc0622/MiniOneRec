"""Durable bounded pilot coordinator: exactly one worker per GPU4/6."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path('/data/zuorongchang/project/MiniOneRec')
OUT=ROOT/'outputs/eval_inference_pilot_20261009'
PYTHON='/data/zuorongchang/miniconda3/envs/minionerec/bin/python'
os.chdir(ROOT)
workers=json.loads((OUT/'workers.json').read_text())
def active(pid,module):
    try:return module.encode() in (Path('/proc')/str(pid)/'cmdline').read_bytes()
    except OSError:return False
while any(active(w['pid'],'latent.eval_pilot') for w in workers):time.sleep(5)
subprocess.run([PYTHON,'-m','latent.eval_pilot_report','--root',str(OUT)],check=True)
cmd=[PYTHON,'-u','-m','latent.eval_fast_pilot','--model','outputs/latent_qwen3_0.6b_balanced_vq15/sft/final_checkpoint','--data','data/Amazon23/variants/rqkmeans_balanced/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv','--index','data/Amazon23/variants/rqkmeans_balanced/Industrial_and_Scientific/Industrial_and_Scientific.index.json','--sample',str(OUT/'sample256.json'),'--shards','2']
children=[];info=[]
for shard,gpu in enumerate([4,6]):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
    with (OUT/f'fast_shard{shard}.log').open('w') as log:
        child=subprocess.Popen(cmd+['--shard',str(shard),'--output',str(OUT/f'fast_shard{shard}.json')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    children.append(child);info.append(dict(pid=child.pid,gpu=gpu,shard=shard))
(OUT/'fast_workers.json').write_text(json.dumps(info,indent=2)+'\n')
print('fast workers',info,flush=True)
codes=[c.wait() for c in children]
assert codes==[0,0],codes
subprocess.run([PYTHON,'-m','latent.eval_fast_pilot_report','--root',str(OUT)],check=True)
(OUT/'completed.json').write_text(json.dumps(dict(exit_codes=codes,completed_at=time.time()))+'\n')
