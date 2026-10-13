"""Resume completed normalized embeddings through the coordinator and export labels."""
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_target_conditioned_vq as runner
from latent.core import index_hash, sha
from latent.prepare import fit
from tests.test_latent import data_fixture, INDEX


class TargetVQResumeTests(unittest.TestCase):
    def test_resume_starts_only_vq_after_normalization_checks_and_exports_bound_labels(self):
        self.check_completed_embeddings(reuse=False)

    def test_reuses_external_completed_embeddings_without_encoding_or_writing_source(self):
        self.check_completed_embeddings(reuse=True)

    def test_history_only_reuse_on_explicitly_shared_gpu_exports_history_bound_labels(self):
        self.check_completed_embeddings(reuse=True,history=True,shared=True)

    def test_history_only_refuses_target_teacher_provenance(self):
        for recipe in [dict(contains_target=True,text='chronological_numbered_titles_v1'),
                       dict(contains_target=False,embedding_input='history_target',text='chronological_numbered_titles_v1'),
                       dict(contains_target=False,text='chronological_numbered_titles_then_next_title_v1')]:
            with self.assertRaises(ValueError):runner.validate_embedding_recipe(recipe,'history')

    def test_shared_gpu_requires_memory_headroom_and_low_utilization(self):
        info=dict(total_mib=46068,used_mib=7122,utilization=0,pids=[123])
        self.assertFalse(runner.gpu_ready(info))
        self.assertTrue(runner.gpu_ready(info,True))
        self.assertFalse(runner.gpu_ready(dict(info,used_mib=30000),True))
        self.assertFalse(runner.gpu_ready(dict(info,utilization=80),True))

    def check_completed_embeddings(self, reuse, history=False, shared=False):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);dataset=root/'data';dataset.mkdir()
            _,paths=data_fixture(dataset)
            category=dataset/'Industrial_and_Scientific';category.mkdir()
            (dataset/'index.json').rename(category/'Industrial_and_Scientific.index.json')
            for split in ('train','valid','test'):
                folder=dataset/split;folder.mkdir()
                old=Path(paths[split]);new=folder/f'{runner.STEM}.csv';old.rename(new);paths[split]=str(new)
            output=root/'output';output.mkdir()
            embeddings=root/'external' if reuse else output/'embeddings';embeddings.mkdir(parents=True)
            (output/'source_snapshot').mkdir()
            counts=dict(train=4,valid=2,test=2)
            for split,count in counts.items():np.save(embeddings/f'{split}.npy',np.full((count,4),.5,dtype=np.float32))
            manifest=dict(source_hashes={s:sha(p) for s,p in paths.items()},sid_sha256=index_hash(INDEX),
                          recipe=dict(dim=4,pooling='last_valid_token',normalization='l2',contains_target=True,
                                      intended_use='latent_supervision_only',inference_input='history_only'),
                          counts=counts,embedding_hashes={s:sha(embeddings/f'{s}.npy') for s in counts})
            if history:manifest['recipe'].update(contains_target=False,embedding_input='history',text='chronological_numbered_titles_v1')
            (embeddings/'embeddings.json').write_text(json.dumps(manifest))
            if not reuse:(embeddings/'shard0.json').write_text('{}')
            before={p.name:sha(p) for p in embeddings.iterdir()}
            calls=[]
            def child(command,**kwargs):
                calls.append(command)
                self.assertEqual(command[command.index('latent.prepare')+1],'vq')
                self.assertEqual(command[command.index('--balance_epsilon')+1],'0')
                self.assertEqual(command[command.index('--input_scaling')+1],'sqrt_dim')
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'6')
                fit(SimpleNamespace(output=str(output/'labels'),embeddings=str(embeddings),num_slots=3,
                                    codebook_size=4,latent_dim=2,layers=[8],commitment=.25,
                                    epochs=2,patience=2,batch=2,lr=.001,seed=42,cpu=True,balance_epsilon=0))
                return SimpleNamespace(pid=54321,wait=lambda:0,poll=lambda:0)
            argv=['run_target_conditioned_vq.py','--root',str(output),'--gpus','6','--batch','64',
                  '--data',str(dataset),'--pooling','last_token','--l2-normalize']
            if reuse:argv.extend(['--reuse-embeddings',str(embeddings)])
            if history:argv.extend(['--embedding-input','history'])
            if shared:argv.append('--allow-shared-gpu')
            with patch('sys.argv',argv),patch.object(runner,'gpu_snapshot',return_value={'6':dict(
                    total_mib=46068,used_mib=7122 if shared else 0,utilization=0,pids=[123] if shared else [])}),patch.object(runner.time,'sleep'),patch.object(
                    runner.subprocess,'Popen',side_effect=child),contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            self.assertEqual(len(calls),1)
            self.assertEqual(json.loads((output/'pipeline_status.json').read_text())['phase'],'complete_vq_only')
            self.assertFalse((output/'sft').exists())
            labels=json.loads((output/'labels/labels.json').read_text())
            self.assertEqual(labels['embedding_recipe']['contains_target'],not history)
            self.assertTrue((output/'embedding_norms.json').exists())
            self.assertEqual(before,{p.name:sha(p) for p in embeddings.iterdir()})


if __name__=='__main__':unittest.main()
