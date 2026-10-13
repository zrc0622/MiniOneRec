"""Two-slot export/protocol/loss and cached inference integration."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np
import torch
from transformers import TrainingArguments
from tests.test_latent import fixture, data_fixture, INDEX
from latent.core import load_labels, sha, index_hash, write_json, history_prompt, validate_protocol
from latent.prepare import fit
from latent.vq import ParallelVQ
from latent.data import supervised_example
from latent.trainer import LatentSFTTrainer, InterestCollator
from latent.eval_unified_small import batch_latent
from latent.eval_at20 import infer

class TwoSlotTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_quantizer_export_reload_and_target_independence(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,paths=data_fixture(root);source=root/'emb';source.mkdir()
            rng=np.random.default_rng(42)
            arrays={s:rng.normal(size=(4 if s=='train' else 2,8)).astype('float32') for s in paths}
            def export_embeddings():
                for s,a in arrays.items():np.save(source/f'{s}.npy',a)
                write_json(source/'embeddings.json',dict(source_hashes={s:sha(p) for s,p in paths.items()},sid_sha256=index_hash(INDEX),counts={s:len(a) for s,a in arrays.items()},recipe=dict(dim=8),embedding_hashes={s:sha(source/f'{s}.npy') for s in paths}))
            opts=dict(embeddings=str(source),num_slots=2,codebook_size=4,latent_dim=2,layers=[12,8],commitment=.25,epochs=2,patience=2,batch=2,lr=.001,seed=42,cpu=True)
            export_embeddings();fit(SimpleNamespace(output=str(root/'labels'),**opts))
            saved=torch.load(root/'labels/vq.pt',weights_only=True);model=ParallelVQ(**saved['options']);model.load_state_dict(saved['state'])
            codes,manifest=load_labels(root/'labels','valid',paths['valid'],INDEX)
            self.assertEqual(codes.shape,(2,2));self.assertEqual(manifest['k'],2)
            torch.testing.assert_close(model.quantize(torch.from_numpy(arrays['valid']))[2],torch.from_numpy(codes))
            arrays['test']*=100;export_embeddings();fit(SimpleNamespace(output=str(root/'labels2'),**opts))
            other=torch.load(root/'labels2/vq.pt',weights_only=True)
            for name,value in saved['state'].items():torch.testing.assert_close(value,other['state'][name],rtol=0,atol=0)

    def test_two_token_mask_loss_and_sid_generation(self):
        tk,model,p,g=fixture(k=2)
        self.assertEqual(validate_protocol(model.config,tk,INDEX)['k'],2)
        row=dict(history_item_sid=str([''.join(INDEX['0'])]))
        record=dict(prompt=history_prompt(row,2),target=''.join(INDEX['2']))
        self.assertIn('two latent',record['prompt'])
        ex=supervised_example(record,[0,1],tk,p['tokens']);self.assertEqual(sum(ex['interest_mask']),2)
        batch=InterestCollator(tk)([ex])
        with tempfile.TemporaryDirectory() as temp:
            trainer=LatentSFTTrainer(model=model,args=TrainingArguments(output_dir=temp,use_cpu=True,report_to=[]),latent_weight=1)
            loss,out=trainer.compute_loss(model,batch,return_outputs=True)
            labels=batch['labels'][:,1:];mask=batch['interest_mask'][:,1:].bool();valid=labels!=-100
            ce=torch.nn.functional.cross_entropy(out.logits[:,:-1].float().transpose(1,2),labels,ignore_index=-100,reduction='none')
            expected=ce[mask].mean()+ce[valid & ~mask].mean()
            torch.testing.assert_close(loss,expected);loss.backward()
            self.assertGreater(model.get_input_embeddings().weight.grad[g.slots[0]].abs().sum().item(),0)
        model.eval();prompts=[tk.encode('history <a_0>'),tk.encode('history <a_3> <b_0> history')]
        for method in ['single20','four20_marginal']:
            result=batch_latent(model,g,prompts,method)
            for prompt,r in zip(prompts,result):
                expected=infer(model,tk,g,dict(prompt=tk.decode(prompt,skip_special_tokens=False)),method)
                self.assertEqual(r['predict'],expected);self.assertEqual(len(r['predict']),20)
                self.assertEqual(len(set(r['predict'])),20)

if __name__=='__main__':unittest.main()
