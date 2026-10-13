"""Target-conditioned teacher embeddings preserve targets and stay out of prompts."""
import ast
import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from tests.test_latent import fixture, data_fixture, INDEX
from latent.core import bounded_history_target, history_key, history_target_key, history_prompt
from latent.prepare import embedding_example, encode, merge, fit


class TargetEmbeddingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tk, _, _, _ = fixture()

    def test_target_changes_teacher_key_but_not_inference_prompt(self):
        row=dict(history_item_title="['history', 'history <a_0>']",history_item_sid=str([''.join(INDEX['0'])]),item_id='2',item_title='<a_2>')
        other=dict(row,item_id='3',item_title='<a_3>')
        self.assertEqual(history_key(row),history_key(other))
        self.assertNotEqual(history_target_key(row),history_target_key(other))
        self.assertEqual(history_prompt(row),history_prompt(other))
        a=embedding_example(row,self.tk,2048,'history_target')
        b=embedding_example(other,self.tk,2048,'history_target')
        self.assertNotEqual(a[1],b[1])
        self.assertEqual(embedding_example(row,self.tk,2048,'history'),embedding_example(other,self.tk,2048,'history'))

    def test_truncation_retains_complete_target_and_recent_history(self):
        short=self.tk.encode(' '.join(['history']*8)+' <a_9>',add_special_tokens=False)
        target='<a_9> <b_0> <c_0>'
        row=dict(history_item_title="['history history history', 'history <a_2>']",item_title=target,item_id='9')
        _,both,_=embedding_example(row,self.tk,2048,'history_target')
        limit=len(both)-2
        ids,kept=bounded_history_target(ast.literal_eval(row['history_item_title']),target,self.tk,limit)
        self.assertEqual(kept,1);self.assertLessEqual(len(ids),limit)
        text=self.tk.decode(ids)
        for token in target.split():self.assertIn(token,text)
        self.assertIn('<a_2>',text)
        with self.assertRaises(ValueError):bounded_history_target(['history'],' ',self.tk,limit)
        with self.assertRaises(ValueError):bounded_history_target(['history'],target,self.tk,1)

    def test_actual_encoder_dedup_merge_and_train_only_vq(self):
        # A deterministic token-dependent stand-in isolates data plumbing.
        class FakeEncoder(torch.nn.Module):
            def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.ones(1))
            def forward(self,input_ids,attention_mask,use_cache=False):
                self.assert_no_cache=not use_cache
                x=input_ids.float()
                return SimpleNamespace(last_hidden_state=torch.stack([x,x.square(),x*.5,x*0+1],-1))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);row,paths=data_fixture(root)
            # Same history, different target; target2 is repeated nonadjacently.
            for split,path in paths.items():
                n=4 if split=='train' else 2
                with open(path,'w') as f:
                    keys=list(row)+['item_title'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
                    for i in range(n):
                        target=2+i%2
                        w.writerow(dict(row,item_id=str(target),item_sid=''.join(INDEX[str(target)]),item_title=f'<a_{target}>'))
            source=root/'emb';encoder=FakeEncoder()
            opts=dict(train=paths['train'],valid=paths['valid'],test=paths['test'],index=str(root/'index.json'),model='fixture',output=str(source),shards=2,batch=4,max_length=2048,dim=4,cpu=True,embedding_input='history_target')
            with patch('latent.prepare.AutoTokenizer.from_pretrained',return_value=self.tk),patch('latent.prepare.AutoModel.from_pretrained',return_value=encoder):
                for shard in [0,1]:encode(SimpleNamespace(shard=shard,**opts))
            merge(SimpleNamespace(output=str(source),shards=2))
            emb=np.load(source/'train.npy')
            self.assertFalse(np.array_equal(emb[0],emb[1]));np.testing.assert_array_equal(emb[0],emb[2]);np.testing.assert_array_equal(emb[1],emb[3])
            # One unsharded batch contains both targets: catches history-only dedup.
            one=root/'one';opts.update(output=str(one),shards=1)
            with patch('latent.prepare.AutoTokenizer.from_pretrained',return_value=self.tk),patch('latent.prepare.AutoModel.from_pretrained',return_value=encoder):encode(SimpleNamespace(shard=0,**opts))
            merge(SimpleNamespace(output=str(one),shards=1))
            np.testing.assert_array_equal(np.load(one/'train.npy'),emb)
            manifest=json.loads((source/'embeddings.json').read_text());self.assertTrue(manifest['recipe']['contains_target']);self.assertEqual(manifest['recipe']['inference_input'],'history_only')
            fit(SimpleNamespace(output=str(root/'labels'),embeddings=str(source),num_slots=3,codebook_size=4,latent_dim=2,layers=[8],commitment=.25,epochs=2,patience=2,batch=2,lr=.001,seed=42,cpu=True))
            labels=json.loads((root/'labels/labels.json').read_text());self.assertEqual(labels['fit_split'],'train');self.assertTrue(labels['embedding_recipe']['contains_target']);self.assertEqual(np.load(root/'labels/train_codes.npy').shape,(4,3))

if __name__=='__main__':unittest.main()
