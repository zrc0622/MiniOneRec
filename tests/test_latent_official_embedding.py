"""Last-token teacher vectors preserve targets and bind exported codes to new SID data."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from latent.core import index_hash, load_labels
from latent.prepare import embedding_pool, mean_pool, encode, merge, fit
from latent.vq import ParallelVQ
from tests.test_latent import fixture, data_fixture, INDEX


class OfficialInterestEmbeddingTests(unittest.TestCase):
    def test_history_only_vectors_and_codes_ignore_target_fields(self):
        torch.set_num_threads(1)
        tk,_,_,_=fixture()
        class CausalEncoder(torch.nn.Module):
            def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.ones(1))
            def forward(self,input_ids,attention_mask,use_cache=False):
                x=(input_ids.float()*attention_mask).cumsum(-1)
                return SimpleNamespace(last_hidden_state=torch.stack([x,x.square(),x*.5,x*0+1],-1))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);row,paths=data_fixture(root)
            for split,path in paths.items():
                with open(path,'w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(row)+['item_title']);writer.writeheader()
                    for i in range(4 if split=='train' else 2):
                        target=2+i%2
                        writer.writerow(dict(row,item_id=str(target),item_sid=''.join(INDEX[str(target)]),
                                             item_title=f'<a_{target}> target text changes {i}'))
            emb=root/'embeddings'
            args=SimpleNamespace(train=paths['train'],valid=paths['valid'],test=paths['test'],
                                 index=str(root/'index.json'),model='fixture',output=str(emb),
                                 shards=1,shard=0,batch=4,max_length=2048,dim=4,cpu=True,
                                 embedding_input='history',pooling='last_token',l2_normalize=True)
            with patch('latent.prepare.AutoTokenizer.from_pretrained',return_value=tk),patch(
                    'latent.prepare.AutoModel.from_pretrained',return_value=CausalEncoder()):encode(args)
            merge(SimpleNamespace(output=str(emb),shards=1))
            x=np.load(emb/'train.npy')
            np.testing.assert_array_equal(x,np.repeat(x[:1],4,axis=0))
            np.testing.assert_allclose(np.linalg.norm(x,axis=1),1,atol=1e-6)
            labels=root/'labels'
            fit(SimpleNamespace(output=str(labels),embeddings=str(emb),num_slots=3,
                                codebook_size=4,latent_dim=2,layers=[8],commitment=.25,
                                epochs=2,patience=2,batch=2,lr=.001,seed=42,cpu=True,balance_epsilon=0))
            for split,path in paths.items():
                codes,manifest=load_labels(labels,split,path,INDEX)
                np.testing.assert_array_equal(codes,np.repeat(codes[:1],len(codes),axis=0))
                self.assertFalse(manifest['embedding_recipe']['contains_target'])
                self.assertEqual(manifest['embedding_recipe']['embedding_input'],'history')
                self.assertEqual(manifest['vq_input_transform']['scale'],2.)

    def test_pooling_uses_last_valid_position_and_normalizes_after_dimension_selection(self):
        hidden = torch.tensor([[[1.,2.,99.],[3.,4.,99.],[500.,500.,500.]],
                               [[500.,500.,500.],[1.,2.,99.],[3.,4.,99.]]], dtype=torch.float16)
        mask = torch.tensor([[1,1,0],[0,1,1]])
        torch.testing.assert_close(embedding_pool(hidden,mask,2,'last_token',True),
                                   torch.tensor([[.6,.8],[.6,.8]]))
        torch.testing.assert_close(embedding_pool(hidden,mask,2),mean_pool(hidden,mask,2),rtol=0,atol=0)
        with self.assertRaises(ValueError):embedding_pool(hidden,mask,4,'last_token',True)

    def test_actual_encode_merge_fit_export_normalized_vectors_with_new_sid_binding(self):
        torch.set_num_threads(1)
        tk,_,_,_=fixture()
        class CausalEncoder(torch.nn.Module):
            def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.ones(1))
            def forward(self,input_ids,attention_mask,use_cache=False):
                x=(input_ids.float()*attention_mask).cumsum(-1)
                return SimpleNamespace(last_hidden_state=torch.stack([x,x.square(),x*.5,x*0+1],-1))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);row,paths=data_fixture(root)
            new_index=dict(INDEX);new_index['2'],new_index['3']=INDEX['3'],INDEX['2']
            (root/'index.json').write_text(json.dumps(new_index))
            for split,path in paths.items():
                with open(path,'w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(row)+['item_title']);writer.writeheader()
                    for i in range(4 if split=='train' else 2):
                        target=2+i%2
                        writer.writerow(dict(row,item_id=str(target),item_sid=''.join(new_index[str(target)]),
                                             item_title=f'<a_{target}>'))
            embeddings=root/'embeddings'
            args=SimpleNamespace(train=paths['train'],valid=paths['valid'],test=paths['test'],
                                 index=str(root/'index.json'),model='fixture',output=str(embeddings),
                                 shards=1,shard=0,batch=4,max_length=2048,dim=4,cpu=True,
                                 embedding_input='history_target',pooling='last_token',l2_normalize=True)
            with patch('latent.prepare.AutoTokenizer.from_pretrained',return_value=tk),patch(
                    'latent.prepare.AutoModel.from_pretrained',return_value=CausalEncoder()):encode(args)
            merge(SimpleNamespace(output=str(embeddings),shards=1))
            vector=np.load(embeddings/'train.npy')
            np.testing.assert_allclose(np.linalg.norm(vector,axis=1),1,atol=1e-6)
            self.assertFalse(np.array_equal(vector[0],vector[1]))
            np.testing.assert_array_equal(vector[0],vector[2])
            manifest=json.loads((embeddings/'embeddings.json').read_text())
            self.assertEqual(manifest['recipe']['pooling'],'last_valid_token')
            self.assertEqual(manifest['recipe']['normalization'],'l2')
            self.assertTrue(manifest['recipe']['contains_target'])
            self.assertEqual(manifest['sid_sha256'],index_hash(new_index))
            labels=root/'labels'
            embedding_hashes={s:manifest['embedding_hashes'][s] for s in paths}
            fit(SimpleNamespace(output=str(labels),embeddings=str(embeddings),num_slots=3,
                                codebook_size=4,latent_dim=2,layers=[8],commitment=.25,
                                epochs=2,patience=2,batch=2,lr=.001,seed=42,cpu=True,balance_epsilon=0))
            saved=torch.load(labels/'vq.pt',weights_only=True)
            self.assertEqual(saved['options']['input_scale'],2.)
            restored=ParallelVQ(**saved['options']);restored.load_state_dict(saved['state'])
            raw_options=dict(saved['options']);raw_options.pop('input_scale')
            manual=ParallelVQ(**raw_options);manual.load_state_dict(saved['state'])
            for split,path in paths.items():
                codes,metadata=load_labels(labels,split,path,new_index)
                self.assertEqual(codes.shape,(4 if split=='train' else 2,3))
                self.assertEqual(metadata['fit_split'],'train')
                self.assertEqual(metadata['selection_split'],'valid')
                self.assertEqual(metadata['assignment_recipe']['training'],'nearest')
                self.assertEqual(metadata['embedding_recipe']['inference_input'],'history_only')
                self.assertEqual(metadata['vq_input_transform']['kind'],'sqrt_dim')
                self.assertEqual(metadata['vq_input_transform']['scale'],2.)
                self.assertEqual(metadata['validation_reconstruction_source_scale'],
                                 metadata['validation_reconstruction']/4)
                a=torch.from_numpy(np.load(embeddings/f'{split}.npy'))
                torch.testing.assert_close(restored.quantize(a)[2],torch.from_numpy(codes))
                torch.testing.assert_close(restored.quantize(a)[2],manual.quantize(a*2)[2])
                from latent.core import sha
                self.assertEqual(embedding_hashes[split],sha(embeddings/f'{split}.npy'))
            metrics=json.loads((labels/'training_metrics.json').read_text())
            self.assertEqual(len(metrics),2)
            self.assertEqual(len(metrics[0]['valid_code_statistics']),3)
            with self.assertRaises(ValueError):load_labels(labels,'train',paths['train'],INDEX)

    def test_scaled_model_matches_explicit_scaling_in_initialization_loss_and_reload(self):
        torch.manual_seed(123)
        x=torch.nn.functional.normalize(torch.randn(12,8),dim=1)
        scaled=ParallelVQ(8,4,2,layers=[12,8],input_scale=8**.5)
        plain=ParallelVQ(8,4,2,layers=[12,8]);plain.load_state_dict(scaled.state_dict())
        scaled.initialize_from_train(x,torch.Generator().manual_seed(42))
        plain.initialize_from_train(x*(8**.5),torch.Generator().manual_seed(42))
        torch.testing.assert_close(scaled.codebooks,plain.codebooks,rtol=0,atol=0)
        a,parts,codes=scaled(x);b,other,expected=plain(x*(8**.5))
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        for name in parts:torch.testing.assert_close(parts[name],other[name],rtol=0,atol=0)
        torch.testing.assert_close(codes,expected)
        a.backward();b.backward()
        for p,q in zip(scaled.parameters(),plain.parameters()):torch.testing.assert_close(p.grad,q.grad)
        self.assertNotIn('input_scale',plain.options)
        for invalid in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):ParallelVQ(8,4,2,layers=[8],input_scale=invalid)


if __name__=='__main__':unittest.main()
