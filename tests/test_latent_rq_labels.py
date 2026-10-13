import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from latent.core import index_hash, load_labels, sha, write_json
from latent.data import sft_datasets
from latent.prepare_rq import capacities, fit, quantize
from tests.test_latent import data_fixture, fixture, INDEX


class RQLabelsTests(unittest.TestCase):
    def test_capacities_and_batch_independent_residual_encoding(self):
        low, high = capacities(129296, 256)
        self.assertEqual((low, high), (504, 506))
        with self.assertRaises(ValueError):
            capacities(3, 4)
        torch.manual_seed(42)
        x, centers = torch.randn(19, 8), torch.randn(3, 4, 8)
        codes, residual = quantize(x, centers)
        pieces = torch.cat([quantize(chunk, centers)[0] for chunk in x.split(3)])
        torch.testing.assert_close(codes, pieces)
        reconstructed = sum(centers[k, codes[:, k]] for k in range(3))
        torch.testing.assert_close(x - residual, reconstructed, atol=1e-6, rtol=1e-6)

    def test_train_only_fitting_and_generic_label_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, paths = data_fixture(root)
            source = root / 'embeddings'
            source.mkdir()
            rng = np.random.default_rng(42)
            arrays = dict(train=rng.normal(size=(4, 8)).astype('float32'),
                          valid=rng.normal(size=(2, 8)).astype('float32'),
                          test=rng.normal(size=(2, 8)).astype('float32'))

            def prepare():
                for split, a in arrays.items():
                    np.save(source / f'{split}.npy', a)
                write_json(source / 'embeddings.json', dict(
                    source_hashes={s: sha(p) for s, p in paths.items()}, sid_sha256=index_hash(INDEX),
                    counts={s: len(a) for s, a in arrays.items()}, recipe=dict(dim=8),
                    embedding_hashes={s: sha(source / f'{s}.npy') for s in arrays}))

            prepare()
            options = dict(embeddings=str(source), codebook_size=2, max_iter=3, tol=1e-7,
                           n_init=1, jobs=1, threads=1, batch=2, seed=42, cpu=True)
            fit(SimpleNamespace(output=str(root / 'a'), **options))
            arrays['test'] *= 100
            prepare()
            fit(SimpleNamespace(output=str(root / 'b'), **options))
            with np.load(root / 'a/rq_kmeans.npz') as a, np.load(root / 'b/rq_kmeans.npz') as b:
                np.testing.assert_array_equal(a['codebooks'], b['codebooks'])
            codes, manifest = load_labels(root / 'a', 'train', paths['train'], INDEX)
            self.assertEqual(codes.shape, (4, 3))
            self.assertEqual(manifest['tokenizer_type'], 'balanced_residual_kmeans')
            # Exercise the actual SFT data path, not only label-file loading.
            tokenizer, model, protocol, _ = fixture(size=2)
            protocol['label_model_sha256'] = manifest['label_model_sha256']
            train, valid = sft_datasets(paths['train'], paths['valid'], str(root / 'items.json'),
                str(root / 'index.json'), INDEX, tokenizer, protocol, str(root / 'a'))
            self.assertEqual(len(valid), 2)
            self.assertTrue(all(sum(row['interest_mask']) == 3 for row in train[:4]))
            from datasets import Dataset
            from transformers import TrainingArguments
            from latent.trainer import LatentSFTTrainer, InterestCollator
            trainer = LatentSFTTrainer(model=model,
                args=TrainingArguments(output_dir=str(root / 'sft'), use_cpu=True, max_steps=1,
                    per_device_train_batch_size=4, report_to=[], save_strategy='no',
                    remove_unused_columns=False, prediction_loss_only=True, disable_tqdm=True),
                processing_class=tokenizer, data_collator=InterestCollator(tokenizer),
                train_dataset=Dataset.from_list(train[:4]), eval_dataset=Dataset.from_list(valid))
            trainer.train()
            self.assertTrue(np.isfinite(trainer.evaluate()['eval_loss']))
            with (root / 'a/rq_kmeans.npz').open('ab') as f:
                f.write(b'tamper')
            with self.assertRaises(ValueError):
                load_labels(root / 'a', 'train', paths['train'], INDEX)


if __name__ == '__main__':
    unittest.main()
