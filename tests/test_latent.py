"""Small real-model checks for the latent experiment; no network/CUDA needed."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import GenerationConfig, PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM, TrainingArguments

from interest.train import InterestRLArguments
from latent.core import bounded_history, history_key, history_records, index_hash, initialize, load_labels, sha, validate_protocol, write_json
from latent.data import rl_records, sft_datasets, supervised_example
from latent.evaluate import merge
from latent.generation import Grammar, constrained_logps, full_logps, generate, pack_trajectories, rollout
from latent.objective import advantages, ranking_rewards
from latent.prepare import fit, mean_pool
from latent.trainer import InterestCollator, LatentGRPOTrainer, LatentSFTTrainer
from latent.vq import ParallelVQ

INDEX = {str(i): [f'<a_{i}>', '<b_0>', '<c_0>'] + (['<d_0>'] if i % 2 else []) for i in range(20)}


def fixture(size=4):
    torch.manual_seed(42)
    tokens = ['<unk>', '<eos>', 'history']
    backend = Tokenizer(WordLevel({t: i for i, t in enumerate(tokens)}, unk_token='<unk>'))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='<unk>', eos_token='<eos>', pad_token='<eos>')
    tokenizer.padding_side = 'left'
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tokenizer), hidden_size=24, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=12,
        max_position_embeddings=1024, eos_token_id=1, pad_token_id=1, tie_word_embeddings=False))
    protocol = initialize(model, tokenizer, INDEX, dict(codebook_size=size, vq_sha256='fixture'))
    return tokenizer, model, protocol, Grammar(tokenizer, INDEX, protocol)


def data_fixture(root):
    row = dict(history_item_id="['0', '1']", history_item_sid=str([''.join(INDEX['0']), ''.join(INDEX['1'])]),
               history_item_title="['History zero', 'History one']", item_id='2', item_sid=''.join(INDEX['2']))
    paths = {}
    for split in ('train', 'valid', 'test'):
        path = root / f'{split}.csv'
        with path.open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            for _ in range(4 if split == 'train' else 2):
                w.writerow(row)
        paths[split] = str(path)
    write_json(root / 'index.json', INDEX)
    write_json(root / 'items.json', {k: dict(title=f'item {k}', description=f'description {k}') for k in INDEX})
    return row, paths


class LatentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_history_only_and_pooling(self):
        row = dict(history_item_title="['one', 'two']", item_id='2', item_title='target')
        changed = dict(row, item_id='9', item_title='different target')
        self.assertEqual(history_key(row), history_key(changed))
        tk, _, _, _ = fixture()
        _, n = bounded_history(['old ' * 200, 'new'], tk, 30)
        self.assertEqual(n, 1)
        with self.assertRaises(ValueError):
            bounded_history(['long ' * 100], tk, 10)
        hidden = torch.tensor([[[2., 4., 9.], [4., 8., 3.], [900., 900., 900.]]])
        torch.testing.assert_close(mean_pool(hidden, torch.tensor([[1, 1, 0]]), 2), torch.tensor([[3., 6.]]))
        # FP16 sum over 2048 activations would overflow without FP32 accumulation.
        torch.testing.assert_close(mean_pool(torch.full((1, 2048, 2), 100., dtype=torch.float16),
                                            torch.ones(1, 2048), 2), torch.full((1, 2), 100.))

    def test_parallel_vq_and_train_only_fit(self):
        model = ParallelVQ(8, 4, 2, layers=[12, 8])
        x = torch.randn(16, 8)
        loss, _, codes = model(x)
        loss.backward()
        self.assertEqual(codes.shape, (16, 3))
        self.assertGreater(float(model.codebooks.grad.abs().sum()), 0)
        # Altering one codebook cannot affect the other branch assignments.
        before = codes.clone()
        with torch.no_grad():
            model.codebooks[0].add_(10)
        torch.testing.assert_close(model.quantize(x)[2][:, 1:], before[:, 1:])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, paths = data_fixture(root)
            source = root / 'emb'; source.mkdir()
            arrays = {'train': np.random.default_rng(42).normal(size=(4, 8)).astype('float32'),
                      'valid': np.zeros((2, 8), dtype='float32'), 'test': np.ones((2, 8), dtype='float32')}
            def prep():
                for split, a in arrays.items(): np.save(source / f'{split}.npy', a)
                write_json(source / 'embeddings.json', dict(source_hashes={s: sha(p) for s, p in paths.items()},
                    sid_sha256=index_hash(INDEX), counts={s: len(a) for s, a in arrays.items()},
                    recipe=dict(dim=8), embedding_hashes={s: sha(source / f'{s}.npy') for s in arrays}))
            prep()
            opts = dict(embeddings=str(source), codebook_size=4, latent_dim=2, layers=[12, 8], commitment=.25,
                        epochs=2, patience=2, batch=2, lr=1e-3, seed=42, cpu=True)
            fit(SimpleNamespace(output=str(root / 'labels'), **opts))
            a = torch.load(root / 'labels/vq.pt', weights_only=True)
            restored = ParallelVQ(**a['options'])
            restored.load_state_dict(a['state'])
            self.assertEqual(restored.options['layers'], opts['layers'])
            arrays['test'] *= 50; prep()
            fit(SimpleNamespace(output=str(root / 'labels2'), **opts))
            b = torch.load(root / 'labels2/vq.pt', weights_only=True)
            for key in a['state']: torch.testing.assert_close(a['state'][key], b['state'][key], rtol=0, atol=0)
            codes, _ = load_labels(root / 'labels', 'train', paths['train'], INDEX)
            self.assertEqual(codes.shape, (4, 3))
            Path(paths['train']).write_text(Path(paths['train']).read_text() + '\n')
            with self.assertRaises(ValueError): load_labels(root / 'labels', 'train', paths['train'], INDEX)

    def test_full_vq_matches_repository_mlp(self):
        from rq.models.rqvae import RQVAE
        from rq.models.layers import MLPLayers
        model = ParallelVQ()
        widths = [2560, 2048, 1024, 512, 256, 128, 96]
        self.assertIsInstance(model.encoder, MLPLayers)
        self.assertEqual(model.encoder.layers, widths)
        self.assertEqual(model.decoder.layers, widths[::-1])
        self.assertEqual(model.codebooks.shape, (3, 256, 32))
        reference = RQVAE(in_dim=2560, num_emb_list=[256] * 3, e_dim=96,
                          layers=widths[1:-1], sk_epsilons=[0.] * 3)
        for name in ('encoder', 'decoder'):
            actual, expected = getattr(model, name), getattr(reference, name)
            actual.load_state_dict(expected.state_dict())
            x = torch.randn(2, actual.layers[0])
            torch.testing.assert_close(actual(x), expected(x))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 16_116_064)
        x = torch.randn(2, 2560)
        z, q, _ = model.quantize(x)
        self.assertEqual(z.shape, (2, 3, 32))
        self.assertEqual(q.shape, (2, 3, 32))
        loss, _, codes = model(x)
        loss.backward()
        self.assertEqual(codes.shape, (2, 3))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_original_ranking_and_logprob(self):
        hits = torch.zeros(16); hits[3] = hits[9] = 1
        disc = [-1 / np.log2(i + 2) for i in range(16)]
        expected = torch.tensor([1 if h else -d / sum(disc) for h, d in zip(hits, disc)], dtype=torch.float32)
        torch.testing.assert_close(ranking_rewards(hits), expected)
        ia, ya, r = advantages(hits, 'original')
        torch.testing.assert_close(ya, (expected - expected.mean()) / (expected.std() + 1e-4))
        torch.testing.assert_close(ranking_rewards(torch.zeros(16)), torch.zeros(16))
        tk, model, p, grammar = fixture()
        samples = rollout(model, [2], grammar, 'original')
        self.assertEqual(len(samples), 16)
        packed = pack_trajectories([[2]] * 16, samples, [3] * 16, [False] * 16, grammar, 'cpu')
        actual = full_logps(model, packed, temperature=2)
        logits = model(input_ids=packed['input_ids'], attention_mask=packed['attention_mask']).logits[:, :-1]
        direct = logits.float().log_softmax(-1).gather(-1, packed['completion_ids'][..., None]).squeeze(-1)
        torch.testing.assert_close(actual, direct)
        self.assertFalse(torch.allclose(actual, constrained_logps(model, packed)))
        full_logps(model, packed).sum().backward()
        self.assertGreater(float(model.get_input_embeddings().weight.grad[grammar.slots[0]].abs().sum()), 0)

    def test_original_beam_sampling_parity(self):
        _, model, _, grammar = fixture()
        model.eval()
        # Match the original ReReTrainer config and its log_softmax+mask
        # processor. Only the allowed-prefix function changes to latent grammar.
        from transformers import LogitsProcessor
        class OriginalProcessor(LogitsProcessor):
            def __call__(self, ids, scores):
                scores = scores.log_softmax(-1)
                mask = torch.full_like(scores, -torch.inf)
                for i, seq in enumerate(ids):
                    mask[i, grammar.allowed(seq[1:].tolist())] = 0
                return scores + mask
        cfg = GenerationConfig(max_new_tokens=128, length_penalty=0., num_beams=16,
            num_return_sequences=16, pad_token_id=grammar.eos, eos_token_id=grammar.eos,
            top_k=None, top_p=None, temperature=1., do_sample=True)
        torch.manual_seed(27)
        expected = model.generate(torch.tensor([[2]]), attention_mask=torch.ones(1, 1, dtype=torch.long),
                                  generation_config=cfg, logits_processor=[OriginalProcessor()])[:, 1:].tolist()
        expected = [s[:s.index(grammar.eos) + 1] for s in expected]
        torch.manual_seed(27)
        actual = rollout(model, [2], grammar, 'original')
        self.assertEqual(expected, actual)

    def test_grammar_hierarchical_and_roundtrip(self):
        tk, model, p, grammar = fixture()
        self.assertIn(grammar.slots[1][0], grammar.allowed([grammar.slots[0][0]]))
        samples = rollout(model, [2], grammar, '4x4')
        self.assertEqual(len(samples), 16)
        for i in range(0, 16, 4): self.assertTrue(all(s[:3] == samples[i][:3] for s in samples[i:i + 4]))
        for s in samples: self.assertIn(grammar.item(s), [''.join(x) for x in INDEX.values()])
        for aux in ('identification', 'title'):
            s = rollout(model, [2], grammar, 'original', aux)
            self.assertEqual(len(s), 16)
            for x in s: grammar.item(x, aux)
        hits = torch.zeros(16); hits[0] = hits[8] = 1
        ia, ya, reward = advantages(hits, '4x4')
        torch.testing.assert_close(reward, torch.tensor([1., 0, 1, 0]))
        self.assertTrue((ya[4:8] == 0).all())
        with tempfile.TemporaryDirectory() as path:
            model.save_pretrained(path); tk.save_pretrained(path)
            model2 = Qwen3ForCausalLM.from_pretrained(path)
            validate_protocol(model2.config, tk, INDEX)
            self.assertEqual(model2.config.latent_protocol, p)
            bad = copy.deepcopy(INDEX); bad['0'][0] = '<a_99>'
            with self.assertRaises(ValueError): validate_protocol(model2.config, tk, bad)

    def test_task_mixture_and_tiny_training(self):
        from data import SidItemFeatDataset, FusionSeqRecDataset, RLTitle2SidDataset, RLSeqTitle2SidDataset
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row, paths = data_fixture(root)
            tk, model, p, grammar = fixture()
            labels = root / 'labels'; labels.mkdir()
            (labels / 'vq.pt').write_bytes(b'fixture')
            hashes = {}
            for split, path in paths.items():
                np.save(labels / f'{split}_codes.npy', np.zeros((4 if split == 'train' else 2, 3), dtype=np.int64))
                hashes[split] = sha(labels / f'{split}_codes.npy')
            manifest = dict(version=1, k=3, sid_sha256=index_hash(INDEX), codebook_size=4, vq_sha256=sha(labels / 'vq.pt'),
                            source_hashes={s: sha(path) for s, path in paths.items()}, label_hashes=hashes)
            write_json(labels / 'labels.json', manifest)
            p['label_model_sha256'] = manifest['vq_sha256']
            train, valid = sft_datasets(paths['train'], paths['valid'], str(root / 'items.json'), str(root / 'index.json'), INDEX, tk, p, str(labels))
            expected = list(SidItemFeatDataset(str(root / 'items.json'), str(root / 'index.json'), tk, max_len=512, seed=42))
            expected += list(FusionSeqRecDataset(paths['train'], str(root / 'items.json'), str(root / 'index.json'), tk, max_len=512, seed=42))
            self.assertEqual([{k: v for k, v in x.items() if k != 'interest_mask'} for x in train[4:]], expected)
            self.assertEqual(len(train), 4 + len(expected))
            records = rl_records(paths['train'], str(root / 'items.json'), str(root / 'index.json'), INDEX, 2)
            aux = list(RLTitle2SidDataset(str(root / 'items.json'), str(root / 'index.json')))
            aux += list(RLSeqTitle2SidDataset(paths['train'], sample=2))
            self.assertEqual([(r['prompt'], r['target']) for r in records[4:]], [(x['prompt'], x['completion'].strip()) for x in aux])
            args = TrainingArguments(output_dir=str(root / 'sft'), use_cpu=True, max_steps=2,
                per_device_train_batch_size=16, per_device_eval_batch_size=16, report_to=[], remove_unused_columns=False,
                save_strategy='no', prediction_loss_only=True, disable_tqdm=True)
            trainer = LatentSFTTrainer(model=model, args=args, processing_class=tk, data_collator=InterestCollator(tk),
                train_dataset=Dataset.from_list(train), eval_dataset=Dataset.from_list(valid))
            trainer.train(); self.assertTrue(np.isfinite(trainer.evaluate()['eval_loss']))
            trainer.save_model(root / 'sft_final')
            for mode in ('original', '4x4'):
                m = Qwen3ForCausalLM.from_pretrained(root / 'sft_final')
                args = InterestRLArguments(output_dir=str(root / mode), use_cpu=True, max_steps=2,
                    per_device_train_batch_size=16, per_device_eval_batch_size=16, report_to=[], remove_unused_columns=False,
                    save_strategy='no', prediction_loss_only=True, disable_tqdm=True, gradient_checkpointing=True,
                    ref_model_sync_steps=1)
                tiny = [records[0], records[4], records[-1]]
                trainer = LatentGRPOTrainer(model=m, args=args, processing_class=tk, grammar=grammar, mode=mode,
                    train_dataset=Dataset.from_list(tiny), eval_dataset=Dataset.from_list(tiny))
                trainer.train(); self.assertTrue(np.isfinite(trainer.evaluate()['eval_loss']))
                trainer.save_model(root / (mode + '_final'))
                samples = generate(m, [2], grammar, 50, sampling='beam')
                self.assertGreater(len(samples), 0)

    def test_merge_rejects_missing_tail_and_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            meta = dict(shards=2, examples=2, beams=50)
            for i in range(2):
                write_json(root / f'{i}.json', dict(shard=i, metadata=meta, rows=[dict(row_id=i, output='item', predict=['item'], unique_before_top50=1, valid_paths=50)]))
            merge([str(root / '0.json'), str(root / '1.json')], str(root / 'merged.json'))
            self.assertEqual(json.loads((root / 'merged.json').read_text())['metrics']['HR@1'], 1)
            with self.assertRaises(ValueError): merge([str(root / '0.json')], str(root / 'bad.json'))


if __name__ == '__main__':
    unittest.main()
