import copy
import itertools
from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from tests.test_latent import fixture, INDEX
from latent.core import validate_protocol
from latent.data import supervised_example
from latent.parallel import (initialize_queries, install_attention, query_ids, query_attention_mask,
                             parallel_example, component_loss, joint_proposals, ConditionalScorer,
                             batch_parallel, ParallelSFTTrainer)
from latent.trainer import InterestCollator
from latent.eval_unified_small import padded


class ParallelInterestTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tk, self.model, self.protocol, self.grammar = fixture(size=8)
        initialize_queries(self.model, self.tk)
        self.model.eval()
        self.q = query_ids(self.model)
        self.h = self.tk.encode('history <a_0> <b_0>')
        self.codes = [ids[0] for ids in self.grammar.slots]
        self.sid = list(next(iter(self.grammar.sid_by_ids[False])))

    def test_symmetric_query_mask_and_no_future_edges(self):
        ids = torch.tensor([self.h + self.q + self.codes + self.sid])
        mask = query_attention_mask(ids, torch.ones_like(ids), self.q, torch.float32)[0, 0]
        p = len(self.h)
        self.assertTrue((mask[p:p + 3, :p + 3] == 0).all())
        self.assertTrue((mask[p:p + 3, p + 3:] < -1e20).all())
        self.assertTrue((mask[:p, p:] < -1e20).all())
        self.assertLess(mask[0, 1], -1e20)

    def test_gold_codes_and_sid_cannot_change_query_predictions(self):
        ids = torch.tensor([self.h + self.q + self.codes + self.sid])
        changed = ids.clone()
        changed[:, len(self.h) + 3:] = self.grammar.slots[0][-1]
        with torch.no_grad():
            a = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
            b = self.model(input_ids=changed, attention_mask=torch.ones_like(ids), use_cache=False).logits
        torch.testing.assert_close(a[:, :len(self.h) + 3], b[:, :len(self.h) + 3], atol=0, rtol=0)
        self.assertGreater((a[:, -1] - b[:, -1]).abs().max().item(), 1e-7)
        with torch.no_grad():
            self.model.model.embed_tokens.weight[self.q[2]].add_(torch.randn_like(self.model.model.embed_tokens.weight[self.q[2]]))
            c = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        torch.testing.assert_close(a[:, :len(self.h)], c[:, :len(self.h)], atol=0, rtol=0)
        self.assertGreater((a[:, len(self.h)] - c[:, len(self.h)]).abs().max().item(), 1e-7)

    def test_parallel_targets_are_at_query_positions_and_history_is_preserved(self):
        old = supervised_example(dict(prompt='history <a_0>', target=''.join(INDEX['2'])),
                                 [0, 1, 2], self.tk, self.protocol['tokens'], max_length=12)
        new = parallel_example(old, self.q)
        p = next(i for i, m in enumerate(old['interest_mask']) if m)
        self.assertEqual(new['input_ids'][:p], old['input_ids'][:p])
        self.assertEqual(new['input_ids'][p:p + 3], self.q)
        self.assertEqual(new['labels'][p:p + 3], old['input_ids'][p:p + 3])
        self.assertEqual(new['labels'][p + 3:p + 6], [-100] * 3)
        self.assertEqual(new['labels'][p + 6:], old['labels'][p + 3:])

    def test_loss_matches_explicit_same_position_interest_and_shifted_sid(self):
        original = supervised_example(dict(prompt='history', target=''.join(INDEX['2'])),
                                      [0, 1, 2], self.tk, self.protocol['tokens'])
        main = parallel_example(original, self.q)
        aux = dict(input_ids=self.h + self.sid, attention_mask=[1] * (len(self.h) + len(self.sid)),
                   labels=[-100] * len(self.h) + self.sid, interest_mask=[0] * (len(self.h) + len(self.sid)))
        packed = InterestCollator(self.tk)([main, aux])
        logits = torch.randn(*packed['input_ids'].shape, len(self.tk), requires_grad=True)
        loss, _ = component_loss(logits, packed['labels'], packed['interest_mask'], .7)
        q = packed['interest_mask'][0].bool()
        expected_latent = F.cross_entropy(logits[0, q], packed['labels'][0, q])
        shifted = packed['labels'][:, 1:].clone()
        shifted[packed['interest_mask'][:, 1:].bool()] = -100
        expected_main = F.cross_entropy(logits[0, :-1], shifted[0], ignore_index=-100)
        expected_aux = F.cross_entropy(logits[1, :-1], shifted[1], ignore_index=-100)
        torch.testing.assert_close(loss, (expected_main + .7 * expected_latent + expected_aux) / 2)
        loss.backward()
        self.assertGreater(logits.grad[0, q].abs().sum().item(), 0)
        aux_loss, _ = component_loss(logits[1:], packed['labels'][1:], packed['interest_mask'][1:], .7)
        torch.testing.assert_close(aux_loss, expected_aux)

    def test_top4_prior_matches_full_cartesian_enumeration(self):
        logits = torch.randn(2, 3, len(self.tk))
        logps = logits.log_softmax(-1)
        codes, scores = joint_proposals(logps, self.grammar.slots, 4)
        for user in range(2):
            expected = sorted(((sum(logps[user, k, t].item() for k, t in enumerate(code)), list(code))
                               for code in itertools.product(*self.grammar.slots)), reverse=True)[:4]
            self.assertEqual(codes[user * 4:(user + 1) * 4], [c for _, c in expected])
            torch.testing.assert_close(scores[user], torch.tensor([p for p, _ in expected]), atol=1e-6, rtol=1e-6)

    def test_cached_conditional_sid_scores_match_full_sequence(self):
        prompts = [self.h, self.h + self.h]
        ids, mask = padded([p + self.q for p in prompts], self.grammar.eos, torch.device('cpu'))
        with torch.no_grad():
            out = self.model(input_ids=ids, attention_mask=mask,
                             position_ids=(mask.cumsum(-1) - 1).clamp_min(0), use_cache=True)
        paths, users, expected = [], [], []
        for user, prompt in enumerate(prompts):
            for sid in list(self.grammar.sid_by_ids[False])[:5]:
                path = self.codes + list(sid)
                paths.append(path)
                users.append(user)
                full = torch.tensor([prompt + self.q + path])
                with torch.no_grad():
                    logits = self.model(input_ids=full, attention_mask=torch.ones_like(full), use_cache=False).logits[0].float()
                start = len(prompt) + 3 + 3 - 1
                logits = logits[start:-1]
                targets = torch.tensor(sid)
                expected.append((logits.gather(-1, targets[:, None]).squeeze(-1) - logits.logsumexp(-1)).sum())
        actual = ConditionalScorer(self.model, out.past_key_values, mask).score(paths, users, 3)
        torch.testing.assert_close(torch.tensor(actual), torch.stack(expected), atol=1e-5, rtol=1e-6)

    def test_parallel_protocol_reload_and_batched_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.model.save_pretrained(tmp)
            self.tk.save_pretrained(tmp)
            tk, model = AutoTokenizer.from_pretrained(tmp), AutoModelForCausalLM.from_pretrained(tmp)
            model.eval()
            with self.assertRaises(ValueError):validate_protocol(model.config, tk, INDEX)
            validate_protocol(model.config, tk, INDEX, allow_parallel=True)
            install_attention(model, tk)
            for method in ('single20', 'four20_marginal'):
                prompts = [self.h, self.h + self.h]
                actual = batch_parallel(model, self.grammar, prompts, method)
                separate = [batch_parallel(model, self.grammar, [p], method)[0] for p in prompts]
                self.assertEqual([r['predict'] for r in actual], [r['predict'] for r in separate])
                self.assertTrue(all(len(r['predict']) == 20 for r in actual))

    def test_trainer_recognizes_labels_and_computes_validation_loss(self):
        from datasets import Dataset
        original = supervised_example(dict(prompt='history', target=''.join(INDEX['2'])),
                                      [0, 1, 2], self.tk, self.protocol['tokens'])
        ds = Dataset.from_list([parallel_example(original, self.q)] * 2)
        with tempfile.TemporaryDirectory() as tmp:
            args = TrainingArguments(output_dir=tmp, use_cpu=True, per_device_eval_batch_size=2,
                                     report_to=[], remove_unused_columns=False, prediction_loss_only=True)
            trainer = ParallelSFTTrainer(model=self.model, args=args, processing_class=self.tk,
                                         data_collator=InterestCollator(self.tk), eval_dataset=ds)
            self.assertEqual(trainer.label_names, ['labels'])
            self.assertIn('eval_loss', trainer.evaluate())


if __name__ == '__main__':unittest.main()
