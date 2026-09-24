"""CPU tests for the interest experiment (same torch/transformers as L40 stack)."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import GenerationConfig, PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM, TrainingArguments

from interest.core import NONE, ITEM, initialize_interests, pseudo_interests, rank_unique_items, validate_protocol
from interest.data import history_records, mid_datasets, rl_records, supervised_example
from interest.generation import Grammar, action_logps, generate, pack_trajectories, rollout
from interest.objective import advantages, grpo_loss
from interest.train import InterestRLArguments
from interest.trainer import GroupDataset, GroupSampler, InterestCollator, InterestGRPOTrainer, MidTrainer


INDEX = {"0": ["<a_0>", "<b_0>", "<c_0>"],
         "1": ["<a_1>", "<b_1>", "<c_1>", "<d_0>"],
         "2": ["<a_1>", "<b_1>", "<c_1>", "<d_1>"],
         "3": ["<a_2>", "<b_0>", "<c_1>"]}


def fixture():
    torch.manual_seed(7)
    tokens = ["<unk>", "<eos>", "history"] + sorted({x for sid in INDEX.values() for x in sid})
    tk = Tokenizer(WordLevel({x: i for i, x in enumerate(tokens)}, unk_token="<unk>"))
    tk.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>")
    tokenizer.add_tokens(tokens[3:])
    tokenizer.padding_side = "left"
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tokenizer), hidden_size=24, intermediate_size=32,
                                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                                        head_dim=12, max_position_embeddings=1024, attention_dropout=0.0,
                                        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
                                        tie_word_embeddings=False))
    return tokenizer, model


class InterestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def ready(self):
        tokenizer, model = fixture()
        initialize_interests(model, tokenizer, INDEX, 2)
        return tokenizer, model, Grammar(tokenizer, INDEX, 2)

    def test_history_only_labels_and_ties(self):
        h = ["<a_1><b_0><c_0>", "<a_2><b_0><c_0>"] * 2
        self.assertEqual(pseudo_interests(h), ["<interest_2>", "<interest_1>"])
        self.assertEqual(pseudo_interests(h[:1]), ["<interest_1>", NONE])
        self.assertEqual(pseudo_interests(h[:1] * 3 + h[1:2]), ["<interest_1>", "<interest_2>"])

    def test_embedding_copy_is_independent_and_reload_safe(self):
        tokenizer, model = fixture()
        original = model.get_input_embeddings().weight.detach().clone()
        initialize_interests(model, tokenizer, INDEX, 2)
        src = tokenizer.convert_tokens_to_ids("<a_0>")
        dst = tokenizer.convert_tokens_to_ids("<interest_0>")
        for matrix in (model.get_input_embeddings().weight, model.get_output_embeddings().weight):
            torch.testing.assert_close(matrix[src], matrix[dst])
        with torch.no_grad():
            model.get_input_embeddings().weight[dst].add_(1)
        torch.testing.assert_close(model.get_input_embeddings().weight[src], original[src])
        with tempfile.TemporaryDirectory() as path:
            model.save_pretrained(path)
            tokenizer.save_pretrained(path)
            loaded = Qwen3ForCausalLM.from_pretrained(path)
            tk = PreTrainedTokenizerFast.from_pretrained(path)
            validate_protocol(loaded.config, tk, INDEX)
            torch.testing.assert_close(loaded.get_input_embeddings().weight[dst], model.get_input_embeddings().weight[dst])
            changed = copy.deepcopy(INDEX)
            changed["0"][1] = "<b_1>"
            with self.assertRaises(ValueError):
                validate_protocol(loaded.config, tk, changed)
        with self.assertRaises(ValueError):
            initialize_interests(model, tokenizer, INDEX, 2)

    def test_two_level_advantage_and_zero_groups(self):
        hits = torch.tensor([1., 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        ia, ya, reward = advantages(hits, "4x4")
        torch.testing.assert_close(reward, torch.tensor([1., 0, 1, 0]))
        self.assertTrue((ia.reshape(4, 4)[[0, 2]] > 0).all())
        self.assertTrue((ia.reshape(4, 4)[[1, 3]] < 0).all())
        self.assertTrue((ya[[0, 9]] > 0).all())
        self.assertTrue((ya[4:8] == 0).all())
        for mode in ("16x1", "4x4"):
            for value in (0., 1.):
                i, y, _ = advantages(torch.full((16,), value), mode)
                self.assertTrue(torch.equal(i, torch.zeros(16)))
                self.assertTrue(torch.equal(y, torch.zeros(16)))

    def test_interest_gradient_once_and_missed_item_not_rewarded(self):
        hits = torch.zeros(16); hits[0] = hits[8] = 1
        ia, ya, _ = advantages(hits, "4x4")
        # Four shared interest parameters feed sixteen stored prefix copies.
        z = torch.zeros(4, requires_grad=True)
        y = torch.zeros(16, requires_grad=True)
        logps = torch.stack([z.repeat_interleave(4), y], dim=1)
        packed = dict(interest_mask=torch.tensor([[1., 0.]] * 16), item_mask=torch.tensor([[0., 1.]] * 16))
        loss, _ = grpo_loss(logps, logps.detach(), packed, [(0, "4x4", False, ia, ya)], 0)
        loss.backward()
        torch.testing.assert_close(z.grad, -ia[::4] / 4)
        self.assertLess(y.grad[0].item(), 0)
        self.assertGreater(y.grad[1].item(), 0)
        self.assertEqual(y.grad[4].item(), 0)

    def test_sampling_branch_identity_grammar_and_logps(self):
        tokenizer, model, grammar = self.ready()
        model.eval()
        prompt = tokenizer.encode("history", add_special_tokens=False)
        for mode in ("16x1", "4x4"):
            seqs, branches, n, m = rollout(model, prompt, grammar, mode)
            self.assertEqual(len(seqs), 16)
            for i in range(n):
                self.assertTrue(all(seq[:3] == seqs[i*m][:3] for seq in seqs[i*m:(i+1)*m]))
            items = [grammar.item(s) for s in seqs]
            self.assertTrue(set(items) <= {"".join(s) for s in INDEX.values()})
            packed = pack_trajectories([prompt] * 16, seqs, [3] * 16, [False] * 16, grammar, "cpu")
            lp = action_logps(model, packed)
            self.assertTrue(torch.isfinite(lp).all())
            self.assertTrue((lp <= 1e-6).all())
            self.assertTrue(torch.equal(lp[:, 2], torch.zeros(16)))  # forced boundary
        a = generate(model, prompt, grammar, 4, sample=False)
        b = generate(model, prompt, grammar, 4, sample=False)
        self.assertEqual(a, b)
        too_many_beams = generate(model, prompt, grammar, 50, sample=False)
        self.assertTrue(too_many_beams)
        for seq in too_many_beams:
            self.assertIn(grammar.item(seq), {"".join(s) for s in INDEX.values()})
        config = GenerationConfig(max_new_tokens=8, do_sample=False, num_beams=4, num_return_sequences=4,
                                   pad_token_id=grammar.eos, eos_token_id=grammar.eos, renormalize_logits=True,
                                   length_penalty=0.0, return_dict_in_generate=True, output_scores=True)
        raw = model.generate(torch.tensor([prompt]), attention_mask=torch.ones(1, len(prompt), dtype=torch.long),
                             generation_config=config, prefix_allowed_tokens_fn=lambda b, ids: grammar.allowed(ids[len(prompt):].tolist()))
        candidates = [s[len(prompt):].tolist() for s in raw.sequences]
        candidates = [s[:s.index(grammar.eos)+1] for s in candidates]
        packed = pack_trajectories([prompt]*4, candidates, [3]*4, [False]*4, grammar, "cpu")
        scores = (action_logps(model, packed) * (packed["interest_mask"] + packed["item_mask"])).sum(1)
        torch.testing.assert_close(scores, raw.sequences_scores, atol=2e-6, rtol=2e-6)
        # RL item-identification must keep three-level targets legal.
        auxiliary = generate(model, prompt, grammar, 16, "item", auxiliary=True)
        for seq in auxiliary:
            self.assertNotIn("<d_", grammar.item(seq, auxiliary=True))

    def test_mid_loss_and_complete_trainer_updates(self):
        for mode in ("mid", "16x1", "4x4"):
            tokenizer, model, grammar = self.ready()
            record = dict(prompt="history", target="".join(INDEX["0"]), history=["".join(INDEX["0"])],
                          task="history_sid", auxiliary=False)
            with tempfile.TemporaryDirectory() as path:
                opts = dict(output_dir=path, use_cpu=True, report_to=[], max_steps=2, learning_rate=1e-3,
                            per_device_train_batch_size=16, per_device_eval_batch_size=16,
                            gradient_accumulation_steps=2, remove_unused_columns=False,
                            eval_strategy="steps", eval_steps=1, save_strategy="steps", save_steps=1,
                            logging_steps=1, disable_tqdm=True, prediction_loss_only=True)
                if mode == "mid":
                    example = supervised_example(record, tokenizer, 2, 64)
                    ds = Dataset.from_list([example] * 32)
                    trainer = MidTrainer(model=model, args=TrainingArguments(**opts), processing_class=tokenizer,
                                         train_dataset=ds, eval_dataset=ds, data_collator=InterestCollator(tokenizer))
                else:
                    aux = dict(record, task="item_identification", auxiliary=True)
                    ds = Dataset.from_list([record, aux])
                    opts["gradient_checkpointing"] = True
                    trainer = InterestGRPOTrainer(model=model, args=InterestRLArguments(**opts), processing_class=tokenizer,
                                                  train_dataset=ds, eval_dataset=ds, grammar=grammar, mode=mode)
                before = model.get_input_embeddings().weight.detach().clone()
                trainer.train()
                self.assertEqual(trainer.state.global_step, 2)
                self.assertFalse(torch.equal(before, model.get_input_embeddings().weight))
                self.assertTrue(any("eval_loss" in x for x in trainer.state.log_history))
                checkpoint = Path(path) / "checkpoint-2"
                loaded = Qwen3ForCausalLM.from_pretrained(checkpoint)
                validate_protocol(loaded.config, tokenizer, INDEX)

    def test_multitask_data_labels_and_index_validation(self):
        tokenizer, model, grammar = self.ready()
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            index_file, item_file, train_file = root / "index.json", root / "items.json", root / "train.csv"
            index_file.write_text(json.dumps(INDEX))
            item_file.write_text(json.dumps({str(i): {"title": f"title {i}", "description": f"description {i}"} for i in range(4)}))
            rows = [dict(history_item_id=repr([0, 3]), history_item_sid=repr(["".join(INDEX["0"]), "".join(INDEX["3"])]),
                         history_item_title=repr(["title 0", "title 3"]), item_id=i, item_sid="".join(INDEX[str(i)])) for i in (1, 2)]
            with train_file.open("w") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            sid = history_records(train_file, INDEX)
            title = history_records(train_file, INDEX, "title")
            self.assertEqual(pseudo_interests(sid[0]["history"]), pseudo_interests(sid[1]["history"]))
            for record in sid + title:
                example = supervised_example(record, tokenizer, 2)
                target = [x for x in example["labels"] if x != -100]
                self.assertEqual(grammar.item(target), record["target"])
            train, valid = mid_datasets(str(train_file), str(train_file), str(item_file), str(index_file), INDEX, tokenizer, 2)
            self.assertGreater(len(train), len(sid) + len(title))
            self.assertEqual(len(valid), 2)
            self.assertEqual(sum(any(x["interest_mask"]) for x in train), 4)
            rl = rl_records(str(train_file), str(item_file), str(index_file), INDEX)
            self.assertEqual({r["task"] for r in rl}, {"history_sid", "history_title", "item_identification"})
            self.assertTrue(all(len(tokenizer.encode(r["target"], add_special_tokens=False)) == 3 for r in rl if r["auxiliary"]))
            rows[0]["history_item_sid"] = repr(["".join(INDEX["1"]), "".join(INDEX["3"])])
            with train_file.open("w") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            with self.assertRaises(ValueError):
                history_records(train_file, INDEX)

    def test_sampler_whole_groups_and_dedup(self):
        sampler = GroupSampler(list(range(7)))
        ids = list(sampler)
        self.assertEqual(len(ids), 112)
        for i in range(0, len(ids), 16):
            self.assertEqual(len(set(ids[i:i+16])), 1)
        sampler.set_epoch(1)
        self.assertNotEqual(ids, list(sampler))
        self.assertEqual(rank_unique_items([("a", -2), ("b", -1), ("a", -.5)]), ["a", "b"])
        grouped = GroupDataset(list(range(7)))
        self.assertEqual(len(grouped), 112)
        emitted = list(GroupSampler(grouped))
        self.assertEqual(sorted(emitted), list(range(112)))
        for i in range(0, len(emitted), 16):
            self.assertEqual(len({grouped[x] for x in emitted[i:i+16]}), 1)


if __name__ == "__main__":
    unittest.main()
