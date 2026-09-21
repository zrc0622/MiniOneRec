"""EOS terminal regression checks, including real CPU beam generation."""

import unittest
import warnings

import torch
from transformers import GenerationConfig, GPT2Config, GPT2LMHeadModel, LogitsProcessorList

from LogitProcessor import ConstrainedLogitsProcessor


class LegacyProcessor(ConstrainedLogitsProcessor):
    """Disable only terminal recognition to retain the pre-fix code path."""

    def _is_completed_sequence(self, batch_id, token_ids):
        return False


class ConstrainedLogitsProcessorTests(unittest.TestCase):
    root = [10, 11, 12]
    eos = 2
    paths = [
        [15, 18, 21, 3, 2],
        [16, 19, 22, 24, 3, 2],
        [16, 19, 22, 25, 3, 2],
        [17, 20, 23, 3, 2],
    ]

    def setUp(self):
        self.trie = {}
        for path in self.paths:
            self.trie.setdefault(tuple(self.root), set()).add(path[0])
            for k in range(1, len(path)):
                self.trie.setdefault(tuple(path[:k]), set()).add(path[k])

    def processor(self, cls=ConstrainedLogitsProcessor, beams=1):
        return cls(lambda batch, key: list(self.trie.get(tuple(key), ())), beams, "qwen3", self.eos)

    def apply(self, processor, generated):
        processor.count = len(generated)
        inputs = torch.tensor([self.root + generated])
        scores = torch.arange(32, dtype=torch.float32).unsqueeze(0) / 10
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = processor(inputs, scores)
        return result, caught

    def test_legal_three_and_four_level_terminals_keep_exact_scores_without_warning(self):
        for path in self.paths:
            for padding in ([], [self.eos], [self.eos, self.eos]):
                with self.subTest(path=path, padding=padding):
                    generated = path + padding
                    old, old_warnings = self.apply(self.processor(LegacyProcessor), generated)
                    new, new_warnings = self.apply(self.processor(), generated)
                    self.assertEqual(len(old_warnings), 1)
                    self.assertEqual(len(new_warnings), 0)
                    self.assertTrue(torch.equal(old, new))
                    self.assertEqual(torch.where(torch.isfinite(new[0]))[0].tolist(), [self.eos])

    def test_real_invalid_prefixes_still_warn_and_force_eos(self):
        for generated in ([29], [29, 2], [15, 2], [2], self.paths[0] + [29]):
            with self.subTest(generated=generated):
                old, _ = self.apply(self.processor(LegacyProcessor), list(generated))
                new, caught = self.apply(self.processor(), list(generated))
                self.assertEqual(len(caught), 1)
                self.assertIn("No valid tokens", str(caught[0].message))
                self.assertTrue(torch.equal(old, new))

    def test_live_prefixes_are_unchanged(self):
        for path in self.paths:
            for length in range(len(path)):
                old, _ = self.apply(self.processor(LegacyProcessor), path[:length])
                new, caught = self.apply(self.processor(), path[:length])
                self.assertFalse(caught)
                self.assertTrue(torch.equal(old, new))

    def test_prompt_eos_is_not_mistaken_for_a_generated_terminal(self):
        processor = self.processor()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # count=0: these are prompt tokens, not a generated completion.
            processor(torch.tensor([[15, 3, self.eos]]), torch.zeros((1, 32)))
        self.assertEqual(len(caught), 1)

    def test_no_eos_configuration_keeps_old_invalid_behavior(self):
        processor = self.processor()
        processor.eos_token_id = None
        masked, caught = self.apply(processor, self.paths[0])
        self.assertEqual(len(caught), 1)
        self.assertTrue(torch.isneginf(masked).all())

    def test_terminal_validation_uses_the_correct_batch(self):
        prefix = self.paths[0][:-1]
        processor = ConstrainedLogitsProcessor(
            lambda batch, key: [self.eos] if batch == 0 and key == prefix else [],
            2, "qwen3", self.eos,
        )
        processor.count = len(self.paths[0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            masked = processor(torch.tensor([self.root + self.paths[0]] * 4), torch.zeros((4, 32)))
        # Batch 0's two beams are legal terminals; batch 1's two are not.
        self.assertEqual(len(caught), 2)
        self.assertEqual(torch.isfinite(masked).sum().item(), 4)

    def test_real_beam_generation_preserves_sequences_and_scores(self):
        torch.manual_seed(42)
        model = GPT2LMHeadModel(GPT2Config(
            vocab_size=32, n_positions=32, n_embd=16, n_layer=1, n_head=2,
            bos_token_id=1, eos_token_id=self.eos, pad_token_id=self.eos,
        )).eval()
        config = GenerationConfig(
            num_beams=4, num_return_sequences=4, do_sample=False,
            max_new_tokens=12, length_penalty=0.0, bos_token_id=1,
            eos_token_id=self.eos, pad_token_id=self.eos,
        )
        results = []
        messages = []
        for cls in (LegacyProcessor, ConstrainedLogitsProcessor):
            with warnings.catch_warnings(record=True) as caught, torch.no_grad():
                warnings.simplefilter("always")
                result = model.generate(
                    torch.tensor([self.root, self.root]),
                    attention_mask=torch.ones((2, 3), dtype=torch.long),
                    generation_config=config,
                    logits_processor=LogitsProcessorList([self.processor(cls, beams=4)]),
                    do_sample=False, return_dict_in_generate=True, output_scores=True,
                )
            results.append(result)
            messages.append([str(w.message) for w in caught if "No valid tokens" in str(w.message)])
        self.assertTrue(messages[0], "legacy processor should reproduce EOS warnings")
        self.assertFalse(messages[1])
        old, new = results
        self.assertTrue(torch.equal(old.sequences, new.sequences))
        self.assertTrue(torch.equal(old.sequences_scores, new.sequences_scores))
        self.assertEqual(len(old.scores), len(new.scores))
        for a, b in zip(old.scores, new.scores):
            self.assertTrue(torch.equal(a, b))
        for row in new.sequences[:, len(self.root):].tolist():
            completion = row[:row.index(self.eos) + 1]
            self.assertIn(completion, self.paths)


if __name__ == "__main__":
    unittest.main()
