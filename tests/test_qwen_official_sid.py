import contextlib
import importlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from rq.text2emb.pooling import pool_embedding
import prepare_qwen_official_sid as official
from prepare_balanced_sid import CATEGORY, STEM, sha256
from tests import test_embedding06b_balanced as fixtures
from run_qwen_official_sid import idle_pair


class ItemPoolingTests(unittest.TestCase):
    def test_queue_rejects_idle_utilization_with_existing_compute_process(self):
        measurements = {"2": dict(used_mib=20, utilization=0, pids=[]),
                        "6": dict(used_mib=100, utilization=0, pids=[123]),
                        "4": dict(used_mib=20, utilization=0, pids=[])}
        consecutive = {}
        for _ in range(2):
            self.assertEqual(idle_pair(measurements, consecutive, ["2", "6", "4"]), [])
        self.assertEqual(idle_pair(measurements, consecutive, ["2", "6", "4"]), ["2", "4"])
        measurements["4"]["pids"] = [456]
        self.assertEqual(idle_pair(measurements, consecutive, ["2", "6", "4"]), [])
        self.assertEqual(consecutive["4"], 0)

    def test_padding_and_batching_do_not_change_last_token_embedding(self):
        hidden = torch.tensor([[[1, 2], [3, 4], [99, 99]],
                               [[99, 99], [1, 2], [3, 4]],
                               [[1, 2], [3, 4], [6, 8]]], dtype=torch.float16)
        mask = torch.tensor([[1, 1, 0], [0, 1, 1], [1, 1, 1]])
        pooled = pool_embedding(hidden, mask, "last_token", True)
        torch.testing.assert_close(pooled, torch.tensor([[.6, .8]] * 3))
        for i in range(3):
            plain = hidden[i, mask[i].bool()][None]
            torch.testing.assert_close(pooled[i:i+1], pool_embedding(
                plain, torch.ones(plain.shape[:2]), "last_token", True))
        mean = pool_embedding(hidden, mask)
        weights = mask[..., None].expand_as(hidden).float()
        torch.testing.assert_close(mean, (hidden * weights).sum(1) / weights.sum(1))
        self.assertFalse(torch.allclose(mean.norm(dim=1), torch.ones(3)))
        with self.assertRaisesRegex(ValueError, "padded"):
            pool_embedding(hidden[:1], torch.zeros_like(mask[:1]), "last_token", True)

    def test_actual_item_extractor_sorts_ids_and_ignores_padding(self):
        module_root = str(Path(__file__).resolve().parents[1] / "rq/text2emb")
        sys.path.insert(0, module_root)
        self.addCleanup(lambda: sys.path.remove(module_root))
        extractor = importlib.import_module("rq.text2emb.amazon_text2emb")

        class Tokenizer:
            pad_token = "<pad>"
            def __call__(self, texts, **kwargs):
                sequences = [[int(t) for t in text.split()] for text in texts]
                width = max(map(len, sequences))
                ids = torch.tensor([seq + [0] * (width-len(seq)) for seq in sequences])
                mask = torch.tensor([[1] * len(seq) + [0] * (width-len(seq)) for seq in sequences])
                class Batch(dict):
                    def to(self, device):
                        return SimpleNamespace(**self)
                return Batch(input_ids=ids, attention_mask=mask)

        calls = []
        def model(input_ids, attention_mask, **kwargs):
            calls.append(kwargs)
            hidden = torch.stack((input_ids.float(), input_ids.cumsum(-1).float()), -1).half()
            return SimpleNamespace(last_hidden_state=hidden)

        accelerator = SimpleNamespace(num_processes=1, process_index=0, is_main_process=True,
                                      is_local_main_process=True, device="cpu", wait_for_everyone=lambda: None)
        texts = [(2, "2 3 6"), (0, "3"), (1, "1 4")]
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(root=tmp, dataset="fixture", plm_name="qwen", max_sent_len=2048,
                                   batch_size=2, pooling="last_token", l2_normalize=True)
            with patch.object(extractor, "gather_object", side_effect=lambda rows: rows), contextlib.redirect_stdout(io.StringIO()):
                extractor.generate_item_embedding(args, texts, Tokenizer(), model, accelerator)
            result = np.load(Path(tmp) / "fixture.emb-qwen-td.npy")
            expected = torch.tensor([[3., 3.], [4., 5.], [6., 11.]])
            expected = torch.nn.functional.normalize(expected, dim=1).numpy()
            np.testing.assert_allclose(result, expected, atol=1e-7)
            self.assertEqual(result.dtype, np.float32)
            self.assertTrue(all(call == {"use_cache": False} for call in calls))


class OfficialSidPreparationTests(unittest.TestCase):
    def setUp(self):
        fixtures.Embedding06bTests.setUp(self)
        self.model.joinpath("config.json").write_text('{"hidden_size":2560,"model_type":"qwen3"}')
        self.encoded = self.root / "encoded"
        item_root = self.encoded / CATEGORY
        item_root.mkdir(parents=True)
        shutil.copy2(self.source / CATEGORY / f"{CATEGORY}.item.json", item_root)
        vectors = np.ones((4, 2560), dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        np.save(self.encoded / official.ARTIFACTS[0], vectors)
        inputs = [self.source / CATEGORY / f"{CATEGORY}.{s}.json" for s in ("item", "index")]
        inputs += [self.source / s / f"{STEM}.csv" for s in ("train", "valid", "test")]
        manifest = dict(recipe=official.RECIPE, embedding_sha256=sha256(self.encoded / official.ARTIFACTS[0]),
                        model_config_sha256=sha256(self.model / "config.json"),
                        source_inputs={str(p): sha256(p) for p in inputs})
        (self.encoded / "encoding_manifest.json").write_text(json.dumps(manifest))

    def fake_cluster(self, command, *, cwd, env, check):
        self.assertEqual(command[1], "rq/rqkmeans_constrained.py")
        for flag, value in (("--k", "256"), ("--l", "3"), ("--max_iter", "100"), ("--seed", "42")):
            self.assertEqual(command[command.index(flag)+1], value)
        np.save(self.encoded / official.ARTIFACTS[1], self.codes)
        np.savez(self.encoded / official.ARTIFACTS[2],
                 **{f"codebook_{i}": np.zeros((256, 2560), dtype=np.float32) for i in range(3)})
        (self.encoded / CATEGORY / f"{CATEGORY}.index.json").write_text(json.dumps(self.new))

    def prepare(self):
        with patch.object(official.subprocess, "run", side_effect=self.fake_cluster), contextlib.redirect_stdout(io.StringIO()):
            official.prepare_experiment(self.source, self.encoded, self.output, self.model)

    def test_publish_preserves_item_ids_text_splits_and_detects_tampering(self):
        fingerprints = {p: sha256(p) for p in self.source.rglob("*") if p.is_file()}
        self.prepare()
        self.assertEqual(fingerprints, {p: sha256(p) for p in fingerprints})
        import csv
        for split in ("train", "valid", "test"):
            with (self.output / split / f"{STEM}.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            for old, new in zip(self.rows, rows):
                self.assertEqual({k:v for k,v in old.items() if k not in {"item_sid", "history_item_sid"}},
                                 {k:v for k,v in new.items() if k not in {"item_sid", "history_item_sid"}})
                self.assertEqual(new["item_sid"], "".join(self.new["3"]))
        with contextlib.redirect_stdout(io.StringIO()):
            official.check_experiment(self.output)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            self.prepare()
        (self.output / official.ARTIFACTS[0]).write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "Changed"):
            official.check_experiment(self.output)

    def test_rejects_wrong_recipe_and_non_normalized_embeddings(self):
        manifest_path = self.encoded / "encoding_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["recipe"]["pooling"] = "masked_mean"
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.prepare()
        self.assertFalse(self.output.exists())
        np.save(self.encoded / official.ARTIFACTS[0], np.ones((4,2560), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "normalized"):
            official.validate_embeddings(self.encoded / official.ARTIFACTS[0], 4)


if __name__ == "__main__":
    unittest.main()
