import ast
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import prepare_embedding06b_balanced as experiment
from prepare_balanced_sid import CATEGORY, STEM, MANIFEST, sha256


class Embedding06bTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="embedding06b-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.source, self.output, self.model = (self.root / name for name in ("source data", "new data", "model"))
        item_root = self.source / CATEGORY
        item_root.mkdir(parents=True)
        self.model.mkdir()
        (self.model / "config.json").write_text(json.dumps({"hidden_size": 1024, "model_type": "qwen3"}))
        self.items = {str(i): {"title": f'商品{i}, "标题"', "description": "同一商品文本"} for i in range(4)}
        self.old = {str(i): [f"<a_{i + 10}>", "<b_1>", "<c_1>"] for i in range(4)}
        self.codes = np.array([[0, 1, 255], [2, 3, 4], [0, 1, 255], [0, 1, 255]], dtype=np.int32)
        self.new = {"0": ["<a_1>", "<b_2>", "<c_256>", "<d_1>"],
                    "1": ["<a_3>", "<b_4>", "<c_5>"],
                    "2": ["<a_1>", "<b_2>", "<c_256>", "<d_2>"],
                    "3": ["<a_1>", "<b_2>", "<c_256>", "<d_3>"]}
        for suffix, data in (("item", self.items), ("index", self.old)):
            (item_root / f"{CATEGORY}.{suffix}.json").write_text(json.dumps(data))
        self.rows = [{"user_id": "001", "history_item_id": "[0, 2]", "item_id": "3",
                      "history_item_sid": repr(["".join(self.old[k]) for k in ("0", "2")]),
                      "item_sid": "".join(self.old["3"]), "item_title": self.items["3"]["title"],
                      "extra": "原样\n保留"}] * 2
        for split in ("train", "valid", "test"):
            folder = self.source / split
            folder.mkdir()
            with (folder / f"{STEM}.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.rows[0])
                writer.writeheader(); writer.writerows(self.rows)
        self.calls = []
        self.bad = None

    def fake_generation(self, command, *, cwd, env, check):
        self.calls.append((command, env))
        item_root = Path(env["DATA_ROOT"]) / CATEGORY
        self.assertEqual(cwd, REPO)
        self.assertEqual(Path(env["EMB_MODEL"]), self.model)
        self.assertEqual((item_root / f"{CATEGORY}.item.json").read_bytes(),
                         (self.source / CATEGORY / f"{CATEGORY}.item.json").read_bytes())
        if command[0] == "bash":
            self.assertEqual(command, ["bash", "rq/text2emb/amazon_text2emb.sh"])
            values = np.ones((4, 2560 if self.bad == "dimension" else 1024), dtype=np.float32)
            if self.bad == "nan": values[0, 0] = np.nan
            np.save(item_root / f"{CATEGORY}.emb-qwen-td.npy", values)
        else:
            self.assertEqual(command[1:], ["rq/rqkmeans_constrained.py", "--dataset", CATEGORY,
                             "--root", str(item_root), "--k", "256", "--l", "3",
                             "--max_iter", "100", "--seed", "42", "--verbose"])
            np.save(item_root / f"{CATEGORY}.codes_constrained.npy", self.codes)
            np.savez(item_root / f"{CATEGORY}.codebooks_constrained.npz",
                     **{f"codebook_{i}": np.zeros((256, 1024), dtype=np.float32) for i in range(3)})
            (item_root / f"{CATEGORY}.index.json").write_text(json.dumps({} if self.bad == "index" else self.new))
            if self.bad == "source_changed":
                path = self.source / "train" / f"{STEM}.csv"
                path.write_text(path.read_text().replace("001", "002"))

    def prepare(self):
        with patch.object(experiment.subprocess, "run", side_effect=self.fake_generation), contextlib.redirect_stdout(io.StringIO()):
            experiment.prepare_experiment(self.source, self.output, self.model)

    def test_preserves_source_rows_and_publishes_portable_checked_dataset(self):
        hashes = {p: sha256(p) for p in self.source.rglob("*") if p.is_file()}
        self.prepare()
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(hashes, {p: sha256(p) for p in hashes})
        self.assertEqual(json.loads((self.output / CATEGORY / f"{CATEGORY}.index.json").read_text()), self.new)
        for split in ("train", "valid", "test"):
            with (self.output / split / f"{STEM}.csv").open(newline="") as f: rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            for before, after in zip(self.rows, rows):
                for key in before.keys() - {"item_sid", "history_item_sid"}: self.assertEqual(before[key], after[key])
                self.assertEqual(after["item_sid"], "".join(self.new["3"]))
                self.assertEqual(ast.literal_eval(after["history_item_sid"]), ["".join(self.new[k]) for k in ("0", "2")])
        manifest = json.loads((self.output / MANIFEST).read_text())
        self.assertTrue(Path(manifest["codes_path"]).is_file())
        self.assertEqual(manifest["inputs"][manifest["codes_path"]], sha256(manifest["codes_path"]))
        with contextlib.redirect_stdout(io.StringIO()): experiment.check_experiment(self.output)
        self.assertFalse(list(self.output.parent.glob(".embedding06b-balanced-*")))
        with self.assertRaisesRegex(ValueError, "输出目录已存在"): self.prepare()

    def test_invalid_generation_never_publishes_dataset(self):
        for failure in ("dimension", "nan", "index", "source_changed"):
            with self.subTest(failure=failure):
                self.bad = failure
                with self.assertRaises(ValueError): self.prepare()
                self.assertFalse(self.output.exists())
                self.assertFalse(list(self.output.parent.glob(".embedding06b-balanced-*")))

    def test_missing_input_or_wrong_model_fails_before_generation(self):
        (self.model / "config.json").write_text('{"hidden_size":2560}')
        with self.assertRaisesRegex(ValueError, "hidden_size=1024"): self.prepare()
        self.assertEqual(self.calls, [])
        (self.source / "test" / f"{STEM}.csv").unlink()
        with self.assertRaises(FileNotFoundError): self.prepare()
        self.assertEqual(self.calls, [])

    def test_rejects_old_recipe_and_tampered_artifacts(self):
        self.prepare()
        recipe_path = self.output / experiment.RECIPE_FILE
        original = recipe_path.read_bytes()
        recipe = json.loads(original); recipe["recipe"]["embedding_dim"] = 2560
        recipe_path.write_text(json.dumps(recipe))
        with self.assertRaisesRegex(ValueError, "1024维"): experiment.check_experiment(self.output)
        recipe_path.write_bytes(original)
        code_path = self.output / CATEGORY / f"{CATEGORY}.codes_constrained.npy"
        code_path.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "产物与清单"): experiment.check_experiment(self.output)

    def test_shell_sft_eval_download_and_tensorboard_isolation(self):
        self.prepare()
        binary = self.root / "bin"; binary.mkdir()
        capture = self.root / "calls.jsonl"
        endpoint = binary / "endpoint"
        endpoint.write_text(f"#!{sys.executable}\n" + '''import json,os,sys
from pathlib import Path
import subprocess
args=sys.argv[1:]
name=Path(sys.argv[0]).name
with open(os.environ['CAPTURE'],'a') as f:
    f.write(json.dumps(dict(name=name,args=args,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),data=os.environ.get('DATA_ROOT')))+'\\n')
if name=='python' and args and args[0]=='prepare_embedding06b_balanced.py' and '--check' in args:
    raise SystemExit(subprocess.call([sys.executable,*args]))
''')
        endpoint.chmod(0o755)
        for name in ("python", "torchrun", "hf", "tensorboard"): (binary / name).symlink_to(endpoint)
        env = os.environ | {"PATH": str(binary) + os.pathsep + os.environ["PATH"], "CAPTURE": str(capture),
            "CUDA_VISIBLE_DEVICES": "3,4,5,6", "EMB06B_SOURCE_DATA_ROOT": str(self.source),
            "EMB06B_DATA_ROOT": str(self.output), "EMB06B_MODEL": str(self.model),
            "QWEN3_06B_MODEL": str(self.root / "backbone06b"),
            "QWEN3_06B_EMB06B_BALANCED_SFT_OUTPUT_DIR": str(self.root / "new model"),
            "QWEN3_06B_BALANCED_SFT_OUTPUT_DIR": str(self.root / "old balanced"),
            "LOG_DIR": str(self.root / "logs"), "RESULT_DIR": str(self.root / "results"),
            "BALANCED_DATA_ROOT": "/wrong/4b/data", "EMB_MODEL": "/wrong/4b/model", "BASE_MODEL": "/wrong/17b",
            "SFT_OUTPUT_DIR": "/wrong/old/output"}
        def run(*args, extra=None):
            return subprocess.run(["bash", "run_qwen3_0.6b_emb06b_balanced.sh", *args], cwd=REPO,
                                  env=env | (extra or {}), capture_output=True, text=True)
        for args in (("download",), ("sft",), ("eval", "sft"), ("tensorboard",)):
            result = run(*args); self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        records = [json.loads(line) for line in capture.read_text().splitlines()]
        train = next(x for x in records if x["name"] == "torchrun")
        flags = train["args"]
        for flag, expected in {"--base_model": str(self.root / "backbone06b"), "--learning_rate": "3e-4",
            "--num_epochs": "10", "--eval_step": "0.05", "--batch_size": "1024", "--micro_batch_size": "16",
            "--deepspeed": "./config/sft_zero2.json", "--train_file": str(self.output / "train" / f"{STEM}.csv"),
            "--sid_index_path": str(self.output / CATEGORY / f"{CATEGORY}.index.json"),
            "--output_dir": str(self.root / "new model")}.items():
            self.assertEqual(flags[flags.index(flag)+1], expected)
        self.assertEqual(train["gpu"], "3,4,5,6")
        evaluations = [x for x in records if x["args"] and x["args"][0] == "evaluate.py"]
        self.assertEqual({x["gpu"] for x in evaluations}, {"3", "4", "5", "6"})
        for record in evaluations:
            self.assertIn(str(self.root / "new model" / "final_checkpoint"), record["args"])
            self.assertIn(str(self.output / "info" / f"{STEM}.txt"), record["args"])
        self.assertEqual(next(x for x in records if x["name"] == "hf")["args"],
                         ["download", "Qwen/Qwen3-Embedding-0.6B", "--local-dir", str(self.model)])
        tb = next(x for x in records if x["name"] == "tensorboard")["args"]
        self.assertIn(f"emb4b_balanced_10ep:{self.root}/old balanced/tensorboard,emb06b_balanced_10ep:{self.root}/new model/tensorboard", tb)
        for args in (("rl",), ("eval",), ("eval", "rl"), ("sft", "--num_epochs", "15")):
            self.assertNotEqual(run(*args).returncode, 0)
        self.assertNotEqual(run("sft", extra={"CUDA_VISIBLE_DEVICES": "3,3,5,6"}).returncode, 0)
        (self.output / "train" / f"{STEM}.csv").write_text("invalid data")
        capture.write_text("")
        self.assertNotEqual(run("sft").returncode, 0)
        self.assertNotIn('"name": "torchrun"', capture.read_text())


if __name__ == "__main__":
    unittest.main()
