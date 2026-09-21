import ast
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_balanced_sid import CATEGORY, STEM, MANIFEST, build_index, check_prepared, prepare, sha256


class BalancedSidTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source data"
        self.output = self.source / "variants" / "balanced"
        item_dir = self.source / CATEGORY
        item_dir.mkdir(parents=True)
        self.codes_path = item_dir / f"{CATEGORY}.codes_constrained.npy"
        self.codes = np.array([[0, 1, 255], [2, 3, 4], [0, 1, 255], [0, 1, 255]], dtype=np.int32)
        np.save(self.codes_path, self.codes)
        self.items = {str(i): {"title": f'商品 {i}, "标题"\n第二行'} for i in range(4)}
        self.old_index = {str(i): [f"<a_{i + 10}>", "<b_1>", "<c_1>"] for i in range(4)}
        for suffix, data in [("item", self.items), ("index", self.old_index)]:
            (item_dir / f"{CATEGORY}.{suffix}.json").write_text(json.dumps(data), encoding="utf-8")
        self.rows = []
        for history, target in [([0, 2], 3), ([], 1), ([1, 0], 2), ([0, 2], 3)]:
            self.rows.append({
                "user_id": "same_user", "history_item_id": repr(history), "item_id": str(target),
                "history_item_sid": repr(["".join(self.old_index[str(i)]) for i in history]),
                "item_sid": "".join(self.old_index[str(target)]),
                "item_title": self.items[str(target)]["title"], "extra_column": "001,原样",
            })
        for split in ("train", "valid", "test"):
            path = self.source / split / f"{STEM}.csv"
            path.parent.mkdir()
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=self.rows[0])
                writer.writeheader(); writer.writerows(self.rows)

    def run_prepare(self):
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(self.source, self.output, self.codes_path)

    def test_matches_original_polars_offset_and_collision_suffix(self):
        # Execute the original helper, avoiding imports of the clustering runtime.
        source = Path(__file__).resolve().parents[1] / "rq/rqkmeans_constrained.py"
        tree = ast.parse(source.read_text())
        helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "deal_with_deduplicate")
        namespace = {"pl": pl}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source), "exec"), namespace)
        df = pl.DataFrame({"codes": (self.codes + 1).tolist()})
        expected = {str(i): [f"<{chr(97 + j)}_{v}>" for j, v in enumerate(row["codes"])]
                    for i, row in enumerate(namespace["deal_with_deduplicate"](df).iter_rows(named=True))}
        result = build_index(self.codes, self.items)
        self.assertEqual(result, expected)
        self.assertEqual([result[str(i)][-1] for i in (0, 2, 3)], ["<d_1>", "<d_2>", "<d_3>"])
        self.assertEqual(len(result["1"]), 3)

    def test_rows_and_sources_preserved_with_consistent_new_info(self):
        inputs = {p: sha256(p) for p in self.source.rglob("*") if p.is_file()}
        self.run_prepare()
        self.assertEqual(inputs, {p: sha256(p) for p in inputs})
        index = json.loads((self.output / CATEGORY / f"{CATEGORY}.index.json").read_text())
        for split in ("train", "valid", "test"):
            with (self.output / split / f"{STEM}.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), len(self.rows))
            for before, after in zip(self.rows, rows):
                for key in before.keys() - {"history_item_sid", "item_sid"}:
                    self.assertEqual(before[key], after[key])
                self.assertEqual(after["item_sid"], "".join(index[before["item_id"]]))
                self.assertEqual(ast.literal_eval(after["history_item_sid"]),
                                 ["".join(index[str(i)]) for i in ast.literal_eval(before["history_item_id"])])
        expected_info = "".join(f'{"".join(index[k])}\t{item["title"]}\t{k}\n' for k, item in self.items.items())
        self.assertEqual((self.output / "info" / f"{STEM}.txt").read_text(), expected_info)
        manifest = json.loads((self.output / MANIFEST).read_text())
        self.assertEqual(manifest["rows"], {s: 4 for s in ("train", "valid", "test")})
        with contextlib.redirect_stdout(io.StringIO()):
            check_prepared(self.output)

    def test_rejects_mismatched_source_sid_without_publishing_partial_data(self):
        path = self.source / "valid" / f"{STEM}.csv"
        path.write_text(path.read_text().replace("<a_10>", "<a_99>"))
        with self.assertRaisesRegex(ValueError, "原 CSV 的 SID"):
            self.run_prepare()
        self.assertFalse(self.output.exists())

    def test_rejects_missing_split_before_creating_output(self):
        (self.source / "test" / f"{STEM}.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_prepare()
        self.assertFalse(self.output.exists())

    def test_never_overwrites_source_or_existing_variant(self):
        with self.assertRaisesRegex(ValueError, "输出目录已存在"):
            prepare(self.source, self.source, self.codes_path)
        self.run_prepare()
        before = sha256(self.output / MANIFEST)
        with self.assertRaisesRegex(ValueError, "输出目录已存在"):
            self.run_prepare()
        self.assertEqual(sha256(self.output / MANIFEST), before)

    def test_detects_tampered_prepared_data(self):
        self.run_prepare()
        (self.output / "train" / f"{STEM}.csv").write_text("wrong CSV")
        with self.assertRaisesRegex(ValueError, "清单不一致"):
            check_prepared(self.output)

    def test_rejects_wrong_code_shape_range_dtype_and_item_ids(self):
        for bad in (self.codes.T, self.codes.astype(float), self.codes - 1, self.codes + 1):
            with self.subTest(codes=bad), self.assertRaises(ValueError):
                build_index(bad, self.items)
        with self.assertRaises(ValueError):
            build_index(self.codes, {"a": {}, "b": {}, "c": {}, "d": {}})


if __name__ == "__main__":
    unittest.main()
