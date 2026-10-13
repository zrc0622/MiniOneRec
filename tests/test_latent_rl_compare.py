import json
from pathlib import Path
import tempfile
import unittest

from latent.rl_compare import compare


class RLCompareTests(unittest.TestCase):
    def test_different_models_same_protocol_and_reject_mismatched_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for reward in ('binary', 'sid_dense'):
                d = root / 'eval' / reward / 'valid.test'; d.mkdir(parents=True)
                meta = dict(model=reward, model_hashes={'weights': reward}, data_sha256='same',
                    index_sha256='same', examples=2, scheme='hierarchical_max', latent_count=4,
                    latent_beams=4, item_beams=50, score_batch=8)
                metrics = {k: .1 for k in ('HR@1', 'HR@10', 'HR@50', 'NDCG@10', 'NDCG@50',
                            'mean_unique_candidates', 'fraction_with_50_candidates')}
                path = d / 'merged.json'; path.write_text(json.dumps(dict(metadata=meta, metrics=metrics)))
                (d.parent / 'latest_valid.txt').write_text(str(d))
                paths.append(path)
            rows = compare(root)
            self.assertEqual(len(rows), 2)
            self.assertTrue((root / 'eval/comparison.md').exists())
            data = json.loads(paths[1].read_text())
            data['metadata']['data_sha256'] = 'different'
            paths[1].write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                compare(root)
