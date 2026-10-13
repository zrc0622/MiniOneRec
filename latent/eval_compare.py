"""Build a cumulative comparison after each completed evaluation scheme."""
import argparse
import csv
import json
from pathlib import Path

SCHEMES = ('joint_max', 'hierarchical_max', 'hierarchical_sum', 'fixed_latent')
FIELDS = ('scheme', 'examples', 'HR@1', 'HR@10', 'HR@50', 'NDCG@50',
          'mean_unique_candidates', 'mean_unique_before_top50',
          'fraction_with_50_candidates', 'mean_valid_paths',
          'mean_unique_latents', 'mean_generation_and_scoring_seconds', 'wall_seconds')


def compare(root):
    root = Path(root)
    rows, reference = [], None
    for scheme in SCHEMES:
        path = root / scheme / 'merged.json'
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        meta = data['metadata']
        identity = {k: meta[k] for k in ('model', 'model_hashes', 'config_sha256',
                                        'data_sha256', 'index_sha256', 'examples')}
        if reference is not None and identity != reference:
            raise ValueError('Cannot compare different models, datasets or row limits')
        reference = identity
        rows.append(dict(scheme=scheme, **data['metrics'],
                         wall_seconds=float((path.parent / 'wall_seconds.txt').read_text())))
    if not rows:
        raise ValueError('No completed evaluation schemes')
    (root / 'comparison.json').write_text(json.dumps(dict(identity=reference, results=rows), indent=2) + '\n')
    with (root / 'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    columns = ('scheme', 'HR@1', 'HR@10', 'HR@50', 'NDCG@50',
               'mean_unique_candidates', 'fraction_with_50_candidates', 'wall_seconds')
    lines = ['# Latent SFT evaluation comparison', '',
             '| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for row in rows:
        lines.append('| ' + ' | '.join(str(row[k]) if isinstance(row[k], str)
                                      else f'{row[k]:.6f}' for k in columns) + ' |')
    lines.extend(['', 'All schemes use the same checkpoint and rows. Search budgets differ.',
                  'hierarchical_sum aggregates only observed unique paths; it is an approximate marginal.',
                  'Missing candidates are reported, not padded. Wall time includes model loading and merging.', ''])
    (root / 'comparison.md').write_text('\n'.join(lines))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    compare(parser.parse_args().root)


if __name__ == '__main__':
    main()
