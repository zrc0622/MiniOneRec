"""Compare different RL checkpoints under the same data and search protocol."""
import argparse
import json
from pathlib import Path


def compare(run_dir, checkpoint_label='final', sft_reference=None):
    root = Path(run_dir) / 'eval'
    rows, identities = [], {}
    for split in ('valid', 'test'):
        for reward in ('binary', 'sid_dense'):
            pointer = root / reward / f'latest_{split}.txt'
            if not pointer.exists():
                continue
            result_dir = Path(pointer.read_text().strip())
            result = json.loads((result_dir / 'merged.json').read_text())
            meta = result['metadata']
            # Model weights deliberately differ. Data, row coverage and decoding
            # settings must not differ within a split.
            keys = ('data_sha256', 'index_sha256', 'examples', 'scheme',
                    'latent_count', 'latent_beams', 'item_beams', 'score_batch')
            identity = {key: meta[key] for key in keys}
            if split in identities and identities[split] != identity:
                raise ValueError('RL comparison requires identical data and search settings')
            identities[split] = identity
            rows.append(dict(experiment=reward, split=split, model=meta['model'],
                             result=str(result_dir / 'merged.json'), **result['metrics']))
    if not rows:
        raise ValueError('No completed RL evaluations')
    if sft_reference and 'test' in identities:
        reference = json.loads(Path(sft_reference).read_text())
        identity = {key: reference['metadata'][key] for key in identities['test']}
        if identity != identities['test']:
            raise ValueError('SFT reference uses different data or search settings')
        rows.insert(0, dict(experiment='sft', split='test', model=reference['metadata']['model'],
                            result=str(Path(sft_reference).resolve()), **reference['metrics']))
    root.mkdir(parents=True, exist_ok=True)
    (root / 'comparison.json').write_text(json.dumps(dict(identities=identities, results=rows), indent=2) + '\n')
    fields = ('experiment', 'split', 'HR@1', 'HR@10', 'HR@50', 'NDCG@10', 'NDCG@50',
              'mean_unique_candidates', 'fraction_with_50_candidates')
    lines = [f'# RL {checkpoint_label}-checkpoint comparison', '',
             'All runs use deterministic hierarchical_max (4 latent x 50 item beams).',
             ('These are final checkpoints at the fixed training budget, not validation-selected checkpoints.'
              if checkpoint_label == 'final' else 'Intermediate checkpoint evaluation; training remains in progress.'), '',
             '| ' + ' | '.join(fields) + ' |', '| ' + ' | '.join(['---'] * len(fields)) + ' |']
    for row in rows:
        lines.append('| ' + ' | '.join(str(row[k]) if isinstance(row[k], str) else f'{row[k]:.6f}'
                                      for k in fields) + ' |')
    (root / 'comparison.md').write_text('\n'.join(lines) + '\n')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    parser.add_argument('--checkpoint-label', default='final')
    parser.add_argument('--sft-reference')
    args = parser.parse_args()
    compare(args.run_dir, args.checkpoint_label, args.sft_reference)


if __name__ == '__main__':
    main()
