"""Bounded, matched VQ scale diagnostics; never exports production labels."""
import argparse
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from latent.vq import ParallelVQ, code_stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--seed', type=int, default=42, help='Model seed; sampled rows and batches stay fixed')
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    device = torch.device('cuda')
    rng = np.random.default_rng(42)
    arrays = {s: np.load(Path(args.source) / 'embeddings' / f'{s}.npy', mmap_mode='r')
              for s in ('train', 'valid')}
    ids = {s: rng.choice(len(arrays[s]), n, replace=False)
           for s, n in [('train', 8192), ('valid', 2048)]}
    np.savez(out / 'sample_ids.npz', **ids)
    data = {s: torch.from_numpy(np.array(arrays[s][ids[s]], copy=True)).to(device)
            for s in arrays}
    d = data['train'].shape[1]
    batches = np.random.default_rng(123).integers(0, len(data['train']), (args.steps, 512))

    def evaluate(model, scale):
        model.eval()
        components = dict(reconstruction=0., codebook=0., commitment=0.)
        cs, encoders, quantized, predictions = [], [], [], []
        with torch.no_grad():
            for x in data['valid'].split(512):
                _, parts, codes = model(x * scale)
                z, q, _ = model.quantize(x * scale)
                pred = model.decoder(q.flatten(1)) / scale
                cs.append(codes.cpu())
                encoders.append(z.cpu())
                quantized.append(q.cpu())
                predictions.append(pred.cpu())
                for k in components:
                    components[k] += float(parts[k]) * len(x) / len(data['valid'])
        codes = torch.cat(cs)
        z, q, pred = map(torch.cat, (encoders, quantized, predictions))
        mu = data['train'].mean(0)
        baseline = float((data['valid'] - mu).square().mean())
        rec = components['reconstruction'] / scale**2
        return dict(loss_components=components, unit_scale_reconstruction=rec,
                    reconstruction_over_train_mean_baseline=rec / baseline,
                    code_statistics=code_stats(codes, 256),
                    unique_triples=len(torch.unique(codes, dim=0)),
                    encoder_rms=float(z.square().mean().sqrt()),
                    encoder_between_sample_rms=float(z.var(0, unbiased=False).mean().sqrt()),
                    selected_code_rms=float(q.square().mean().sqrt()),
                    decoded_between_sample_rms=float(pred.var(0, unbiased=False).mean().sqrt()))

    arms = [('unit_original', 1., 1., .001),
            ('scaled_sqrt_dimension', math.sqrt(d), 1., .001),
            ('unit_reconstruction_weight_dimension', 1., float(d), .001),
            ('unit_lr_div10', 1., 1., .0001)]
    results = dict(source=str(Path(args.source).resolve()), seed=args.seed, sample_seed=42,
                   train_rows=8192, valid_rows=2048, batch=512, steps=args.steps,
                   same_directions_and_batches=True, production_labels_exported=False, arms={})
    saved = torch.load(Path(args.source) / 'labels/vq.pt', map_location='cpu', weights_only=True)
    fitted = ParallelVQ(**saved['options']).to(device)
    fitted.load_state_dict(saved['state'])
    results['original_full_data_checkpoint'] = evaluate(fitted, 1.)
    del fitted
    for name, scale, rec_weight, lr in arms:
        torch.manual_seed(args.seed)
        model = ParallelVQ(input_dim=d).to(device)
        gen = torch.Generator().manual_seed(args.seed)
        init_order = torch.randperm(len(data['train']), generator=gen)
        model.initialize_from_train(data['train'][init_order.to(device)] * scale, gen)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
        history = [dict(step=0, **evaluate(model, scale))]
        best_rec, best = float('inf'), None
        start = time.monotonic()
        for step, batch_ids in enumerate(batches, 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            x = data['train'][torch.as_tensor(batch_ids, device=device)] * scale
            _, parts, _ = model(x)
            loss = rec_weight * parts['reconstruction'] + parts['codebook'] + .25 * parts['commitment']
            if not torch.isfinite(loss):
                raise ValueError(f'Nonfinite loss in {name}, step{step}')
            loss.backward()
            if step == 1:
                before = next(model.encoder.parameters()).detach().clone()
                first = dict(encoder_first_weight_gradient_rms=float(next(model.encoder.parameters()).grad.square().mean().sqrt()),
                             loss_components={k: float(v) for k, v in parts.items()})
            optimizer.step()
            if step == 1:
                first['encoder_first_weight_update_rms'] = float((next(model.encoder.parameters()).detach() - before).square().mean().sqrt())
            if step == 1 or step % 50 == 0 or step == args.steps:
                row = dict(step=step, **evaluate(model, scale))
                history.append(row)
                if row['unit_scale_reconstruction'] < best_rec:
                    best_rec, best = row['unit_scale_reconstruction'], copy.deepcopy(row)
                print(json.dumps(dict(arm=name, step=step, used=[s['used'] for s in row['code_statistics']],
                                      rec_ratio=row['reconstruction_over_train_mean_baseline'])), flush=True)
        results['arms'][name] = dict(scale=scale, reconstruction_weight=rec_weight, lr=lr,
                                    seconds=time.monotonic() - start, first_step=first,
                                    history=history, best_observed_validation=best)
        (out / 'results.json').write_text(json.dumps(results, indent=2))
        del model, optimizer
    print('diagnostic_complete', flush=True)


if __name__ == '__main__':
    main()
