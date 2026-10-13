"""Three parallel, independently quantized branches with one joint decoder."""
import math
import torch
from torch import nn
import torch.nn.functional as F

from rq.models.layers import MLPLayers

DEFAULT_LAYERS = (2048, 1024, 512, 256, 128)
DEFAULT_LATENT_DIM = 32


@torch.no_grad()
def balanced_transport(distances, epsilon, iterations):
    """Batchwise equal-mass OT; hard argmax need not have equal counts.

    Accept (..., samples, codes). Use log-space Sinkhorn to avoid underflow.
    Only assignment costs are rescaled, never the input history embeddings.
    """
    costs = distances.detach().float()
    lo = costs.amin(dim=(-2, -1), keepdim=True)
    hi = costs.amax(dim=(-2, -1), keepdim=True)
    costs = (costs - (hi + lo) / 2) / ((hi - lo) / 2).clamp_min(1e-5)
    log_q = -costs / epsilon
    samples, codes = costs.shape[-2:]
    for _ in range(iterations):
        log_q = log_q - torch.logsumexp(log_q, dim=-1, keepdim=True)
        log_q = log_q - torch.logsumexp(log_q, dim=-2, keepdim=True)
        log_q = log_q + torch.log(costs.new_tensor(samples / codes))
    return log_q.exp()


class ParallelVQ(nn.Module):
    def __init__(self, input_dim=2560, codebook_size=256, latent_dim=DEFAULT_LATENT_DIM,
                 layers=DEFAULT_LAYERS, commitment=0.25,
                 balance_epsilon=0.0, balance_iterations=50, num_slots=3, input_scale=1.0):
        super().__init__()
        layers = list(layers)
        if not layers or any(not isinstance(n, int) or n <= 0 for n in layers):
            raise ValueError('MLP layers must be positive integers')
        self.options = dict(input_dim=input_dim, codebook_size=codebook_size, latent_dim=latent_dim,
                            layers=layers, commitment=commitment)
        self.commitment = commitment
        if not math.isfinite(input_scale) or input_scale <= 0:
            raise ValueError('Input scale must be finite and positive')
        self.input_scale = float(input_scale)
        # Legacy checkpoints keep their original options and exact computation.
        if self.input_scale != 1.0:
            self.options['input_scale'] = self.input_scale
        self.latent_dim = latent_dim
        if num_slots not in (2, 3):
            raise ValueError('Supported latent slot counts are 2 and 3')
        self.num_slots = num_slots
        if num_slots != 3:
            self.options['num_slots'] = num_slots
        if balance_epsilon < 0 or balance_iterations <= 0:
            raise ValueError('Balance epsilon must be nonnegative and iterations positive')
        self.balance_epsilon = balance_epsilon
        self.balance_iterations = balance_iterations
        if balance_epsilon:
            self.options.update(balance_epsilon=balance_epsilon,
                                balance_iterations=balance_iterations)
        # Reuse RQ-VAE's MLP building blocks, ReLU, Xavier initialization,
        # zero biases, dropout=0 and bn=False. The requested default goes
        # directly from width 128 to 96, split into three 32-dim branches.
        dims = [input_dim] + layers + [num_slots * latent_dim]
        self.encoder = MLPLayers(dims, dropout=0.0, bn=False)
        self.codebooks = nn.Parameter(torch.empty(num_slots, codebook_size, latent_dim))
        nn.init.normal_(self.codebooks, std=0.02)
        self.decoder = MLPLayers(dims[::-1], dropout=0.0, bn=False)

    def quantize(self, x, balanced=False):
        z = self.encoder(x * self.input_scale if self.input_scale != 1.0 else x).reshape(-1, self.num_slots, self.latent_dim)
        # Each branch sees its own encoder projection; no residual subtraction.
        distances = z.square().sum(-1, keepdim=True) + self.codebooks.square().sum(-1)[None] - 2 * torch.einsum('bkd,kcd->bkc', z, self.codebooks)
        if balanced and self.balance_epsilon:
            assignment = balanced_transport(distances.transpose(0, 1),
                                            self.balance_epsilon, self.balance_iterations)
            codes = assignment.argmax(-1).transpose(0, 1)
        else:
            codes = distances.argmin(-1)
        q = torch.stack([self.codebooks[k, codes[:, k]] for k in range(self.num_slots)], dim=1)
        return z, q, codes

    def forward(self, x):
        z, q, codes = self.quantize(x, balanced=self.training)
        reconstruction = self.decoder((z + (q - z).detach()).flatten(1))
        target = x * self.input_scale if self.input_scale != 1.0 else x
        rec = F.mse_loss(reconstruction, target)
        codebook = F.mse_loss(q, z.detach())
        commitment = F.mse_loss(z, q.detach())
        return rec + codebook + self.commitment * commitment, dict(reconstruction=rec, codebook=codebook, commitment=commitment), codes

    @torch.no_grad()
    def initialize_from_train(self, x, generator):
        z = self.encoder(x * self.input_scale if self.input_scale != 1.0 else x).reshape(-1, self.num_slots, self.latent_dim)
        for k in range(self.num_slots):
            choice = torch.randint(len(z), (self.codebooks.shape[1],), generator=generator)
            self.codebooks[k].copy_(z[choice.to(z.device), k])


def code_stats(codes, size):
    stats = []
    for k in range(codes.shape[1]):
        counts = torch.bincount(codes[:, k].cpu(), minlength=size).float()
        p = counts / counts.sum()
        stats.append(dict(used=int((counts > 0).sum()), dead=int((counts == 0).sum()),
                          perplexity=float(torch.exp(-(p[p > 0] * p[p > 0].log()).sum()))))
    return stats
