"""Three parallel, independently quantized branches with one joint decoder."""
import torch
from torch import nn
import torch.nn.functional as F

from rq.models.layers import MLPLayers

DEFAULT_LAYERS = (2048, 1024, 512, 256, 128)
DEFAULT_LATENT_DIM = 32


class ParallelVQ(nn.Module):
    def __init__(self, input_dim=2560, codebook_size=256, latent_dim=DEFAULT_LATENT_DIM,
                 layers=DEFAULT_LAYERS, commitment=0.25):
        super().__init__()
        layers = list(layers)
        if not layers or any(not isinstance(n, int) or n <= 0 for n in layers):
            raise ValueError('MLP layers must be positive integers')
        self.options = dict(input_dim=input_dim, codebook_size=codebook_size, latent_dim=latent_dim,
                            layers=layers, commitment=commitment)
        self.commitment = commitment
        self.latent_dim = latent_dim
        # Reuse RQ-VAE's MLP building blocks, ReLU, Xavier initialization,
        # zero biases, dropout=0 and bn=False. The requested default goes
        # directly from width 128 to 96, split into three 32-dim branches.
        dims = [input_dim] + layers + [3 * latent_dim]
        self.encoder = MLPLayers(dims, dropout=0.0, bn=False)
        self.codebooks = nn.Parameter(torch.empty(3, codebook_size, latent_dim))
        nn.init.normal_(self.codebooks, std=0.02)
        self.decoder = MLPLayers(dims[::-1], dropout=0.0, bn=False)

    def quantize(self, x):
        z = self.encoder(x).reshape(-1, 3, self.latent_dim)
        # Each branch sees its own encoder projection; no residual subtraction.
        distances = z.square().sum(-1, keepdim=True) + self.codebooks.square().sum(-1)[None] - 2 * torch.einsum('bkd,kcd->bkc', z, self.codebooks)
        codes = distances.argmin(-1)
        q = torch.stack([self.codebooks[k, codes[:, k]] for k in range(3)], dim=1)
        return z, q, codes

    def forward(self, x):
        z, q, codes = self.quantize(x)
        reconstruction = self.decoder((z + (q - z).detach()).flatten(1))
        rec = F.mse_loss(reconstruction, x)
        codebook = F.mse_loss(q, z.detach())
        commitment = F.mse_loss(z, q.detach())
        return rec + codebook + self.commitment * commitment, dict(reconstruction=rec, codebook=codebook, commitment=commitment), codes

    @torch.no_grad()
    def initialize_from_train(self, x, generator):
        z = self.encoder(x).reshape(-1, 3, self.latent_dim)
        for k in range(3):
            choice = torch.randint(len(z), (self.codebooks.shape[1],), generator=generator)
            self.codebooks[k].copy_(z[choice.to(z.device), k])


def code_stats(codes, size):
    stats = []
    for k in range(3):
        counts = torch.bincount(codes[:, k].cpu(), minlength=size).float()
        p = counts / counts.sum()
        stats.append(dict(used=int((counts > 0).sum()), dead=int((counts == 0).sum()),
                          perplexity=float(torch.exp(-(p[p > 0] * p[p > 0].log()).sum()))))
    return stats
