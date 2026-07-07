"""
TabNet (Arik & Pfister 2019, "TabNet: Attentive Interpretable Tabular
Learning") — sequential decision steps, each attending to a sparse subset of
features (via a learned, sparsemax-projected mask) before a GLU-based
"feature transformer" produces that step's contribution to the output. A
"prior scale" discourages (but does not forbid, via `gamma`) reusing a
feature across steps.

Simplification vs. the paper: each step's feature transformer is fully
independent (own GLU-block weights) rather than the paper's design of 2
GLU blocks shared across all steps + 2 step-specific ones. This drops a
parameter-sharing efficiency trick but keeps every mechanism that actually
defines TabNet as an architecture (sequential sparse attention, prior-scale
feature-reuse penalty, decision aggregation) — sizing (n_d/n_a/n_steps)
still controls param count the same way. Known from the literature (and
part of why it's included here as a baseline, not a target) that TabNet's
capacity doesn't scale as cleanly as ResNet-MLP/FT-Transformer's — pushing
n_d/n_a up shows diminishing or negative returns past ~1M params.

Categorical columns are embedded and concatenated with numeric features
(same flat representation as MLP/ResNet-MLP), then BatchNorm-normalized —
this fixed-width vector is what the attention masks operate over.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import CatEmbedding


def sparsemax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Martins & Astudillo (2016) sparsemax projection — like softmax but
    produces exactly-zero probabilities for low-scoring entries, which is
    what makes TabNet's feature masks genuinely sparse (interpretable)
    rather than just peaked."""
    x_sorted, _ = torch.sort(x, dim=dim, descending=True)
    cumsum = torch.cumsum(x_sorted, dim=dim)
    r = torch.arange(1, x.shape[dim] + 1, device=x.device, dtype=x.dtype)
    shape = [1] * x.dim()
    shape[dim] = -1
    r = r.view(*shape)
    support = (1 + r * x_sorted) > cumsum
    k = support.sum(dim=dim, keepdim=True).clamp(min=1).to(x.dtype)
    cumsum_k = torch.gather(cumsum, dim, (k.long() - 1))
    tau = (cumsum_k - 1) / k
    return torch.clamp(x - tau, min=0)


class GLUBlock(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.fc = nn.Linear(d_in, 2 * d_out)
        self.bn = nn.BatchNorm1d(2 * d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.glu(self.bn(self.fc(x)), dim=-1)


class FeatureTransformer(nn.Module):
    def __init__(self, d_in: int, d_feat: int, dropout: float):
        super().__init__()
        self.block1 = GLUBlock(d_in, d_feat)
        self.block2 = GLUBlock(d_feat, d_feat)
        self.dropout = nn.Dropout(dropout)
        self.scale = 0.5 ** 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = (h + self.block2(h)) * self.scale
        return self.dropout(h)


class AttentiveTransformer(nn.Module):
    def __init__(self, d_a: int, num_features: int):
        super().__init__()
        self.fc = nn.Linear(d_a, num_features)
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, a: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        return sparsemax(self.bn(self.fc(a)) * prior, dim=-1)


class TabNet(nn.Module):
    def __init__(
        self,
        num_numeric: int,
        cat_cardinalities: List[int],
        num_out: int,
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 4,
        gamma: float = 1.5,
        cat_emb_dim: int = 8,
        dropout: float = 0.1,
        lambda_sparse: float = 1e-3,
    ):
        super().__init__()
        self.cat_embedding = CatEmbedding(cat_cardinalities, cat_emb_dim)
        num_features = num_numeric + self.cat_embedding.output_dim
        self.input_bn = nn.BatchNorm1d(num_features)

        self.n_d = n_d
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambda_sparse = lambda_sparse
        self.last_sparsity_loss = torch.tensor(0.0)

        d_feat = n_d + n_a
        self.initial_transform = FeatureTransformer(num_features, d_feat, dropout)
        self.step_transforms = nn.ModuleList(
            [FeatureTransformer(num_features, d_feat, dropout) for _ in range(n_steps)]
        )
        self.attentive_transforms = nn.ModuleList(
            [AttentiveTransformer(n_a, num_features) for _ in range(n_steps)]
        )
        self.head = nn.Linear(n_d, num_out)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_num, self.cat_embedding(x_cat)], dim=-1)
        x = self.input_bn(x)
        b, num_features = x.shape

        prior = torch.ones_like(x)
        agg_decision = x.new_zeros(b, self.n_d)
        entropy = x.new_zeros(())

        a = self.initial_transform(x)[:, self.n_d:]
        for step in range(self.n_steps):
            mask = self.attentive_transforms[step](a, prior)
            prior = prior * (self.gamma - mask)
            entropy = entropy - (mask * torch.log(mask + 1e-15)).sum(dim=-1).mean()

            masked_x = mask * x
            out = self.step_transforms[step](masked_x)
            d, a = out[:, :self.n_d], out[:, self.n_d:]
            agg_decision = agg_decision + F.relu(d)

        self.last_sparsity_loss = self.lambda_sparse * entropy / max(self.n_steps, 1)
        return self.head(agg_decision)
