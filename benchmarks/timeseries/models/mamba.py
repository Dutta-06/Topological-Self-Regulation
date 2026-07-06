"""
Mamba-style selective state-space block (Gu & Dao, 2023) — pure-PyTorch reference.

Implements the paper's selective-SSM recurrence directly:
    A: (d_inner, d_state), input-independent (learned, negative for stability)
    B, C, delta: input-DEPENDENT (selective) — the key departure from S4/S4D
    discretization: Abar = exp(delta * A), Bbar = delta * B  (zero-order hold)
    h_t = Abar_t * h_{t-1} + Bbar_t * x_t ;  y_t = C_t . h_t + D * x_t

This is a sequential (for-loop over time) scan rather than the paper's
hardware-aware parallel/chunked CUDA kernel — correct and faithful to the
recurrence, just not hardware-optimized. Fine at the sequence lengths used
here (~100-500 steps); would not scale to the paper's long-context regime
without the real kernel (mamba_ssm package).

Block structure (matches the paper): input projection -> depthwise causal
conv -> SiLU -> selective SSM -> gate with a parallel SiLU branch -> output
projection, wrapped in a residual + pre-norm, stacked num_layers times.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SelectiveSSM(nn.Module):
    def __init__(self, d_inner: int, d_state: int = 16, dt_rank: int = None):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        dt_rank = dt_rank or max(1, d_inner // 16)

        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))  # A = -exp(A_log), always negative
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_inner) -> (B, L, d_inner)."""
        b, L, _ = x.shape
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        dt_rank = dbl.shape[-1] - 2 * self.d_state
        delta_in, B_sel, C_sel = torch.split(dbl, [dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta_in))  # (B, L, d_inner)

        Abar = torch.exp(delta.unsqueeze(-1) * A)          # (B, L, d_inner, d_state)
        Bbar = delta.unsqueeze(-1) * B_sel.unsqueeze(2)     # (B, L, d_inner, d_state)

        h = torch.zeros(b, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = Abar[:, t] * h + Bbar[:, t] * x[:, t].unsqueeze(-1)
            y_t = (h * C_sel[:, t].unsqueeze(1)).sum(-1)  # (B, d_inner)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B, L, d_inner)
        return y + x * self.D


class _MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, conv_kernel: int = 4):
        super().__init__()
        d_inner = expand * d_model
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner)
        self.conv = nn.Conv1d(
            d_inner, d_inner, kernel_size=conv_kernel,
            padding=conv_kernel - 1, groups=d_inner,
        )
        self.conv_kernel = conv_kernel
        self.ssm = _SelectiveSSM(d_inner, d_state)
        self.out_proj = nn.Linear(d_inner, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) -> (B, L, d_model), residual applied internally."""
        residual = x
        x = self.norm(x)
        x_and_gate = self.in_proj(x)  # (B, L, 2*d_inner)
        x_in, gate = x_and_gate.chunk(2, dim=-1)

        x_conv = self.conv(x_in.transpose(1, 2))[:, :, :x_in.shape[1]].transpose(1, 2)
        x_conv = F.silu(x_conv)

        y = self.ssm(x_conv)
        y = y * F.silu(gate)
        return residual + self.out_proj(y)


class MambaEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        d_state: int = 16,
        num_layers: int = 3,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.blocks = nn.ModuleList([
            _MambaBlock(d_model, d_state, expand) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm_out = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, C) -> (B, L, d_model)."""
        h = self.input_proj(x)
        for block in self.blocks:
            h = self.dropout(block(h))
        return self.norm_out(h)


class MambaForecaster(nn.Module):
    def __init__(self, input_size: int, pred_len: int, d_model: int = 64, d_state: int = 16,
                 num_layers: int = 3, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = MambaEncoder(input_size, d_model, d_state, num_layers, expand, dropout)
        self.pred_len = pred_len
        self.input_size = input_size
        self.head = nn.Linear(d_model, pred_len * input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        out = self.head(h[:, -1, :])
        return out.view(-1, self.pred_len, self.input_size)


class MambaClassifier(nn.Module):
    def __init__(self, input_size: int, num_classes: int, d_model: int = 64,
                 d_state: int = 16, num_layers: int = 3, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = MambaEncoder(input_size, d_model, d_state, num_layers, expand, dropout)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.head(h.mean(dim=1))
