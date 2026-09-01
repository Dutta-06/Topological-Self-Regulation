"""Timeseries reference architecture, shared by every bench/train_ts_*.py
script — the timeseries analogue of bench/models.py.

TCN: dilated residual Conv1d/BatchNorm1d blocks, operating in (B, C, L).
Verified (see tests_tsrx/test_timeseries.py) to need ZERO changes to the
tsrx engine: groups.py already treats Conv1d as a first-class producer/
consumer, and on (B, C, L) the channel axis IS axis 1, so cost.py's
`_spatial_size` and saliency.py's `ActivationStats` reduction — both of
which assume axis 1 is channels — are already correct. This is exactly
why Stage 1 uses BatchNorm1d, not LayerNorm: LayerNorm's statistics span
the candidate columns, breaking condition (N) (Lemma 5.6) — see
tsrx/sense/candidates.py's `UnsupportedGroupError`. That fix is Stage 2,
out of scope here.

The loader (data/ltsf.py) yields (B, L, C); the transpose to (B, C, L)
happens at the model boundary, not in the loader, so the loader's tensor
layout matches every other consumer of it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _TCNBlock(nn.Module):
    """Two dilated convs with BatchNorm1d + a 1x1 projection shortcut.
    padding=dilation with kernel=3 preserves sequence length exactly."""

    def __init__(self, c_in: int, c_out: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, 3, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, 3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.downsample = nn.Conv1d(c_in, c_out, 1, bias=False) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y + self.downsample(x))


class TCNForecaster(nn.Module):
    def __init__(self, n_vars: int, pred_len: int, hidden: int = 64, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.n_vars = n_vars
        self.pred_len = pred_len
        blocks = []
        c_in = n_vars
        for d in dilations:
            blocks.append(_TCNBlock(c_in, hidden, d))
            c_in = hidden
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden, pred_len * n_vars)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C) from the loader -> (B, C, L) for Conv1d
        x = x.transpose(1, 2)
        y = self.blocks(x)             # (B, hidden, L)
        # Last timestep, not a global average: forecasting is dominated by
        # recency, and mean-pooling the whole window destroys exactly the
        # positional information (what happened most recently) the head
        # needs. Symmetric ('same') padding still lets earlier positions see
        # the whole window through the dilated receptive field, so the last
        # position is not merely a local view — it's the point in the
        # sequence closest to the forecast horizon.
        last = y[:, :, -1]              # (B, hidden)
        out = self.head(last)           # (B, pred_len * n_vars)
        return out.view(-1, self.pred_len, self.n_vars)


class RevIN(nn.Module):
    """Reversible instance normalization (Kim et al. 2022), the standard
    LTSF front-end: z-score each (sample, variate) series over TIME, then
    de-normalize the forecast with the same statistics.

    Safe for TSR-X, and the reason matters: RevIN reduces over the time
    axis per (batch, variate) and holds NO parameters on the reallocated
    channel axis. Condition (N) is a statement about the channel axis the
    candidates live on, so RevIN cannot couple candidates into real units
    the way LayerNorm does. `tests_tsrx/test_c3_and_ci.py` asserts the
    dormancy invariant with RevIN active rather than taking this on faith.
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def normalize(self, x: torch.Tensor):
        # x: (B, L, C) -> stats over L, per (B, C)
        mean = x.mean(dim=1, keepdim=True)
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps)
        return (x - mean) / std, mean, std

    def denormalize(self, y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        return y * std + mean


class TCNForecasterCI(nn.Module):
    """Channel-independent TCN: every variate passes through the SAME conv
    body, so the body is shared and the head is Linear(hidden, pred_len)
    rather than Linear(hidden, pred_len * n_vars).

    This exists because the channel-MIXING TCNForecaster above is an
    unusable testbed for capacity reallocation. Measured on the first full
    sweep, its head is 58-99% of all parameters (traffic_h720: 99.2%), so
    the conv body TSR-X can actually index was 0.8-41% of the model and the
    reported "15% reduction" was almost entirely one knob — shrinking the
    residual-stream group that feeds the head. Channel-independence flips
    that to 93.3% body at h=96 / 65.1% at h=720, and makes the parameter
    count independent of n_vars so one architecture is comparable across
    every dataset. It is also the prevailing LTSF convention (PatchTST,
    DLinear), not an invention for this project.
    """

    def __init__(self, n_vars: int, pred_len: int, hidden: int = 64,
                 dilations=(1, 2, 4, 8), use_revin: bool = True):
        super().__init__()
        self.n_vars = n_vars
        self.pred_len = pred_len
        self.revin = RevIN() if use_revin else None
        blocks = []
        c_in = 1                      # one variate at a time
        for d in dilations:
            blocks.append(_TCNBlock(c_in, hidden, d))
            c_in = hidden
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        B, L, C = x.shape
        if self.revin is not None:
            x, mean, std = self.revin.normalize(x)
        # (B, L, C) -> (B*C, 1, L): each variate is its own sequence
        z = x.permute(0, 2, 1).reshape(B * C, 1, L)
        y = self.blocks(z)                       # (B*C, hidden, L)
        out = self.head(y[:, :, -1])             # (B*C, pred_len)
        out = out.reshape(B, C, self.pred_len).permute(0, 2, 1)   # (B, pred_len, C)
        if self.revin is not None:
            out = self.revin.denormalize(out, mean, std)
        return out


def build_ts_model(arch: str, n_vars: int, pred_len: int, hidden: int = 64,
                    use_revin: bool = True, seq_len: int = 96, d_model: int = 128,
                    d_ff: int = 256, n_heads: int = 8, n_blocks: int = 3,
                    patch_len: int = 16, stride: int = 8) -> nn.Module:
    if arch == "tcn":
        return TCNForecaster(n_vars, pred_len, hidden=hidden)
    if arch == "tcn_ci":
        return TCNForecasterCI(n_vars, pred_len, hidden=hidden, use_revin=use_revin)
    if arch == "patchtst":
        from bench.patchtst import build_patchtst
        return build_patchtst(n_vars, seq_len, pred_len, d_model=d_model, d_ff=d_ff,
                               n_heads=n_heads, n_blocks=n_blocks, patch_len=patch_len,
                               stride=stride, use_revin=use_revin)
    raise ValueError(f"unknown timeseries arch {arch!r}")


def ts_model_kwargs(args) -> dict:
    """Pull the architecture hyperparameters out of an argparse namespace.

    Kept in one place so the reference / TSR-X / C2 / C3 arms cannot drift
    apart -- a C2 control built at a different d_model than the discovery
    run is not a control at all.
    """
    return dict(
        hidden=getattr(args, "hidden", 64),
        use_revin=getattr(args, "use_revin", True),
        seq_len=getattr(args, "seq_len", 96),
        d_model=getattr(args, "d_model", 128),
        d_ff=getattr(args, "d_ff", 256),
        n_heads=getattr(args, "n_heads", 8),
        n_blocks=getattr(args, "n_blocks", 3),
        patch_len=getattr(args, "patch_len", 16),
        stride=getattr(args, "stride", 8),
    )


def describe(model: nn.Module, seq_len: int, n_vars: int) -> dict:
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lengths = {}
    hooks, mods = [], dict(model.named_modules())
    for name, mod in mods.items():
        if name.startswith("blocks.") and name.count(".") == 1:
            hooks.append(mod.register_forward_hook(
                lambda m, i, o, n=name: lengths.__setitem__(n, tuple(o.shape[1:]))))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        dev = next(model.parameters()).device
        model(torch.zeros(1, seq_len, n_vars, device=dev))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return {"params": params, "block_shapes": lengths}
