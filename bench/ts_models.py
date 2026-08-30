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


def build_ts_model(arch: str, n_vars: int, pred_len: int, hidden: int = 64) -> nn.Module:
    if arch == "tcn":
        return TCNForecaster(n_vars, pred_len, hidden=hidden)
    raise ValueError(f"unknown timeseries arch {arch!r}")


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
