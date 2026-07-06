"""
ETTh1 / ETTh2 (Electricity Transformer Temperature, hourly) forecasting loaders.

Both series share the same 7-channel schema (6 covariates + OT target) and the
same canonical split convention (12 months train / 4 val / 4 test). This module
generalizes data/etth1.py to also cover ETTh2 without duplicating the window/
split logic.

Each batch yields (x, y):
    x: (batch, seq_len, 7)        — a window of all 7 channels
    y: (batch, pred_len, 7)       — the next pred_len steps, all 7 channels

Multivariate multi-horizon target — matches the standard ETT forecasting
protocol used by Informer/Autoformer/PatchTST (predict the full future window
across all channels, not a single scalar), so results are comparable to
published numbers at the same (seq_len, pred_len) operating point.
"""

import os
from typing import Tuple
from urllib.request import urlretrieve

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

_URLS = {
    "etth1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "etth2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
}
# Matches data/etth1.py's existing cache filename (mixed case) so both loaders
# share one cached download instead of duplicating it under a different name.
_FILENAMES = {"etth1": "ETTh1.csv", "etth2": "ETTh2.csv"}

# Canonical ETT split boundaries in hours (months * 30 days * 24 hours).
_TRAIN_HOURS = 12 * 30 * 24
_VAL_HOURS = 4 * 30 * 24
_TEST_HOURS = 4 * 30 * 24


def _download(name: str, root: str) -> str:
    os.makedirs(root, exist_ok=True)
    filename = _FILENAMES[name]
    csv_path = os.path.join(root, filename)
    if not os.path.exists(csv_path):
        print(f"Downloading {filename} to {csv_path} ...")
        urlretrieve(_URLS[name], csv_path)
    return csv_path


def _load_array(name: str, root: str) -> np.ndarray:
    csv_path = _download(name, root)
    rows = []
    with open(csv_path, "r") as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            rows.append([float(v) for v in parts[1:8]])
    return np.asarray(rows, dtype=np.float32)  # (T, 7)


class _ETTWindowDataset(Dataset):
    def __init__(self, series: np.ndarray, seq_len: int, pred_len: int):
        self.series = series
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n = len(series) - seq_len - pred_len + 1

    def __len__(self) -> int:
        return max(self.n, 0)

    def __getitem__(self, idx: int):
        x = self.series[idx: idx + self.seq_len]                                    # (seq_len, C)
        y = self.series[idx + self.seq_len: idx + self.seq_len + self.pred_len]      # (pred_len, C)
        return torch.from_numpy(x), torch.from_numpy(y)


def get_ett_loaders(
    name: str = "etth1",
    batch_size: int = 128,
    seq_len: int = 96,
    pred_len: int = 1,
    root: str = "./data",
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train, val, test) DataLoaders for ETTh1 or ETTh2 forecasting.

    Args:
        name: "etth1" or "etth2".
        batch_size: Batch size for all three loaders.
        seq_len: Input window length.
        pred_len: Forecast horizon (steps ahead of the window's end).
        root: Directory to cache the CSV under.
        num_workers: DataLoader workers.
    """
    name = name.lower()
    if name not in _URLS:
        raise ValueError(f"Unknown ETT dataset '{name}', expected one of {list(_URLS)}")

    series = _load_array(name, root)

    train_end = _TRAIN_HOURS
    val_end = _TRAIN_HOURS + _VAL_HOURS
    test_end = min(_TRAIN_HOURS + _VAL_HOURS + _TEST_HOURS, len(series))

    train_raw = series[:train_end]
    mean = train_raw.mean(axis=0, keepdims=True)
    std = train_raw.std(axis=0, keepdims=True) + 1e-8
    norm = (series - mean) / std

    train_series = norm[:train_end]
    val_series = norm[train_end - seq_len: val_end]
    test_series = norm[val_end - seq_len: test_end]

    pin = torch.cuda.is_available()

    def _loader(arr, shuffle):
        ds = _ETTWindowDataset(arr, seq_len, pred_len)
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin, drop_last=False,
        )

    return _loader(train_series, True), _loader(val_series, False), _loader(test_series, False)
