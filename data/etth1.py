"""
ETTh1 (Electricity Transformer Temperature, hourly) forecasting loaders.

Contract (matches benchmarks/bench_etth1.py):
    get_etth1_loaders(batch_size, seq_len, pred_len)
        -> (train_loader, val_loader, test_loader)
    where each batch yields (x, y):
        x: (batch, seq_len, 7)   — a window of all 7 channels
        y: (batch,)              — the target (OT) `pred_len` steps ahead,
                                   single-step when pred_len == 1

ETTh1 has 7 variates; the last column "OT" (oil temperature) is the forecast
target, following the standard ETT benchmark setup (Informer, Autoformer, etc.).

Split: the canonical ETT split is 12 months train / 4 months val / 4 months
test = 12*30*24 / 4*30*24 / 4*30*24 hours. Normalization uses TRAIN-split
statistics only (applied to all splits) to avoid leakage.

The raw CSV (~2.5 MB) is downloaded from the public Informer dataset mirror on
first use and cached under `root`.
"""

import os
from typing import Tuple
from urllib.request import urlretrieve

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


_ETTH1_URL = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
)

# Canonical ETT split boundaries in hours (months * 30 days * 24 hours).
_TRAIN_HOURS = 12 * 30 * 24   # 8640
_VAL_HOURS = 4 * 30 * 24      # 2880
_TEST_HOURS = 4 * 30 * 24     # 2880


def _download_etth1(root: str) -> str:
    """Ensure ETTh1.csv exists under `root`, downloading it if necessary."""
    os.makedirs(root, exist_ok=True)
    csv_path = os.path.join(root, "ETTh1.csv")
    if not os.path.exists(csv_path):
        print(f"Downloading ETTh1.csv to {csv_path} ...")
        urlretrieve(_ETTH1_URL, csv_path)
    return csv_path


def _load_array(root: str) -> np.ndarray:
    """Load the 7 numeric channels of ETTh1 as a (T, 7) float32 array.

    The CSV has a leading 'date' column followed by 7 numeric channels, the
    last of which ('OT') is the target. We drop the date and keep the 7
    channels in their original order.
    """
    csv_path = _download_etth1(root)
    # Parse without pandas to avoid an extra hard dependency: column 0 is the
    # date string, columns 1..7 are the numeric channels.
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
    """Sliding-window dataset over a contiguous slice of the series.

    Each item is (window of seq_len steps across all 7 channels, target OT
    value pred_len steps after the window).
    """

    def __init__(self, series: np.ndarray, seq_len: int, pred_len: int):
        self.series = series
        self.seq_len = seq_len
        self.pred_len = pred_len
        # Last valid start index s.t. window [s, s+seq_len) and target at
        # s+seq_len+pred_len-1 both fit.
        self.n = len(series) - seq_len - pred_len + 1

    def __len__(self) -> int:
        return max(self.n, 0)

    def __getitem__(self, idx: int):
        x = self.series[idx: idx + self.seq_len]                       # (seq_len, 7)
        target_idx = idx + self.seq_len + self.pred_len - 1
        y = self.series[target_idx, -1]                                # OT channel
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


def get_etth1_loaders(
    batch_size: int = 128,
    seq_len: int = 96,
    pred_len: int = 1,
    root: str = "./data",
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train, val, test) DataLoaders for ETTh1 forecasting.

    Args:
        batch_size: Batch size for all three loaders.
        seq_len: Input window length (default 96, the standard short-horizon
            ETT setting).
        pred_len: Forecast horizon. 1 = next-step single-target.
        root: Directory to cache the CSV under.
        num_workers: DataLoader workers (0 is fine; the series is tiny).

    Returns:
        (train_loader, val_loader, test_loader).
    """
    series = _load_array(root)  # (T, 7)

    # Canonical contiguous split.
    train_end = _TRAIN_HOURS
    val_end = _TRAIN_HOURS + _VAL_HOURS
    test_end = _TRAIN_HOURS + _VAL_HOURS + _TEST_HOURS
    test_end = min(test_end, len(series))

    train_raw = series[:train_end]
    # Normalize with TRAIN statistics only (no leakage). Val/test windows are
    # extended backward by seq_len so the first target still has full context.
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
