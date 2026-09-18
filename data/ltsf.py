"""Long-term timeseries forecasting (LTSF) loaders: ETTh1, ETTh2, Weather,
Electricity (ECL), Traffic.

Multivariate protocol matching the Informer/Autoformer/PatchTST line of
work — NOT the single-target scalar setup in `data/etth1.py` (that loader
predicts one channel from a window; this one predicts every channel
`pred_len` steps ahead, which is the standard benchmark task and the only
way results are comparable to published numbers):

    get_ltsf_loaders(name, seq_len, pred_len, batch_size)
        -> (train_loader, val_loader, test_loader)
    each batch yields (x, y):
        x: (batch, seq_len, C)
        y: (batch, pred_len, C)

Split convention (matches Informer's `Dataset_Custom`/`Dataset_ETT_hour`,
required for the numbers to mean anything against the literature):
  - ETTh1 / ETTh2: canonical 12/4/4 months (8640 / 2880 / 2880 hours).
  - weather / electricity / traffic: 70% / (T - 70% - 20%) / 20% of the
    series length.
  - In both cases val/test windows are extended `seq_len` steps into the
    PRECEDING split so the first window has real context instead of zeros
    (`border1s`/`border2s` below) — this is a deliberate, small train/val
    leak of input-only context that every paper in this line reports
    under, not a bug.
  - Normalization (z-score) is fit on the TRAIN split only, then applied
    to the whole series before windowing.
"""

import gzip
import os
from typing import Tuple
from urllib.request import urlretrieve

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

_ETT_URLS = {
    "ETTh1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "ETTh2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
}
_ELECTRICITY_URL = "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/electricity/electricity.txt.gz"
_TRAFFIC_URL = "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/traffic/traffic.txt.gz"

_ETT_HOUR_BORDERS = (12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24)  # train, val, test (hours)


def _ensure_file(root: str, filename: str, url: str) -> str:
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, filename)
    if not os.path.exists(path):
        print(f"Downloading {filename} to {path} ...")
        urlretrieve(url, path)
    return path


def _load_ett(root: str, name: str) -> np.ndarray:
    csv_path = _ensure_file(root, f"{name}.csv", _ETT_URLS[name])
    rows = []
    with open(csv_path, "r") as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            rows.append([float(v) for v in parts[1:8]])
    return np.asarray(rows, dtype=np.float32)  # (T, 7)


def _load_weather(root: str) -> np.ndarray:
    csv_path = os.path.join(root, "weather.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. weather.csv (Max Planck Institute, 21 channels) "
            "has no small canonical raw mirror — place it under data/ manually."
        )
    rows = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        header = next(f).strip().split(",")
        n_cols = len(header) - 1  # drop "Date Time"
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < n_cols + 1:
                continue
            try:
                rows.append([float(v) for v in parts[1:1 + n_cols]])
            except ValueError:
                continue  # occasional malformed row in the raw MPI export
    return np.asarray(rows, dtype=np.float32)  # (T, 21)


def _load_gzipped_matrix(root: str, filename: str, url: str) -> np.ndarray:
    """electricity.txt / traffic.txt: headerless comma-separated numeric
    matrix, no date column (LSTNet's raw preprocessed hourly aggregates —
    the same files the Informer/Autoformer/PatchTST papers derive their
    'electricity'/'traffic' benchmark CSVs from)."""
    gz_path = _ensure_file(root, filename + ".gz", url)
    txt_path = os.path.join(root, filename)
    if not os.path.exists(txt_path):
        with gzip.open(gz_path, "rt") as fin, open(txt_path, "w") as fout:
            fout.write(fin.read())
    rows = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([float(v) for v in line.split(",")])
    return np.asarray(rows, dtype=np.float32)


def _load_raw(name: str, root: str) -> np.ndarray:
    if name in _ETT_URLS:
        return _load_ett(root, name)
    if name == "weather":
        return _load_weather(root)
    if name == "electricity":
        return _load_gzipped_matrix(root, "electricity.txt", _ELECTRICITY_URL)
    if name == "traffic":
        return _load_gzipped_matrix(root, "traffic.txt", _TRAFFIC_URL)
    raise ValueError(f"unknown LTSF dataset {name!r}")


def _borders(name: str, T: int, seq_len: int) -> Tuple[list, list]:
    if name in _ETT_URLS:
        n_train, n_val, n_test = _ETT_HOUR_BORDERS
        border1s = [0, n_train - seq_len, n_train + n_val - seq_len]
        border2s = [n_train, n_train + n_val, n_train + n_val + n_test]
    else:
        n_train = int(T * 0.7)
        n_test = int(T * 0.2)
        n_val = T - n_train - n_test
        border1s = [0, n_train - seq_len, T - n_test - seq_len]
        border2s = [n_train, n_train + n_val, T]
    return border1s, border2s


class _LTSFWindowDataset(Dataset):
    """Sliding-window dataset: x = window of seq_len steps (all channels),
    y = the next pred_len steps (all channels)."""

    def __init__(self, series: np.ndarray, seq_len: int, pred_len: int):
        self.series = series
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n = len(series) - seq_len - pred_len + 1

    def __len__(self) -> int:
        return max(self.n, 0)

    def __getitem__(self, idx: int):
        x = self.series[idx: idx + self.seq_len]
        y = self.series[idx + self.seq_len: idx + self.seq_len + self.pred_len]
        return torch.from_numpy(x), torch.from_numpy(y)


def get_ltsf_loaders(
    name: str,
    seq_len: int = 96,
    pred_len: int = 96,
    batch_size: int = 32,
    root: str = "./data",
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train, val, test) DataLoaders for one of
    {"ETTh1", "ETTh2", "weather", "electricity", "traffic"}.

    Every batch is (x, y) with x: (B, seq_len, C), y: (B, pred_len, C).
    """
    series = _load_raw(name, root)  # (T, C)
    T = len(series)
    border1s, border2s = _borders(name, T, seq_len)

    train_raw = series[border1s[0]: border2s[0]]
    mean = train_raw.mean(axis=0, keepdims=True)
    std = train_raw.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    normed = (series - mean) / std

    loaders = []
    for i, shuffle in enumerate((True, False, False)):
        split = normed[border1s[i]: border2s[i]]
        ds = _LTSFWindowDataset(split, seq_len, pred_len)
        loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                   num_workers=num_workers, drop_last=False))
    return tuple(loaders)


def n_channels(name: str) -> int:
    return {"ETTh1": 7, "ETTh2": 7, "weather": 21, "electricity": 321, "traffic": 862}[name]
