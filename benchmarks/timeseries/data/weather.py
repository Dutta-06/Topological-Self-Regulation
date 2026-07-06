"""
Weather forecasting loader — MPI Jena weather station, 2020, 21 variates.

Source: the Max Planck Institute for Biogeochemistry's roof-station weather
data (https://www.bgc-jena.mpg.de/wetter/), 10-minute resolution. The
standard "Weather" long-horizon forecasting benchmark (Informer/Autoformer/
PatchTST) uses the full 2020 year of this station's data, 21 numeric
channels — this loader reproduces that exactly (same station, same year,
same channel count), concatenating the institute's own half-year archives.

Split: 70% train / 10% val / 20% test by row count — the generic ratio-based
split Informer/Autoformer use for every non-ETT dataset (ETT uses fixed
calendar months instead; that's a dataset-specific convention, not a bug).

Same batch contract as get_ett_loaders: (x, y) with
    x: (batch, seq_len, 21)
    y: (batch, pred_len, 21)  — next pred_len steps, all channels.
"""

import os
from typing import Tuple
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

_URLS = [
    "https://www.bgc-jena.mpg.de/wetter/mpi_roof_2020a.zip",
    "https://www.bgc-jena.mpg.de/wetter/mpi_roof_2020b.zip",
]


def _download_and_load(root: str) -> pd.DataFrame:
    os.makedirs(root, exist_ok=True)
    csv_path = os.path.join(root, "weather.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, index_col=0, parse_dates=True)

    halves = []
    for url in _URLS:
        fname = url.rsplit("/", 1)[-1]
        zip_path = os.path.join(root, fname)
        if not os.path.exists(zip_path):
            print(f"Downloading {fname} to {zip_path} ...")
            urlretrieve(url, zip_path)
        with ZipFile(zip_path) as zf:
            inner_name = zf.namelist()[0]
            with zf.open(inner_name) as f:
                half = pd.read_csv(f, index_col=0, parse_dates=True, encoding="latin1")
        halves.append(half)

    df = pd.concat(halves, axis=0)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df.to_csv(csv_path)
    return df


def _load_array(root: str) -> np.ndarray:
    df = _download_and_load(root)
    return df.to_numpy(dtype=np.float32)  # (T, 21)


class _WindowDataset(Dataset):
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


def get_weather_loaders(
    batch_size: int = 128,
    seq_len: int = 96,
    pred_len: int = 1,
    root: str = "./data",
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train, val, test) DataLoaders for the Weather forecasting benchmark.

    Args:
        batch_size: Batch size for all three loaders.
        seq_len: Input window length.
        pred_len: Forecast horizon (steps ahead of the window's end, in 10-min steps).
        root: Directory to cache the concatenated CSV under.
        num_workers: DataLoader workers.
    """
    series = _load_array(root)  # (T, 21)
    n = len(series)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)

    train_raw = series[:train_end]
    mean = train_raw.mean(axis=0, keepdims=True)
    std = train_raw.std(axis=0, keepdims=True) + 1e-8
    norm = (series - mean) / std

    train_series = norm[:train_end]
    val_series = norm[train_end - seq_len: val_end]
    test_series = norm[val_end - seq_len:]

    pin = torch.cuda.is_available()

    def _loader(arr, shuffle):
        ds = _WindowDataset(arr, seq_len, pred_len)
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin, drop_last=False,
        )

    return _loader(train_series, True), _loader(val_series, False), _loader(test_series, False)
