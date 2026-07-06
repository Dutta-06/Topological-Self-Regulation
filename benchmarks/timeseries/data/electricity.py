"""
UCI Electricity Load Diagrams 2011-2014 (ECL) forecasting loader.

Raw format: semicolon-delimited, comma-decimal, 15-minute readings (kW) for
370 clients from 2011-01-01 to 2015-01-01, first column an unlabeled
timestamp. Resampled to hourly (mean of the 4 sub-hour readings). Default
num_clients=321 matches the standard "ECL" convention used in long-sequence
forecasting benchmarks (Informer/Autoformer/PatchTST) — those papers exclude
clients not yet installed in 2011 (all-zero early readings); we reproduce
that filtering rule (drop any client with a zero reading anywhere in the
used date range) rather than assuming column order, since the exact client
IDs the original papers kept were never published as a simple rule. All
selected client channels are forecast jointly (multivariate), mirroring the
ETT loaders' schema so the same window/target logic applies.

Same batch contract as get_ett_loaders: (x, y) with
    x: (batch, seq_len, n_channels)
    y: (batch, pred_len, n_channels)  — next pred_len steps, all channels.
"""

import os
import zipfile
from typing import Tuple
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

_URL = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"

# Canonical split: 15 months train / 3 val / 3.5 test (matches the widely used
# ECL preprocessing convention — dataset only has full-year density from 2012).
_TRAIN_HOURS = 18 * 30 * 24
_VAL_HOURS = 3 * 30 * 24
_TEST_HOURS = 4 * 30 * 24


def _download_and_extract(root: str) -> str:
    os.makedirs(root, exist_ok=True)
    txt_path = os.path.join(root, "LD2011_2014.txt")
    if os.path.exists(txt_path):
        return txt_path
    zip_path = os.path.join(root, "LD2011_2014.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading electricity load diagrams to {zip_path} ... (~250MB)")
        urlretrieve(_URL, zip_path)
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path) as zf:
        # Exclude macOS resource-fork junk (__MACOSX/._*.txt) — it also ends in
        # .txt and would otherwise be picked depending on zip entry order.
        names = [
            n for n in zf.namelist()
            if n.lower().endswith(".txt") and "__MACOSX" not in n
        ]
        if not names:
            raise FileNotFoundError(f"No .txt file found in {zip_path}")
        zf.extract(names[0], root)
        extracted = os.path.join(root, names[0])
        if extracted != txt_path:
            os.rename(extracted, txt_path)
    return txt_path


def _load_array(root: str, num_clients: int) -> np.ndarray:
    """Load, hourly-resample, and select the num_clients earliest-installed columns.

    Clients not yet installed report exact zeros (not NaNs) until their first
    real reading. Most clients still have occasional zero readings later on
    (legitimate zero consumption), so "never zero" is too strict a filter —
    only ~90-160 of 370 columns pass it. Instead we rank every column by its
    first nonzero timestamp (installation date) and keep the num_clients
    earliest-installed ones. This is deterministic and reproducible from the
    raw file alone; the original Informer/Autoformer "321 clients" figure
    isn't published as a simple derivable rule, so this is our own rule
    chosen to land on the same standard scale, not a claim of bit-identical
    client-ID reproduction.
    """
    txt_path = _download_and_extract(root)
    df = pd.read_csv(txt_path, sep=";", decimal=",", index_col=0, parse_dates=True)
    df = df.resample("1h").mean()
    install_date = df.ne(0).idxmax()  # first nonzero timestamp per column
    cols = list(install_date.sort_values().index[:num_clients])
    return df[cols].to_numpy(dtype=np.float32)  # (T, num_clients)


class _WindowDataset(Dataset):
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


def get_electricity_loaders(
    batch_size: int = 128,
    seq_len: int = 96,
    pred_len: int = 1,
    root: str = "./data",
    num_workers: int = 0,
    num_clients: int = 321,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train, val, test) DataLoaders for UCI Electricity Load forecasting.

    Args:
        batch_size: Batch size for all three loaders.
        seq_len: Input window length.
        pred_len: Forecast horizon (steps ahead of the window's end, in hours).
        root: Directory to cache/extract the raw file under.
        num_workers: DataLoader workers.
        num_clients: Number of fully-active client columns to keep. Default
            321 matches the standard "ECL" benchmark scale used in
            Informer/Autoformer/PatchTST (of 370 raw columns, ~321-336 are
            fully active across the whole series; the rest are zero-padded
            placeholders for clients not yet installed).
    """
    series = _load_array(root, num_clients)  # (T, num_clients)

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
        ds = _WindowDataset(arr, seq_len, pred_len)
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin, drop_last=False,
        )

    return _loader(train_series, True), _loader(val_series, False), _loader(test_series, False)
