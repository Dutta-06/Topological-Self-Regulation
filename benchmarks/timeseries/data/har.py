"""
UCI HAR (Human Activity Recognition Using Smartphones) classification loader.

Pre-windowed inertial signals: 9 channels (body_acc x/y/z, body_gyro x/y/z,
total_acc x/y/z) at 50Hz, 128-step windows (2.56s, 50% overlap), 6 activity
classes. One of the most standard small-scale multivariate TS classification
benchmarks.

Returns (x, y):
    x: (batch, 128, 9)
    y: (batch,)  — integer class in [0, 5]
"""

import os
import zipfile
from typing import Tuple
from urllib.request import urlretrieve

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"

_SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]


def _download_and_extract(root: str) -> str:
    """Ensure 'UCI HAR Dataset/' exists under root; download+extract if not."""
    dataset_dir = os.path.join(root, "UCI HAR Dataset")
    if os.path.isdir(dataset_dir):
        return dataset_dir
    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, "uci_har.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading UCI HAR dataset to {zip_path} ...")
        urlretrieve(_URL, zip_path)
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    return dataset_dir


def _load_split(dataset_dir: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    signal_dir = os.path.join(dataset_dir, split, "Inertial Signals")
    channels = []
    for sig in _SIGNALS:
        path = os.path.join(signal_dir, f"{sig}_{split}.txt")
        channels.append(np.loadtxt(path, dtype=np.float32))  # (n_samples, 128)
    x = np.stack(channels, axis=-1)  # (n_samples, 128, 9)

    y_path = os.path.join(dataset_dir, split, f"y_{split}.txt")
    y = np.loadtxt(y_path, dtype=np.int64) - 1  # labels are 1..6 -> 0..5
    return x, y


class _HARDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.x[idx]), torch.tensor(self.y[idx], dtype=torch.long)


def get_har_loaders(
    batch_size: int = 64,
    root: str = "./data",
    num_workers: int = 0,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, int]:
    """Return (train, val, test, num_channels, num_classes) for UCI HAR.

    The archive ships only train/test; a held-out val split is carved from
    train (stratified is unnecessary here — a random split is standard
    practice for this benchmark since classes are fairly balanced).
    """
    dataset_dir = _download_and_extract(root)
    x_train_full, y_train_full = _load_split(dataset_dir, "train")
    x_test, y_test = _load_split(dataset_dir, "test")

    # Per-channel normalization using train statistics only.
    mean = x_train_full.mean(axis=(0, 1), keepdims=True)
    std = x_train_full.std(axis=(0, 1), keepdims=True) + 1e-8
    x_train_full = (x_train_full - mean) / std
    x_test = (x_test - mean) / std

    rng = np.random.RandomState(seed)
    n = len(x_train_full)
    perm = rng.permutation(n)
    n_val = int(n * val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    x_train, y_train = x_train_full[train_idx], y_train_full[train_idx]
    x_val, y_val = x_train_full[val_idx], y_train_full[val_idx]

    pin = torch.cuda.is_available()

    def _loader(x, y, shuffle):
        ds = _HARDataset(x, y)
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin, drop_last=False,
        )

    num_channels = x_train.shape[-1]
    num_classes = int(max(y_train_full.max(), y_test.max())) + 1

    return (
        _loader(x_train, y_train, True),
        _loader(x_val, y_val, False),
        _loader(x_test, y_test, False),
        num_channels,
        num_classes,
    )
