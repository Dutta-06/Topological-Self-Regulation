"""
UCR/UEA time series classification archive loader, via aeon.

The archive spans 100+ datasets in inconsistent per-dataset raw formats
(.ts/.arff, variable length, missing values); aeon's load_classification
handles the download, caching, and parsing correctly, so this module is a
thin adapter rather than a hand-rolled parser.

Returns (x, y):
    x: (batch, seq_len, n_channels)
    y: (batch,)  — integer class
"""

from typing import List, Tuple

import numpy as np
import torch
from aeon.datasets import load_classification
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

# A small, representative subset spanning univariate/multivariate and dataset
# size — not the full archive. Extend via the `datasets` config list.
DEFAULT_SUBSET = ["ECG200", "FordA", "BasicMotions"]


class _UCRDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.x[idx]), torch.tensor(self.y[idx], dtype=torch.long)


def _to_batch_first(x: np.ndarray) -> np.ndarray:
    """aeon returns (n_instances, n_channels, series_length); models here expect
    (n_instances, series_length, n_channels)."""
    return np.ascontiguousarray(x.transpose(0, 2, 1).astype(np.float32))


def get_ucr_uea_loaders(
    name: str,
    batch_size: int = 32,
    root: str = "./data/ucr_uea",
    num_workers: int = 0,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, int, int]:
    """Return (train, val, test, seq_len, num_channels, num_classes) for one
    UCR/UEA dataset.

    A held-out val split is carved from the archive's train split (the
    archive ships only train/test); test is left untouched for final report.
    """
    x_train_full, y_train_full = load_classification(name, split="train", extract_path=root)
    x_test, y_test = load_classification(name, split="test", extract_path=root)

    x_train_full = _to_batch_first(x_train_full)
    x_test = _to_batch_first(x_test)

    encoder = LabelEncoder()
    y_train_full = encoder.fit_transform(y_train_full)
    y_test = encoder.transform(y_test)

    mean = x_train_full.mean(axis=(0, 1), keepdims=True)
    std = x_train_full.std(axis=(0, 1), keepdims=True) + 1e-8
    x_train_full = (x_train_full - mean) / std
    x_test = (x_test - mean) / std

    rng = np.random.RandomState(seed)
    n = len(x_train_full)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    x_train, y_train = x_train_full[train_idx], y_train_full[train_idx]
    x_val, y_val = x_train_full[val_idx], y_train_full[val_idx]

    pin = torch.cuda.is_available()

    def _loader(x, y, shuffle):
        ds = _UCRDataset(x, y)
        return DataLoader(
            ds, batch_size=min(batch_size, len(ds)), shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin, drop_last=False,
        )

    seq_len = x_train_full.shape[1]
    num_channels = x_train_full.shape[2]
    num_classes = int(len(encoder.classes_))

    return (
        _loader(x_train, y_train, True),
        _loader(x_val, y_val, False),
        _loader(x_test, y_test, False),
        seq_len,
        num_channels,
        num_classes,
    )


def list_default_subset() -> List[str]:
    return list(DEFAULT_SUBSET)
