"""Shared Dataset/encoding utilities for the tabular loaders."""

from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TabularDataset(Dataset):
    """x_num: (N, F_num) float32, x_cat: (N, F_cat) int64, y: (N,) float32/int64."""

    def __init__(self, x_num: np.ndarray, x_cat: np.ndarray, y: np.ndarray, y_dtype: torch.dtype):
        self.x_num = x_num.astype(np.float32)
        self.x_cat = x_cat.astype(np.int64)
        self.y = y
        self.y_dtype = y_dtype

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.x_num[idx]),
            torch.from_numpy(self.x_cat[idx]),
            torch.tensor(self.y[idx], dtype=self.y_dtype),
        )


def make_loader(x_num, x_cat, y, y_dtype, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    ds = TabularDataset(x_num, x_cat, y, y_dtype)
    pin = torch.cuda.is_available()
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin, drop_last=False,
    )


class CategoryEncoder:
    """Label-encodes a categorical column using categories seen at fit time.

    Any category seen only at transform time (not in the fitted vocabulary)
    maps to a single extra "unknown" index — the standard way to handle
    train/test category mismatches without leaking test-time information
    into the vocabulary.
    """

    def __init__(self):
        self.mapping = {}
        self.cardinality = 0

    def fit(self, values: np.ndarray) -> "CategoryEncoder":
        uniques = sorted(set(values.tolist()))
        self.mapping = {v: i for i, v in enumerate(uniques)}
        self.cardinality = len(uniques) + 1  # +1 for unknown
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        unk = self.cardinality - 1
        return np.array([self.mapping.get(v, unk) for v in values], dtype=np.int64)


def standardize(train: np.ndarray, *others: np.ndarray):
    """Fit mean/std on `train`, apply to train + others. Returns (train, *others, mean, std)."""
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-8
    out = [(train - mean) / std]
    for arr in others:
        out.append((arr - mean) / std)
    return (*out, mean, std)


def empty_cat(n: int) -> np.ndarray:
    return np.zeros((n, 0), dtype=np.int64)
