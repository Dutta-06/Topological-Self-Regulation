"""
California Housing regression loader (sklearn.datasets.fetch_california_housing).

20,640 rows, 8 continuous features (median income, house age, rooms/bedrooms
per household, population, occupancy, lat/long), no categoricals. Target:
median house value in units of $100,000 (range ~0.15-5.00).

Small dataset by design — no preset sweep needed here, single fixed model
size per baseline (see configs/california_housing.yaml).

Returns (x, y) via TabularDataset: x_cat is always an empty (B, 0) tensor.
"""

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import torch

from .common import empty_cat, make_loader, standardize


def get_california_housing_loaders(
    batch_size: int = 256,
    root: str = "./data",
    num_workers: int = 0,
    val_fraction: float = 0.1,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> dict:
    """Return a dict: train/val/test loaders, num_numeric, cat_cardinalities,
    num_classes (None — this is a regression task), y_mean/y_std (for
    unstandardizing predictions back to $100,000 units before scoring)."""
    bunch = fetch_california_housing(data_home=root, download_if_missing=True)
    x, y = bunch.data.astype(np.float32), bunch.target.astype(np.float32)

    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x, y, test_size=test_fraction, random_state=seed
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval, test_size=val_fraction / (1 - test_fraction), random_state=seed
    )

    x_train, x_val, x_test, _, _ = standardize(x_train, x_val, x_test)
    # Standardize the regression target too (training stability); unstandardize
    # predictions before computing metrics — see run_tabular.py.
    y_mean, y_std = float(y_train.mean()), float(y_train.std() + 1e-8)
    y_train_n = (y_train - y_mean) / y_std
    y_val_n = (y_val - y_mean) / y_std
    y_test_n = (y_test - y_mean) / y_std

    def _loader(xn, y, shuffle):
        return make_loader(xn, empty_cat(len(y)), y, torch.float32, batch_size, shuffle, num_workers)

    return {
        "train": _loader(x_train, y_train_n, True),
        "val": _loader(x_val, y_val_n, False),
        "test": _loader(x_test, y_test_n, False),
        "num_numeric": x_train.shape[1],
        "cat_cardinalities": [],
        "num_classes": None,
        "y_mean": y_mean,
        "y_std": y_std,
    }
