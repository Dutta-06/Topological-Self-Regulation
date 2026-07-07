"""
HIGGS classification loader (UCI, Baldi/Sadowski/Whiteson 2014).

Full dataset: 11,000,000 rows, 28 features (21 low-level kinematic + 7
high-level derived), binary label (signal vs. background). No categoricals.

Protocol (matches the UCI page's stated convention, the one every paper
using this dataset follows): the **last 500,000 rows are the test set**.
We read them via `skiprows` rather than loading all 11M rows, since the
exact row count is fixed and documented.

Subsampling: the full training pool (10.5M rows) is far more than any of
these baselines need to reach their param-vs-accuracy operating point on a
single GPU, and reading all of it is a multi-minute, multi-GB pandas parse
for no benefit. `num_train_rows` (config: `data.num_train_rows`, default
1,000,000) takes a **prefix** of the training file rather than a random
sample across the full 10.5M — cheap (avoids parsing rows we'd discard) and
valid because HIGGS ships as pre-shuffled simulated collision events with no
ordering key (no sort-by-label or chronological structure), which is why
downstream scaling-law papers routinely use row-count prefixes of this file
interchangeably with random subsamples. This is what makes Higgs the
"bulkier model" dataset in this suite: even at a 1M-row subsample, 28
continuous features and a famously non-linear decision boundary support a
genuine multi-million-param Pareto sweep (l/xl presets) without padding.

A validation split is carved from the tail of that training prefix.
"""

import gzip
import os
import shutil
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch

from .common import empty_cat, make_loader, standardize

_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
_TOTAL_ROWS = 11_000_000
_TEST_ROWS = 500_000
_NUM_FEATURES = 28


def _download(root: str) -> str:
    os.makedirs(root, exist_ok=True)
    gz_path = os.path.join(root, "HIGGS.csv.gz")
    if not os.path.exists(gz_path):
        print(f"Downloading HIGGS dataset (~2.6GB) to {gz_path} ... this may take a while.")
        urlretrieve(_URL, gz_path)
    return gz_path


def get_higgs_loaders(
    batch_size: int = 512,
    root: str = "./data",
    num_workers: int = 0,
    num_train_rows: int = 1_000_000,
    val_fraction: float = 0.05,
    seed: int = 42,
) -> dict:
    gz_path = _download(root)
    dtype = {i: np.float32 for i in range(_NUM_FEATURES + 1)}

    train_pool = pd.read_csv(gz_path, header=None, nrows=num_train_rows, dtype=dtype).values
    test_arr = pd.read_csv(
        gz_path, header=None, skiprows=_TOTAL_ROWS - _TEST_ROWS, nrows=_TEST_ROWS, dtype=dtype
    ).values

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(train_pool))
    n_val = int(len(train_pool) * val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    y_train, y_val = train_pool[train_idx, 0].astype(np.int64), train_pool[val_idx, 0].astype(np.int64)
    y_test = test_arr[:, 0].astype(np.int64)
    x_train, x_val, x_test = train_pool[train_idx, 1:], train_pool[val_idx, 1:], test_arr[:, 1:]

    x_train, x_val, x_test, _, _ = standardize(x_train, x_val, x_test)

    def _loader(xn, y, shuffle):
        return make_loader(xn, empty_cat(len(y)), y, torch.long, batch_size, shuffle, num_workers)

    return {
        "train": _loader(x_train, y_train, True),
        "val": _loader(x_val, y_val, False),
        "test": _loader(x_test, y_test, False),
        "num_numeric": x_train.shape[1],
        "cat_cardinalities": [],
        "num_classes": 2,
        "y_mean": None,
        "y_std": None,
    }
