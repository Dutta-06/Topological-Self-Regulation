"""
Forest Cover Type classification loader (sklearn.datasets.fetch_covtype).

581,012 rows, 54 features, 7 classes. All 54 columns are already numeric —
10 continuous (elevation, aspect, slope, hillshade indices, distances) plus
44 binary indicator columns (4 wilderness-area + 40 soil-type one-hot flags
shipped pre-encoded by UCI) — so there are no categorical columns to embed
here; the full 54-dim vector is treated as `x_num`.

No official train/test split is shipped, so we use a fixed-seed 80/10/10
split (documented convention, not a claim of matching any specific paper's
exact split) — large enough that test-set variance from the split choice is
negligible.

581K rows / 54 features is genuinely large enough to need real capacity
(matches the "Revisiting Tabular DL" paper's ResNet/FT-Transformer configs
landing in the 1-2M param range for this dataset) — this is the tabular
suite's multi-million-param Pareto-sweep dataset alongside Higgs.
"""

import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
import torch

from .common import empty_cat, make_loader, standardize


def get_covertype_loaders(
    batch_size: int = 512,
    root: str = "./data",
    num_workers: int = 0,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> dict:
    bunch = fetch_covtype(data_home=root, download_if_missing=True)
    x = bunch.data.astype(np.float32)
    y = (bunch.target - 1).astype(np.int64)  # labels are 1..7 -> 0..6

    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x, y, test_size=test_fraction, random_state=seed, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval, test_size=val_fraction / (1 - test_fraction),
        random_state=seed, stratify=y_trainval,
    )

    x_train, x_val, x_test, _, _ = standardize(x_train, x_val, x_test)

    def _loader(xn, y, shuffle):
        return make_loader(xn, empty_cat(len(y)), y, torch.long, batch_size, shuffle, num_workers)

    return {
        "train": _loader(x_train, y_train, True),
        "val": _loader(x_val, y_val, False),
        "test": _loader(x_test, y_test, False),
        "num_numeric": x_train.shape[1],
        "cat_cardinalities": [],
        "num_classes": int(y.max()) + 1,
        "y_mean": None,
        "y_std": None,
    }
