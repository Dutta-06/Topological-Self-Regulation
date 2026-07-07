"""
UCI Adult (Census Income) classification loader.

32,561 train + 16,281 test rows (official UCI split, preserved rather than
re-shuffled — this is the one dataset in the tabular suite with genuine
categorical columns, so it's the code path that exercises embedding-based
categorical handling in every model).

6 numeric features: age, fnlwgt, education-num, capital-gain, capital-loss,
hours-per-week.
8 categorical features: workclass, education, marital-status, occupation,
relationship, race, sex, native-country. "?" (UCI's missing-value token) is
kept as its own category rather than dropping rows — Adult is small enough
that discarding ~7% of rows with missing values would be wasteful, and this
matches standard practice in the tabular-DL literature.

Target: income >50K (1) vs <=50K (0).

Small/simple dataset by design — no preset sweep needed (see
configs/adult.yaml), single fixed model size per baseline.
"""

import os
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch

from .common import CategoryEncoder, make_loader, standardize

_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult"

_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
]
_NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
_CATEGORICAL = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]


def _download(root: str) -> tuple:
    os.makedirs(root, exist_ok=True)
    train_path = os.path.join(root, "adult.data")
    test_path = os.path.join(root, "adult.test")
    if not os.path.exists(train_path):
        print(f"Downloading Adult train split to {train_path} ...")
        urlretrieve(f"{_BASE_URL}/adult.data", train_path)
    if not os.path.exists(test_path):
        print(f"Downloading Adult test split to {test_path} ...")
        urlretrieve(f"{_BASE_URL}/adult.test", test_path)
    return train_path, test_path


def _load_df(path: str, skiprows: int = 0) -> pd.DataFrame:
    df = pd.read_csv(
        path, header=None, names=_COLUMNS, skiprows=skiprows,
        sep=r",\s*", engine="python", na_values="?",
    )
    df = df.dropna(subset=["income"])  # trailing blank lines in adult.test
    # adult.test labels have a trailing period ("<=50K.") that adult.data lacks.
    df["income"] = df["income"].str.rstrip(".")
    for col in _CATEGORICAL:
        df[col] = df[col].fillna("?")
    return df


def get_adult_loaders(
    batch_size: int = 256,
    root: str = "./data",
    num_workers: int = 0,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> dict:
    train_path, test_path = _download(root)
    df_train_full = _load_df(train_path)
    df_test = _load_df(test_path, skiprows=1)  # adult.test has a junk header line

    y_train_full = (df_train_full["income"] == ">50K").astype(np.int64).values
    y_test = (df_test["income"] == ">50K").astype(np.int64).values

    x_num_train_full = df_train_full[_NUMERIC].astype(np.float32).values
    x_num_test = df_test[_NUMERIC].astype(np.float32).values

    encoders = [CategoryEncoder().fit(df_train_full[col].values) for col in _CATEGORICAL]
    cat_cardinalities = [enc.cardinality for enc in encoders]
    x_cat_train_full = np.stack(
        [enc.transform(df_train_full[col].values) for enc, col in zip(encoders, _CATEGORICAL)], axis=1
    )
    x_cat_test = np.stack(
        [enc.transform(df_test[col].values) for enc, col in zip(encoders, _CATEGORICAL)], axis=1
    )

    rng = np.random.RandomState(seed)
    n = len(y_train_full)
    perm = rng.permutation(n)
    n_val = int(n * val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    x_num_train, x_num_val = x_num_train_full[train_idx], x_num_train_full[val_idx]
    x_cat_train, x_cat_val = x_cat_train_full[train_idx], x_cat_train_full[val_idx]
    y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]

    x_num_train, x_num_val, x_num_test, _, _ = standardize(x_num_train, x_num_val, x_num_test)

    def _loader(xn, xc, y, shuffle):
        return make_loader(xn, xc, y, torch.long, batch_size, shuffle, num_workers)

    return {
        "train": _loader(x_num_train, x_cat_train, y_train, True),
        "val": _loader(x_num_val, x_cat_val, y_val, False),
        "test": _loader(x_num_test, x_cat_test, y_test, False),
        "num_numeric": x_num_train.shape[1],
        "cat_cardinalities": cat_cardinalities,
        "num_classes": 2,
        "y_mean": None,
        "y_std": None,
    }
