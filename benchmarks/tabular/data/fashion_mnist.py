"""
Fashion-MNIST classification loader, treated as a "flattened image" tabular
dataset: each 28x28 grayscale image becomes 784 numeric features (raw pixel
intensities, standardized), 10 classes, no categoricals.

Fashion-MNIST over plain MNIST deliberately: digit MNIST is saturated for any
reasonably-sized model (every baseline here would land at ~97-99%, making the
accuracy column uninformative for comparing architectures) — Fashion-MNIST
was purpose-built by Zalando Research as a harder, non-saturated drop-in
replacement, while remaining exactly as reviewer-familiar. 60,000 train +
10,000 test (official split, preserved).

At 784 features and 60K rows this is genuinely large enough to need real
capacity (like Covertype/Higgs) — the third dataset in this suite that gets
an l/xl preset sweep rather than a single fixed size. One exception: SAINT's
intersample attention flattens all per-sample tokens into one vector before
attending across the batch, so its cost scales with (784 * d_token)^2 — at
this feature count that forces d_token down to single digits to stay
tractable (see configs/fashion_mnist.yaml and the README's fidelity note).
"""

import numpy as np
import torch
from torchvision.datasets import FashionMNIST

from .common import empty_cat, make_loader, standardize


def get_fashion_mnist_loaders(
    batch_size: int = 256,
    root: str = "./data",
    num_workers: int = 0,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> dict:
    train_full = FashionMNIST(root=root, train=True, download=True)
    test = FashionMNIST(root=root, train=False, download=True)

    x_train_full = train_full.data.numpy().reshape(len(train_full), -1).astype(np.float32)
    y_train_full = train_full.targets.numpy().astype(np.int64)
    x_test = test.data.numpy().reshape(len(test), -1).astype(np.float32)
    y_test = test.targets.numpy().astype(np.int64)

    rng = np.random.RandomState(seed)
    n = len(x_train_full)
    perm = rng.permutation(n)
    n_val = int(n * val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    x_train, x_val = x_train_full[train_idx], x_train_full[val_idx]
    y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]

    x_train, x_val, x_test, _, _ = standardize(x_train, x_val, x_test)

    def _loader(xn, y, shuffle):
        return make_loader(xn, empty_cat(len(y)), y, torch.long, batch_size, shuffle, num_workers)

    return {
        "train": _loader(x_train, y_train, True),
        "val": _loader(x_val, y_val, False),
        "test": _loader(x_test, y_test, False),
        "num_numeric": x_train.shape[1],
        "cat_cardinalities": [],
        "num_classes": 10,
        "y_mean": None,
        "y_std": None,
    }
