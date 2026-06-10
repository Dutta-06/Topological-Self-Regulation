"""
CIFAR-10 / CIFAR-100 data loaders for TSR experiments.

Contract (matches scripts/train.py):
    get_cifar10_loaders(root, batch_size, num_workers, augmentation)
        -> (train_loader, val_loader)
    get_cifar100_loaders(...) -> (train_loader, val_loader)

The "val_loader" is the standard test split used for evaluation — this is the
conventional CIFAR protocol (train on the 50k train split, report on the 10k
test split). Augmentation is applied to train only; eval uses center crop +
normalization so that accuracy numbers are clean and reproducible.
"""

from typing import Tuple

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


# Per-channel mean/std computed over the respective training sets (standard values).
_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2470, 0.2435, 0.2616)
_CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
_CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def _build_transforms(mean, std, augmentation: str):
    """Build (train_transform, eval_transform) for a given augmentation level.

    Args:
        mean, std: Per-channel normalization statistics.
        augmentation: One of "none", "standard", "full". "standard" is
            random-crop + horizontal-flip (the usual CIFAR recipe). "full"
            adds RandAugment. Eval transform never augments.
    """
    normalize = T.Normalize(mean, std)
    eval_tf = T.Compose([T.ToTensor(), normalize])

    if augmentation == "none":
        train_tf = eval_tf
    elif augmentation == "full":
        train_tf = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.RandAugment(),
            T.ToTensor(),
            normalize,
        ])
    else:  # "standard" (default)
        train_tf = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ])
    return train_tf, eval_tf


def _make_loaders(
    dataset_cls,
    mean,
    std,
    root: str,
    batch_size: int,
    num_workers: int,
    augmentation: str,
) -> Tuple[DataLoader, DataLoader]:
    train_tf, eval_tf = _build_transforms(mean, std, augmentation)

    train_set = dataset_cls(root=root, train=True, download=True, transform=train_tf)
    val_set = dataset_cls(root=root, train=False, download=True, transform=eval_tf)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    return train_loader, val_loader


def get_cifar10_loaders(
    root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    augmentation: str = "standard",
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) for CIFAR-10."""
    return _make_loaders(
        torchvision.datasets.CIFAR10, _CIFAR10_MEAN, _CIFAR10_STD,
        root, batch_size, num_workers, augmentation,
    )


def get_cifar100_loaders(
    root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    augmentation: str = "standard",
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) for CIFAR-100."""
    return _make_loaders(
        torchvision.datasets.CIFAR100, _CIFAR100_MEAN, _CIFAR100_STD,
        root, batch_size, num_workers, augmentation,
    )
