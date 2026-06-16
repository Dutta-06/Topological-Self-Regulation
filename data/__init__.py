"""Data loading for TSR experiments (CIFAR-10/100, ETTh1 forecasting)."""

from data.cifar import get_cifar10_loaders, get_cifar100_loaders
from data.etth1 import get_etth1_loaders

__all__ = [
    "get_cifar10_loaders",
    "get_cifar100_loaders",
    "get_etth1_loaders",
]
