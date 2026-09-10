"""Tiny-ImageNet-200 fast data loader with vectorized GPU batch transforms.

Tiny-ImageNet-200:
    - 200 classes (ImageNet subset)
    - 500 training images per class (100,000 total)
    - 50 validation images per class (10,000 total)
    - Resolution: 64x64x3 RGB
"""

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.transforms.v2 as v2

_TINY_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_TINY_IMAGENET_STD = (0.229, 0.224, 0.225)
_TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


class FastTensorDataset(Dataset):
    """Zero-overhead dataset backed directly by uint8 (N, 3, 64, 64) tensor."""
    def __init__(self, data_tensor: torch.Tensor, targets: torch.Tensor):
        self.data = data_tensor  # uint8 (N, 3, 64, 64)
        self.targets = targets   # int64 (N,)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class GPUBatchTransform(nn.Module):
    """Vectorized, instant GPU-accelerated batch augmentation module."""
    def __init__(self, is_train: bool = True, augmentation: str = "standard"):
        super().__init__()
        mean = [0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0]
        std = [0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0]
        
        if is_train and augmentation != "none":
            self.tf = nn.Sequential(
                v2.RandomCrop(64, padding=8),
                v2.RandomHorizontalFlip(),
                v2.Normalize(mean=mean, std=std),
            )
        else:
            self.tf = nn.Sequential(
                v2.Normalize(mean=mean, std=std),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, 3, 64, 64) uint8 tensor on CUDA
        return self.tf(x.float())


def _download_and_extract_tiny_imagenet(root: str) -> Path:
    """Download and prepare Tiny-ImageNet-200 in ImageFolder-compatible structure."""
    data_dir = Path(root) / "tiny-imagenet-200"
    zip_path = Path(root) / "tiny-imagenet-200.zip"

    if (data_dir / "train").exists() and (data_dir / "val_formatted").exists():
        return data_dir

    Path(root).mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        if not zip_path.exists():
            print(f"Downloading Tiny-ImageNet-200 from {_TINY_IMAGENET_URL}...")
            urllib.request.urlretrieve(_TINY_IMAGENET_URL, zip_path)
            print("Download complete.")

        print("Extracting Tiny-ImageNet-200 archive...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(root)
        print("Extraction complete.")

    val_formatted = data_dir / "val_formatted"
    if not val_formatted.exists():
        print("Formatting Tiny-ImageNet validation directory for ImageFolder...")
        val_img_dir = data_dir / "val" / "images"
        val_annot_path = data_dir / "val" / "val_annotations.txt"

        val_formatted.mkdir(parents=True, exist_ok=True)
        with open(val_annot_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    img_name, class_id = parts[0], parts[1]
                    class_dir = val_formatted / class_id
                    class_dir.mkdir(parents=True, exist_ok=True)
                    src = val_img_dir / img_name
                    dst = class_dir / img_name
                    if src.exists() and not dst.exists():
                        shutil.copy(src, dst)
        print("Validation formatting complete.")

    return data_dir


def _compile_tensor_cache(data_dir: Path) -> Tuple[Path, Path]:
    """Compile dataset into binary .pt files for instant loading without disk I/O lag."""
    train_cache = data_dir / "train_cache.pt"
    val_cache = data_dir / "val_cache.pt"

    if train_cache.exists() and val_cache.exists():
        return train_cache, val_cache

    print("Compiling Tiny-ImageNet tensor cache for fast training (one-time setup)...")
    raw_train = datasets.ImageFolder(str(data_dir / "train"))
    raw_val = datasets.ImageFolder(str(data_dir / "val_formatted"))

    def to_tensors(ds):
        n = len(ds)
        data = torch.empty(n, 3, 64, 64, dtype=torch.uint8)
        targets = torch.empty(n, dtype=torch.long)
        for i, (img_pil, tgt) in enumerate(ds):
            arr = np.array(img_pil)
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            data[i] = torch.from_numpy(arr).permute(2, 0, 1)
            targets[i] = tgt
        return data, targets

    import numpy as np
    train_x, train_y = to_tensors(raw_train)
    torch.save({"data": train_x, "targets": train_y}, train_cache)
    del train_x, train_y

    val_x, val_y = to_tensors(raw_val)
    torch.save({"data": val_x, "targets": val_y}, val_cache)
    del val_x, val_y
    print("Tensor cache compiled successfully.")
    return train_cache, val_cache


def get_tiny_imagenet_loaders(
    root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 0,
    augmentation: str = "standard",
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) for Tiny-ImageNet-200."""
    data_dir = _download_and_extract_tiny_imagenet(root)
    train_cache, val_cache = _compile_tensor_cache(data_dir)

    train_dict = torch.load(train_cache, weights_only=False)
    val_dict = torch.load(val_cache, weights_only=False)

    train_dataset = FastTensorDataset(train_dict["data"], train_dict["targets"])
    val_dataset = FastTensorDataset(val_dict["data"], val_dict["targets"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader
