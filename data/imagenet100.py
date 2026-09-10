"""ImageNet-100 data loader and dataset preparation module.

ImageNet-100:
    - 100 classes from ILSVRC2012 ImageNet
    - ~126,689 training images (~1,260 per class)
    - 5,000 validation images (50 per class)
    - Full photographic resolution (standard 224x224 RGB crops)
    - Uses the official ImageNet stem (7x7 stride-2 conv + 3x3 maxpool)
"""

import io
import os
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_HF_REPO_ID = "clane9/imagenet-100"
_NUM_TRAIN_SHARDS = 17
_NUM_VAL_SHARDS = 1


def _extract_parquet_shard(parquet_path: Path, target_dir: Path, split_name: str, shard_idx: int) -> int:
    """Extract PNG/JPEG bytes from one parquet file into ImageFolder structure."""
    import pyarrow.parquet as pq

    # Pre-create all 100 class directories to avoid 100k filesystem stat calls
    for c in range(100):
        (target_dir / f"class_{c:03d}").mkdir(parents=True, exist_ok=True)

    table = pq.read_table(str(parquet_path))
    labels = table["label"].to_pylist()
    img_list = table["image"].to_pylist()

    count = 0
    for i, (label, img_struct) in enumerate(zip(labels, img_list)):
        img_path = target_dir / f"class_{label:03d}" / f"{split_name}_{shard_idx:02d}_{i:06d}.png"
        if not img_path.exists():
            with open(img_path, "wb") as f:
                f.write(img_struct["bytes"])
        count += 1

    return count


def _download_with_retry(repo_id: str, filename: str, max_retries: int = 8, initial_wait: float = 15.0) -> str:
    """Download a file from Hugging Face with exponential backoff on HTTP 429 / network errors."""
    import time
    from huggingface_hub import hf_hub_download

    for attempt in range(max_retries):
        try:
            return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = initial_wait * (1.5 ** attempt)
            print(f"    [WARN] Download hit error: {e}. Backing off for {wait_time:.1f}s (retry {attempt+1}/{max_retries})...")
            time.sleep(wait_time)


def download_and_prepare_imagenet100(root: str = "./data") -> Path:
    """Download and prepare ImageNet-100 in ImageFolder-compatible structure.

    Extracts to:
        {root}/imagenet-100/train/class_XXX/train_*.png
        {root}/imagenet-100/val/class_XXX/val_*.png
    """
    import time

    data_dir = Path(root) / "imagenet-100"
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    done_marker = data_dir / ".complete.done"

    if done_marker.exists() and train_dir.exists() and val_dir.exists():
        return data_dir

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  PREPARING IMAGENET-100 DATASET FROM HUGGING FACE ({_HF_REPO_ID})")
    print(f"  Target Directory: {data_dir.resolve()}")
    print(f"{'='*70}\n")

    # 1. Download & Extract Validation Shard (1 shard, 5,000 images)
    val_done = data_dir / ".val.done"
    if not val_done.exists() and len(list(val_dir.glob("*/*.png"))) < 5000:
        print("Preparing validation split (5,000 images)...")
        val_file = "data/validation-00000-of-00001.parquet"
        val_parquet = _download_with_retry(repo_id=_HF_REPO_ID, filename=val_file)
        extracted_val = _extract_parquet_shard(Path(val_parquet), val_dir, "val", 0)
        val_done.touch()
        print(f"  [DONE] Extracted {extracted_val:,} validation images across 100 classes.")
    else:
        print("Validation split already extracted (5,000 images), skipping.")

    # 2. Download & Extract Training Shards (17 shards, ~126,689 images)
    print("Preparing training split (17 shards, ~126,689 images)...")
    
    # Auto-detect already existing shards on disk
    existing_shards = set(p.name.split('_')[1] for p in train_dir.glob('*/*.png'))
    for s_id in existing_shards:
        try:
            (data_dir / f".train_shard_{int(s_id):02d}.done").touch()
        except Exception:
            pass

    for shard in range(_NUM_TRAIN_SHARDS):
        shard_done = data_dir / f".train_shard_{shard:02d}.done"
        if shard_done.exists() or f"{shard:02d}" in existing_shards:
            print(f"  Train shard {shard+1}/{_NUM_TRAIN_SHARDS} already extracted, skipping.")
            continue

        train_file = f"data/train-{shard:05d}-of-00017.parquet"
        print(f"  Downloading & extracting train shard {shard+1}/{_NUM_TRAIN_SHARDS}...")
        train_parquet = _download_with_retry(repo_id=_HF_REPO_ID, filename=train_file)
        extracted = _extract_parquet_shard(Path(train_parquet), train_dir, "train", shard)
        shard_done.touch()
        print(f"    -> Extracted {extracted:,} images from shard {shard+1}.")
        # Gentle pacing between shard downloads to avoid rate limits
        time.sleep(5)

    done_marker.touch()
    current_train = len(list(train_dir.glob('*/*.png')))
    current_val = len(list(val_dir.glob('*/*.png')))
    print(f"\n[SUCCESS] ImageNet-100 preparation complete: {current_train:,} train, {current_val:,} val.\n")
    return data_dir


def get_imagenet100_transforms(is_train: bool = True) -> transforms.Compose:
    """Standard ImageNet transforms for 224x224 input."""
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])


def get_imagenet100_loaders(
    root: str = "./data",
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) for ImageNet-100 at 224x224."""
    data_dir = Path(root) / "imagenet-100"
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if download and (not train_dir.exists() or not val_dir.exists() or len(list(val_dir.glob("class_*"))) < 100):
        data_dir = download_and_prepare_imagenet100(root)
        train_dir = data_dir / "train"
        val_dir = data_dir / "val"

    train_dataset = datasets.ImageFolder(
        str(train_dir),
        transform=get_imagenet100_transforms(is_train=True)
    )
    val_dataset = datasets.ImageFolder(
        str(val_dir),
        transform=get_imagenet100_transforms(is_train=False)
    )

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader
