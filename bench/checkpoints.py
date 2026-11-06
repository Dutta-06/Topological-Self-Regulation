"""Checkpoint contract for the benchmark arms.

A reference checkpoint is written repeatedly while its run is in progress --
`train_reference` re-saves on every improving epoch -- so a consumer that simply
opens the file cannot tell a finished run from one that is still training. That
is how nine of thirteen published reference accuracies came to be understated:
nothing downstream could distinguish "best so far" from "final", and the two
readers that wanted a best-accuracy field silently defaulted to 0.0 when it was
absent instead of failing.

This module makes both failures loud:

  * `save_atomic` writes to a temporary file and renames, so a partially written
    checkpoint is never observable under the real name;
  * `require_complete` refuses a checkpoint that does not carry `complete: True`;
  * `best_accuracy` refuses one that has no recorded best, rather than reporting
    zero.

Checkpoints written before this contract existed can be brought up to it with
`scripts/stamp_reference_checkpoints.py`, which only stamps files whose accuracy
has actually been re-measured.
"""

import os
from pathlib import Path
from typing import Optional

import torch


class IncompleteCheckpointError(RuntimeError):
    """A checkpoint was read that does not record a finished training run."""


def save_atomic(payload: dict, path) -> None:
    """Save so that `path` never holds a half-written file.

    torch.save straight to the destination leaves a window in which a reader
    sees a truncated or torn file; worse, it leaves no way to tell a file that
    is mid-write from one that is merely mid-training.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def require_complete(ckpt: dict, path) -> dict:
    """Return `ckpt`, or raise if it does not record a finished run."""
    if not ckpt.get("complete"):
        epoch = ckpt.get("epoch")
        total = (ckpt.get("args") or {}).get("epochs")
        raise IncompleteCheckpointError(
            f"{path} does not carry `complete: True`, so it cannot be "
            f"distinguished from a run that is still training "
            f"(last recorded epoch {epoch} of {total}). Re-run it with the "
            f"current bench/train_reference.py, or stamp it with "
            f"scripts/stamp_reference_checkpoints.py once its accuracy has "
            f"been re-measured."
        )
    return ckpt


def best_accuracy(ckpt: dict, path) -> float:
    """The run's best validation accuracy, or raise.

    Never defaults. `ckpt.get("best_val_acc", 0.0)` is what made every sequence
    runner print the reference arm as 0.00%.
    """
    for key in ("best_val_acc", "val_acc"):
        if key in ckpt and ckpt[key] is not None:
            return float(ckpt[key])
    raise KeyError(
        f"{path} records neither `best_val_acc` nor `val_acc`; refusing to "
        f"report an accuracy for it."
    )


def load_reference(path, expect_cifar_stem: Optional[bool] = None) -> dict:
    """Load a reference checkpoint, insisting it is finished and well formed.

    `expect_cifar_stem` additionally checks the stem the checkpoint was trained
    with against the one the caller is about to build, which is the mismatch
    that produced the stale Tiny-ImageNet MobileNetV2 and EfficientNet numbers.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no reference checkpoint at {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    require_complete(ckpt, path)

    if expect_cifar_stem is not None:
        stem = ckpt.get("cifar_stem", (ckpt.get("args") or {}).get("cifar_stem"))
        if stem is None:
            raise IncompleteCheckpointError(
                f"{path} does not record which stem it was trained with; "
                f"refusing to guess."
            )
        if bool(stem) != bool(expect_cifar_stem):
            raise ValueError(
                f"{path} was trained with cifar_stem={bool(stem)}, but the "
                f"caller is building cifar_stem={bool(expect_cifar_stem)}."
            )
    return ckpt
