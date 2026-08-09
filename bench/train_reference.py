"""Train a plain torchvision reference architecture (no TSR anywhere) and
checkpoint it, so bench/gate.py has something real to measure.

training/trainer.py is TSRNetwork-specific (calls model.topology_summary()
etc.) and none of the three TSR benchmark harnesses save checkpoints at
all — this is a new, minimal, standard training loop for exactly this gap.

Usage:
    python -m bench.train_reference --arch resnet18 --dataset cifar100 \
        --epochs 200 --out results/reference/resnet18_cifar100.pt
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision


def build_model(arch: str, num_classes: int) -> nn.Module:
    fn = {"resnet18": torchvision.models.resnet18, "vgg16_bn": torchvision.models.vgg16_bn}[arch]
    return fn(num_classes=num_classes)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["resnet18", "vgg16_bn"], required=True)
    ap.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--augmentation", default="standard")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True, help="checkpoint path (.pt)")
    ap.add_argument("--checkpoint-every", type=int, default=0, help="0 = only save at the end/best")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from data.cifar import get_cifar10_loaders, get_cifar100_loaders
    num_classes = 10 if args.dataset == "cifar10" else 100
    loader_fn = get_cifar10_loaders if args.dataset == "cifar10" else get_cifar100_loaders
    train_loader, val_loader = loader_fn(root=args.data_root, batch_size=args.batch_size,
                                          augmentation=args.augmentation)

    model = build_model(args.arch, num_classes).to(args.device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                           weight_decay=args.weight_decay, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        total_loss, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
        sched.step()

        val_acc = evaluate(model, val_loader, args.device)
        is_best = val_acc > best_acc
        best_acc = max(best_acc, val_acc)
        print(f"epoch {epoch+1:3d}/{args.epochs}  train_loss={total_loss/n:.4f}  "
              f"val_acc={val_acc:.4f}  best={best_acc:.4f}  ({time.time()-t0:.1f}s)")

        if is_best:
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "arch": args.arch,
                "dataset": args.dataset,
                "args": vars(args),
            }, out_path)

        if args.checkpoint_every and (epoch + 1) % args.checkpoint_every == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": sched.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
            }, out_path.with_name(out_path.stem + f"_epoch{epoch+1}.pt"))

    print(f"\nDONE. best val_acc={best_acc:.4f}  checkpoint={out_path}")


if __name__ == "__main__":
    main()
