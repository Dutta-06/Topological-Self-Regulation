"""Automated Sequential Pipeline for MobileNetV2 CIFAR-100 3-Way Triangulation Benchmark."""

import os
import subprocess
import sys
import time
from pathlib import Path
import torch

TSRX_CKPT = Path("results/tsrx/mobilenetv2_cifar100_two_regime.pt")
STATIC_MATCHED_CKPT = Path("results/static_matched/mobilenetv2_cifar100_two_regime.pt")
REFERENCE_CKPT = Path("results/reference/mobilenetv2_cifar100.pt")


def run_arm2(epochs: int = 100, batch_size: int = 128, num_workers: int = 0):
    print(f"\n{'='*70}", flush=True)
    print("  LAUNCHING ARM 2: TSR-X DYNAMIC PLASTICITY (Online Channel Discovery)", flush=True)
    print("  Architecture: MobileNetV2 | Dataset: CIFAR-100", flush=True)
    print(f"{'='*70}\n", flush=True)
    cmd = [
        sys.executable, "-m", "bench.train_tsrx",
        "--arch", "mobilenet_v2",
        "--dataset", "cifar100",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--num-workers", str(num_workers),
        "--budget-ratio", "0.85",
        "--out", str(TSRX_CKPT),
        "--seed", "42",
        "--amp",
    ]
    print("Command:", " ".join(cmd), flush=True)
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"[ERROR] Arm 2 failed with returncode {res.returncode}")
        sys.exit(res.returncode)
    print("\n[DONE] Arm 2 finished successfully!", flush=True)


def run_arm3(epochs: int = 100, batch_size: int = 128, num_workers: int = 0):
    print(f"\n{'='*70}", flush=True)
    print("  LAUNCHING ARM 3: C2 STATIC MATCHED CONTROL (Trained from Scratch)", flush=True)
    print("  Architecture: MobileNetV2 (Discovered Topology) | Dataset: CIFAR-100", flush=True)
    print(f"{'='*70}\n", flush=True)
    cmd = [
        sys.executable, "-m", "bench.train_static_matched",
        "--tsrx-checkpoint", str(TSRX_CKPT),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--num-workers", str(num_workers),
        "--out", str(STATIC_MATCHED_CKPT),
        "--seed", "42",
        "--amp",
    ]
    print("Command:", " ".join(cmd), flush=True)
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"[ERROR] Arm 3 failed with returncode {res.returncode}")
        sys.exit(res.returncode)
    print("\n[DONE] Arm 3 finished successfully!", flush=True)


def run_arm1(epochs: int = 100, batch_size: int = 128, num_workers: int = 0):
    print(f"\n{'='*70}", flush=True)
    print("  LAUNCHING ARM 1: STANDARD STATIC REFERENCE BASELINE (100% Parameters)", flush=True)
    print("  Architecture: MobileNetV2 | Dataset: CIFAR-100", flush=True)
    print(f"{'='*70}\n", flush=True)
    cmd = [
        sys.executable, "-m", "bench.train_reference",
        "--arch", "mobilenet_v2",
        "--dataset", "cifar100",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--num-workers", str(num_workers),
        "--out", str(REFERENCE_CKPT),
        "--seed", "42",
        "--amp",
    ]
    print("Command:", " ".join(cmd), flush=True)
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"[ERROR] Arm 1 failed with returncode {res.returncode}")
        sys.exit(res.returncode)
    print("\n[DONE] Arm 1 finished successfully!", flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--skip-arm2", action="store_true")
    ap.add_argument("--skip-arm3", action="store_true")
    ap.add_argument("--skip-arm1", action="store_true")
    args = ap.parse_args()

    if not args.skip_arm2:
        run_arm2(epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers)
    if not args.skip_arm3:
        run_arm3(epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers)
    if not args.skip_arm1:
        run_arm1(epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers)

    print("\nEvaluating MobileNetV2 CIFAR-100 3-Way Triangulation Summary...")
    eval_script = Path(__file__).parent / "eval_mobilenetv2_cifar100_triangulation.py"
    if eval_script.exists():
        subprocess.run([sys.executable, str(eval_script)])

    report_script = Path(__file__).parent / "generate_mobilenetv2_cifar100_pdf_report.py"
    if report_script.exists():
        print("\nGenerating MobileNetV2 CIFAR-100 PDF and Markdown Benchmark Reports...")
        subprocess.run([sys.executable, str(report_script)])


if __name__ == "__main__":
    main()
