"""
TSR Training Script — Main entry point.

Usage:
    # Train TSR on CIFAR-10 with default config
    python scripts/train.py

    # Override config values
    python scripts/train.py --dataset cifar100 --max-epochs 100 --seed 123

    # Quick smoke test (1000 steps)
    python scripts/train.py --smoke-test
"""

import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tsr.model import TSRNetwork
from tsr.utils import count_parameters
from training.trainer import TSRTrainer
from data.cifar import get_cifar10_loaders, get_cifar100_loaders


def setup_logging(level: str = "INFO"):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    default_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs",
        "default.yaml",
    )
    path = config_path or default_path

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


def main():
    parser = argparse.ArgumentParser(description="Train TSR Network")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset: cifar10, cifar100")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--max-epochs", type=int, default=None, help="Max training epochs")
    parser.add_argument("--max-steps", type=int, default=None, help="Max training steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device: auto, cuda, cpu")
    parser.add_argument("--smoke-test", action="store_true", help="Quick 1000-step test")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger("tsr.train")

    # Load and override config
    config = load_config(args.config)

    if args.dataset:
        config["training"]["dataset"] = args.dataset
    if args.seed:
        config["experiment"]["seed"] = args.seed
    if args.max_epochs:
        config["training"]["max_epochs"] = args.max_epochs
    if args.max_steps:
        config["training"]["max_steps"] = args.max_steps
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.smoke_test:
        config["training"]["max_steps"] = 1000
        config["training"]["eval_interval"] = 200
        config["regulation"]["update_interval"] = 100
        config["regulation"]["monitor_window"] = 50

    # Set seed
    seed = config.get("experiment", {}).get("seed", 42)
    set_seed(seed)
    logger.info(f"Seed: {seed}")

    # Device
    device_str = args.device or config.get("experiment", {}).get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info(f"Device: {device}")

    # Data
    dataset = config.get("training", {}).get("dataset", "cifar10")
    batch_size = config.get("training", {}).get("batch_size", 128)
    num_workers = config.get("data", {}).get("num_workers", 4)
    augmentation = config.get("data", {}).get("augmentation", "standard")
    data_root = config.get("data", {}).get("root", "./data")

    logger.info(f"Dataset: {dataset}, batch_size: {batch_size}")

    if dataset == "cifar10":
        train_loader, val_loader = get_cifar10_loaders(
            root=data_root, batch_size=batch_size,
            num_workers=num_workers, augmentation=augmentation,
        )
        num_classes = 10
    elif dataset == "cifar100":
        train_loader, val_loader = get_cifar100_loaders(
            root=data_root, batch_size=batch_size,
            num_workers=num_workers, augmentation=augmentation,
        )
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Model
    model_cfg = config.get("model", {})
    seed_channels = model_cfg.get("seed_channels", [8, 8])
    model = TSRNetwork(
        in_channels=3,
        seed_channels=seed_channels,
        num_classes=num_classes,
        gate_init=model_cfg.get("gate_init", 3.0),
        act_init=model_cfg.get("act_init", "relu"),
        norm_group_size=model_cfg.get("norm_group_size", 8),
    )

    total_params = count_parameters(model)
    logger.info(f"Model: {model.topology_summary()}")
    logger.info(f"Total parameters: {total_params:,}")

    # Train
    trainer = TSRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    results = trainer.train()

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  Best val accuracy: {results.get('best_val_accuracy', 0):.4f}")
    logger.info(f"  Final topology: {results.get('final_topology', 'N/A')}")
    logger.info(f"  Total training FLOPs: {results.get('total_training_flops', 0):,.0f}")
    logger.info(f"  Structural overhead: {results.get('structural_overhead_pct', 0):.2f}%")
    logger.info(f"  Structural events: {results.get('num_structural_events', 0)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
