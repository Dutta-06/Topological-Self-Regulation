# Topological Self-Regulation (TSR)

**A neural network that modifies its own intrinsic structural properties during training.**

TSR networks continuously adjust their topology — neuron count, connectivity patterns, layer depth, and activation functions — driven by intrinsic gradient and activation signals. The network is both the learner and the architect.

## Key Features

- **Bidirectional structural change**: neurons are born and pruned during training
- **Differentiable gating**: soft neuron gates enable gradient-based importance estimation
- **Learnable activation mixing**: per-layer activation functions adapt to the task
- **Two-timescale optimization**: fast weight updates + slow structural updates
- **Compute-optimal convergence**: grows capacity precisely when and where needed

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Train TSR on CIFAR-10
python scripts/train.py model=tsr dataset=cifar10

# Run benchmark suite
python scripts/sweep.py experiment=aaai_submission
```

## Project Structure

```
tsr/                 # Core TSR library
  layers/            # TSR-aware layers (Linear, Conv2d, Norm)
  regulation/        # Structural plasticity monitor & actions
  model.py           # TSRNetwork assembly
  topology.py        # Topology state & serialization
  flops.py           # FLOPs tracking
baselines/           # Baseline implementations
benchmarks/          # AAAI benchmark runners
data/                # Data loading & augmentation
training/            # Training loop, evaluation, logging
analysis/            # Visualization & paper figures
configs/             # Hydra configuration files
tests/               # Unit & integration tests
```

## Target Venue

**AAAI** — 7 pages + references. Implementation: PyTorch only.

## Citation

Paper in preparation.
