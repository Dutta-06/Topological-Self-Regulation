#!/bin/bash
# Runs Experiment 1 (ceiling-break, no budget) then Experiment 2 (Pareto, λ-FLOPs).
# Outputs go to:
#   gate_v3_ceiling/   — Experiment 1 results
#   gate_v3_pareto/    — Experiment 2 results
#
# Usage:
#   bash run_experiments.sh              # full run, 3 seeds, 100 epochs each
#   bash run_experiments.sh --smoke      # 1 seed, 5 epochs (sanity check)
#   bash run_experiments.sh --exp1-only  # skip Exp 2
#   bash run_experiments.sh --exp2-only  # skip Exp 1 (Exp 1 must already be done)

set -e

SEEDS="42 123 456"
EPOCHS=100
DEVICE="auto"
DATA_ROOT="./data"
FLOPS_PRICE="1e-8"   # λ for Experiment 2 — tune if model grows too large/small

SMOKE=false
EXP1=true
EXP2=true

for arg in "$@"; do
    case $arg in
        --smoke)     SMOKE=true; SEEDS="42"; EPOCHS=5 ;;
        --exp1-only) EXP2=false ;;
        --exp2-only) EXP1=false ;;
    esac
done

source .venv/bin/activate

# ── Workstream A: sanity check — does a real ResNet clear 90% under this recipe? ──
# Cheap, single-seed. Run this FIRST. If it also caps ~90%, the ceiling is the
# recipe (aug/schedule/head), not topology — fix that before trusting Experiment 1.
if $EXP1; then
    echo ""
    echo "========================================================"
    echo "WORKSTREAM A — ResNet sanity check (1 seed, confirms residuals help)"
    echo "Results → gate_v3_sanity/"
    echo "========================================================"
    python scripts/gate_experiment.py \
        --variants resnet_sanity \
        --seeds 42 \
        --epochs $EPOCHS \
        --data-root $DATA_ROOT \
        --results-dir gate_v3_sanity \
        --device $DEVICE
fi

# ── Experiment 1: ceiling-break (free growth, no λ penalty) ─────────────────
if $EXP1; then
    echo ""
    echo "========================================================"
    echo "EXPERIMENT 1 — Ceiling-break (flops_price=0.0)"
    echo "Results → gate_v3_ceiling/"
    echo "========================================================"
    python scripts/gate_experiment.py \
        --seeds $SEEDS \
        --epochs $EPOCHS \
        --data-root $DATA_ROOT \
        --results-dir gate_v3_ceiling \
        --device $DEVICE
fi

# ── Experiment 2: Pareto (λ-FLOPs budget pressure) ──────────────────────────
if $EXP2; then
    echo ""
    echo "========================================================"
    echo "EXPERIMENT 2 — Pareto (flops_price=$FLOPS_PRICE)"
    echo "Results → gate_v3_pareto/"
    echo "========================================================"
    python scripts/gate_experiment.py \
        --seeds $SEEDS \
        --epochs $EPOCHS \
        --data-root $DATA_ROOT \
        --results-dir gate_v3_pareto \
        --flops-price $FLOPS_PRICE \
        --device $DEVICE
fi

echo ""
echo "Done. Send back gate_v3_ceiling/ and gate_v3_pareto/ directories."
