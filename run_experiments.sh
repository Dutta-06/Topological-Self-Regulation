#!/bin/bash
# ONE command, everything: TSR (phantom + heuristic ablation), every fixed
# baseline (vgg_tiny, vgg_small, static_final, vgg16, vgg19), and the residual
# competitor (resnet_sanity) — 3 seeds, 100 epochs, all in a single results dir.
#
# Variants run (in this order; static_final auto-uses TSR's discovered shape+skips):
#   tsr_phantom    — TSR, the actual contribution
#   tsr_heuristic  — TSR with the old bottleneck-score signal (ablation)
#   vgg_tiny       — FixedVGG([8,8,16])
#   vgg_small      — FixedVGG([16,16,32])
#   static_final   — FixedVGG matching TSR's discovered shape + skip connections
#   vgg16 / vgg19  — standard deep VGG configs (large, ~15-20M params)
#   resnet_sanity  — ResNet-20-style with real residual shortcuts (competitor)
#
# Usage:
#   bash run_experiments.sh              # full run: all variants, 3 seeds, 100 epochs
#   bash run_experiments.sh --smoke      # 1 seed, 5 epochs (fast sanity check)
#   bash run_experiments.sh --with-pareto  # also run the lambda-FLOPs Pareto experiment after
#
# NOTE: vgg16/vgg19 are ~15-20M params each; expect the full 3-seed x 100-epoch
# run to take several hours on a single GPU (RTX Titan / ADA 2000 class).

set -e

SEEDS="42 123 456"
EPOCHS=100
DEVICE="auto"
DATA_ROOT="./data"
FLOPS_PRICE="1e-8"   # lambda for the optional Pareto experiment; tune if needed

WITH_PARETO=false

for arg in "$@"; do
    case $arg in
        --smoke)       SEEDS="42"; EPOCHS=5 ;;
        --with-pareto) WITH_PARETO=true ;;
    esac
done

source .venv/bin/activate

echo ""
echo "========================================================"
echo "FULL GATE EXPERIMENT — all variants, seeds=[$SEEDS], epochs=$EPOCHS"
echo "tsr_phantom tsr_heuristic vgg_tiny vgg_small static_final vgg16 vgg19 resnet_sanity"
echo "Results -> gate_v3_full/"
echo "========================================================"
python scripts/gate_experiment.py \
    --seeds $SEEDS \
    --epochs $EPOCHS \
    --data-root $DATA_ROOT \
    --results-dir gate_v3_full \
    --device $DEVICE

if $WITH_PARETO; then
    echo ""
    echo "========================================================"
    echo "PARETO EXPERIMENT — flops_price=$FLOPS_PRICE (budget-pressure ablation)"
    echo "Results -> gate_v3_pareto/"
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
echo "Done."
echo "Full comparison table + gate check: gate_v3_full/summary.json"
echo "Per-run detail (topology, events, skip connections): gate_v3_full/<variant>/seed<N>/final.json"
