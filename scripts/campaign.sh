#!/usr/bin/env bash
# TSR-X results campaign: 4 architectures x 3 datasets, 3 arms (reference /
# TSR-X / C2) per point, plus the C3 mechanism control.
#
# "3 arms" matches vision's reporting structure exactly: reference is the
# baseline being beaten, TSR-X is the discovery run (search cost, not the
# headline), C2 (discovered widths trained from scratch) is THE claim. C3
# (random widths, same budget) is a control layered on top to show the
# topological SIGNAL did the work, not just non-uniform widths -- it is not
# counted as a 4th "arm" any more than it was in the vision reports.
#
# Why 4 architectures, not the internal TCN alone: a channel-mixing TCN
# scored test MSE 0.4946 on weather/h96 against PatchTST's published 0.1525
# -- a non-competitive, non-standard architecture. "We compress our own weak
# model by 70%" says nothing about whether the method works on anything a
# reviewer recognizes. The four here are chosen for real published numbers
# AND genuine architectural diversity (conv / patch-attention /
# variate-attention / all-MLP), not piled on for their own sake:
#
#   tcn           channel-mixing conv (kept: already validated, cheap, shows
#                 the method also works on a simple non-SOTA baseline)
#   patchtst      patch + channel-independent attention (Nie et al. 2023)
#   itransformer  variate-token attention (Liu et al. 2024) -- reuses
#                 PatchTST's exact block/safety-probe machinery
#   tsmixer       all-MLP, BatchNorm not LayerNorm by the paper's OWN design
#                 (Chen et al. 2023) -- ceiling scales with n_vars, so its
#                 budget list is generated per-dataset, not hand-picked
#
# DLinear was considered and rejected: it is one Linear(seq_len,pred_len)
# with no hidden/channel dimension at all, so there is nothing for TSR-X to
# reallocate -- not a matter of tuning, a structural mismatch with the
# method. TimesNet was considered and set aside: its top-k FFT frequency
# selection produces data-dependent shapes that risk breaking torch.fx's
# static tracing, and iTransformer -- already in the canonical 4, already
# built here -- was the safer swap.
#
# Ceilings measured before this file was written (`--min-size 8`, budget
# floor = 1 - ceiling):
#   patchtst      ~35% (all datasets -- channel-independent, n_vars-agnostic)
#   itransformer  ~45% (all datasets -- also n_vars-agnostic)
#   tsmixer       weather 47% / electricity 90% / traffic 94% (scales with
#                 n_vars because feature-mixing mixes ACROSS variates)
#   tcn           weather 91% / electricity 87% / traffic 86%
#
# Resumable: every step skips if its checkpoint exists. Ordered cheapest
# arch/dataset first so an interrupted run still yields complete cheap cells
# rather than a uniformly half-finished grid.
#
#   bash scripts/campaign.sh 2>&1 | tee campaign.log
#
# Env overrides: EPOCHS, DEV, PY, STAGES (e.g. STAGES="1 2"), ARCHS, DATASETS.

set -u -o pipefail

EPOCHS=${EPOCHS:-50}
END_FRAC=${END_FRAC:-0.3}   # NOT 0.5 -- the search must converge before early
                            # stopping bites, or C2/C3 train a barely-pruned
                            # architecture. That bug voided a whole campaign.
DEV=${DEV:-cuda}
PY=${PY:-python}
SEED=${SEED:-42}
STAGES=${STAGES:-"1 2 3"}
DATASETS=${DATASETS:-"weather electricity traffic"}
ARCHS=${ARCHS:-"patchtst itransformer tcn tsmixer"}   # cheapest-ish first
H=${H:-96}

mkdir -p results/ts_sweep results/ts_reference logs

bs_for () { case "$1" in traffic) echo 8 ;; electricity) echo 16 ;; *) echo 32 ;; esac; }
has_stage () { [[ " $STAGES " == *" $1 "* ]]; }

# The first (TCN) sweep already completed under the ORIGINAL untagged naming
# (results/ts_sweep/tsrx_<ds>_h<H>_br<BR>.pt, no arch prefix). Keep TCN on
# that convention so this script reuses those ~12h of finished runs instead
# of silently re-running them under a new tsrx_tcn_... filename. New
# architectures get a real tag so their files never collide with each other.
tag_for () { [ "$1" = "tcn" ] && echo "" || echo "$1"; }
sweep_tag () { local t; t=$(tag_for "$1"); [ -n "$t" ] && echo "${t}_$2_h$3" || echo "$2_h$3"; }

# Per-arch hyperparameter flags. lr=1e-4 for attention models (1e-3
# diverges), 1e-3 for conv/MLP. seq_len=336 for the transformer family
# (their published numbers use it); the TCN/TSMixer path was validated at
# 96 and changing it now would invalidate the existing TCN sweep data.
arch_flags () {
  case "$1" in
    patchtst)     echo "--seq-len 336 --lr 1e-4 --d-model 128 --d-ff 512 --n-heads 8 --n-blocks 3 --patch-len 16 --stride 8" ;;
    itransformer) echo "--seq-len 336 --lr 1e-4 --d-model 128 --d-ff 256 --n-heads 8 --n-blocks 3" ;;
    tcn)          echo "--seq-len 96  --lr 1e-3" ;;
    tsmixer)      echo "--seq-len 96  --lr 1e-3 --d-ff 256 --n-blocks 4" ;;
    *) echo "unknown arch $1" >&2; exit 1 ;;
  esac
}

# Budget lists per arch's measured ceiling (see header). TSMixer's ceiling
# depends on n_vars, so its list is generated per-dataset rather than fixed.
budgets_for () {
  local arch=$1 ds=$2
  case "$arch" in
    patchtst)     echo "0.95 0.90 0.85 0.80 0.75 0.70 0.66" ;;
    itransformer) echo "0.95 0.90 0.85 0.80 0.75 0.65 0.56" ;;
    tcn)          echo "0.85 0.50 0.30 0.20 0.15 0.10" ;;
    tsmixer)
      case "$ds" in
        weather)     echo "0.90 0.80 0.70 0.60 0.55" ;;
        electricity) echo "0.85 0.60 0.40 0.25 0.15" ;;
        traffic)     echo "0.85 0.55 0.35 0.20 0.10" ;;
      esac ;;
  esac
}

run () {  # run <out.pt> <module+args...>
  local out="$1"; shift
  if [ -f "$out" ]; then echo "    skip (exists): $out"; return 0; fi
  local log="logs/$(basename "${out%.pt}").log"
  echo "    >>> $(date +%H:%M:%S) $out"
  if ! $PY "$@" --out "$out" > "$log" 2>&1; then
    echo "    !!! FAILED: $out"; tail -4 "$log"; return 1
  fi
}

# TSR-X -> C2 for one (arch, dataset, budget, seed). C2/C3 read lr/d_model/
# etc. from the TSR-X checkpoint's own args (bench/train_ts_static_matched.py
# and train_c3_random.py both inherit rather than default), so this does NOT
# need arch_flags again -- passing the wrong lr to a matched control is
# exactly the class of bug that silently invalidates a result.
triple () {
  local arch=$1 ds=$2 br=$3 seed=$4 c3seed=${5:-} suffix=${6:-}
  local bs; bs=$(bs_for "$ds")
  local tag="$(sweep_tag "$arch" "$ds" "$H")_br${br}"
  [ "$seed" != "$SEED" ] && tag="${tag}_s${seed}"
  local T="results/ts_sweep/tsrx_${tag}.pt"

  run "$T" -m bench.train_ts_tsrx --arch "$arch" $(arch_flags "$arch") --dataset "$ds" \
      --pred-len "$H" --epochs "$EPOCHS" --batch-size "$bs" --budget-ratio "$br" \
      --budget-end-frac "$END_FRAC" --seed "$seed" --device "$DEV" || return 0
  [ -f "$T" ] || { echo "    (no TSR-X checkpoint for $tag -- infeasible budget, skipping C2/C3)"; return 0; }

  run "results/ts_sweep/c2_${tag}.pt" -m bench.train_ts_static_matched \
      --tsrx-checkpoint "$T" --epochs "$EPOCHS" --seed "$seed" --device "$DEV"

  if [ -n "$c3seed" ]; then
    run "results/ts_sweep/c3_${tag}${suffix}.pt" -m bench.train_c3_random \
        --tsrx-checkpoint "$T" --domain ts --epochs "$EPOCHS" \
        --seed "$seed" --c3-seed "$c3seed" --device "$DEV"
  else
    run "results/ts_sweep/c3_${tag}.pt" -m bench.train_c3_random \
        --tsrx-checkpoint "$T" --domain ts --epochs "$EPOCHS" \
        --seed "$seed" --device "$DEV"
  fi
}

reference_for () {
  local arch=$1 ds=$2 bs; bs=$(bs_for "$ds")
  run "results/ts_reference/${arch}_${ds}_h${H}.pt" -m bench.train_ts_reference \
      --arch "$arch" $(arch_flags "$arch") --dataset "$ds" --pred-len "$H" \
      --epochs "$EPOCHS" --batch-size "$bs" --seed "$SEED" --device "$DEV"
}

# ------------------------------------------------------------------ Stage 1
if has_stage 1; then
  echo "=== STAGE 1: compression frontier per architecture — locate each knee ==="
  for arch in $ARCHS; do
    for ds in $DATASETS; do
      echo "  [$arch / $ds] reference"
      reference_for "$arch" "$ds"
      for br in $(budgets_for "$arch" "$ds"); do
        echo "  [$arch / $ds br=$br]"
        triple "$arch" "$ds" "$br" "$SEED"
      done
      echo "--- frontier so far: $arch / $ds ---"
      $PY -m analysis.sweep_table --arch "$(tag_for "$arch")" --dataset "$ds" --horizon "$H" \
          --reference "results/ts_reference/${arch}_${ds}_h${H}.pt" 2>/dev/null || true
    done
  done
fi

# Tightest budget whose C2 test MSE stays within TOL of the reference. Fixed
# tolerance declared up front rather than argmin — Stage 1's own deltas were
# already shown to be non-monotone in compression on the first (TCN) sweep,
# so picking the best-looking point would be selecting on noise.
detect_knee () {
  local arch=$1 ds=$2 tag; tag=$(sweep_tag "$arch" "$ds" "$H")
  $PY - "$arch" "$ds" "$tag" <<'EOF'
import glob, os, re, sys, torch
arch, ds, tag = sys.argv[1], sys.argv[2], sys.argv[3]
TOL = float(os.environ.get("TOL", "0.03"))
ref = None
for p in glob.glob(f"results/ts_reference/{arch}_{ds}_h96.pt"):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    ref = ck.get("test_mse"); break
if ref is None: print(""); raise SystemExit
best = ""
for p in sorted(glob.glob(f"results/ts_sweep/c2_{tag}_br*.pt")):
    m = re.search(r"_br([0-9.]+)\.pt$", p)
    if not m: continue
    ck = torch.load(p, map_location="cpu", weights_only=False)
    t = ck.get("test_mse")
    if t is None: continue
    if (t - ref) / ref <= TOL:
        if best == "" or float(m.group(1)) < float(best):
            best = m.group(1)
print(best)
EOF
}

# ------------------------------------------------------------------ Stage 2
if has_stage 2; then
  echo "=== STAGE 2: error bars at each knee (seeds 43, 44) ==="
  for arch in $ARCHS; do
    for ds in $DATASETS; do
      K=$(detect_knee "$arch" "$ds")
      if [ -z "$K" ]; then echo "  [$arch/$ds] no budget within tolerance — skipping"; continue; fi
      echo "  [$arch/$ds] knee budget = $K"
      for s in 43 44; do triple "$arch" "$ds" "$K" "$s"; done
    done
  done
fi

# ------------------------------------------------------------------ Stage 3
if has_stage 3; then
  echo "=== STAGE 3: mechanism robustness — extra C3 random draws at each knee ==="
  for arch in $ARCHS; do
    for ds in $DATASETS; do
      K=$(detect_knee "$arch" "$ds")
      [ -z "$K" ] && continue
      T="results/ts_sweep/tsrx_$(sweep_tag "$arch" "$ds" "$H")_br${K}.pt"
      [ -f "$T" ] || continue
      for cs in 1 2 3; do
        run "results/ts_sweep/c3_$(sweep_tag "$arch" "$ds" "$H")_br${K}_c${cs}.pt" -m bench.train_c3_random \
            --tsrx-checkpoint "$T" --domain ts --epochs "$EPOCHS" \
            --seed "$SEED" --c3-seed "$cs" --device "$DEV"
      done
    done
  done
fi

# ------------------------------------------------------------------ report
echo
echo "=== FINAL FRONTIERS ==="
{
for arch in $ARCHS; do
  for ds in $DATASETS; do
    echo
    $PY -m analysis.sweep_table --arch "$(tag_for "$arch")" --dataset "$ds" --horizon "$H" \
        --reference "results/ts_reference/${arch}_${ds}_h${H}.pt" || true
  done
done
} | tee results/ts_frontier.md
echo
echo "DONE $(date)"
