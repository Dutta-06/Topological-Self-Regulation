"""A deliberately mis-allocated ResNet-18 at the dense parameter count, to test
whether growth repairs a poor starting allocation (rebuttal experiment).

Stage 1 (stem + layer1 groups) is starved to 16 channels and stage 2 to 32;
stage 3 is unchanged; stage 4 is widened uniformly to the smallest width that
restores the dense count. Writes the starting widths for
bench/train_tsrx.py --init-widths, and C2-style specs to train this
allocation as it is (no reallocation) with scripts/run_c2_from.py --specs.

    python scripts/make_misalloc_widths.py
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bench.models import build_model  # noqa: E402
from bench.resize import resize_model_to_widths  # noqa: E402
from tsrx.graph.bundle import build_all_bundles  # noqa: E402
from tsrx.graph.groups import discover_groups  # noqa: E402
from tsrx.graph.trace import trace_model  # noqa: E402

ARCH, DATASET, NCLS = "resnet18", "cifar100", 100
STAGE_WIDTH = {1: 16, 2: 32}  # starved stages; stage 3 kept; stage 4 solved for


def main():
    ex = torch.zeros(2, 3, 32, 32)
    dense_model = build_model(ARCH, NCLS, cifar_stem=True)
    dense = sum(p.numel() for p in dense_model.parameters())
    bundles = build_all_bundles(discover_groups(trace_model(dense_model.eval(), (ex,))), dense_model)
    stage = {}
    for tap, bd in bundles.items():
        names = [s.module_name for s in bd.producer_slots]
        st = {1 if (n == "conv1" or n.startswith("layer1")) else int(n[5]) for n in names if n.startswith(("layer", "conv1"))}
        if not st:  # the classifier output: task-fixed, left alone
            continue
        if len(st) != 1:
            raise SystemExit(f"tap {tap}: ambiguous stage {st} for producers {names}")
        stage[int(tap)] = st.pop()

    def count(w4):
        widths = {t: (STAGE_WIDTH.get(s) or (w4 if s == 4 else int(bundles[t].size))) for t, s in stage.items()}
        m = resize_model_to_widths(build_model(ARCH, NCLS, cifar_stem=True), widths, ex)
        return sum(p.numel() for p in m.parameters()), widths

    lo, hi = 512, 1024
    while lo < hi:  # smallest stage-4 width whose count reaches the dense count
        mid = (lo + hi) // 2
        if count(mid)[0] >= dense:
            hi = mid
        else:
            lo = mid + 1
    params, widths = count(lo)
    out_dir = ROOT / "results" / "misalloc"
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = {"arch": ARCH, "dataset": DATASET, "widths": {str(t): w for t, w in sorted(widths.items())},
           "stage_of_tap": {str(t): s for t, s in sorted(stage.items())}, "params": params,
           "dense_params": dense,
           "description": "stage 1 at 16 and stage 2 at 32 channels, stage 3 unchanged, stage 4 widened "
                          "uniformly to the smallest width reaching the dense count"}
    (out_dir / f"{ARCH}_{DATASET}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    for seed in (42, 43, 44):
        spec = {"arm": "misalloc_static", "arch": ARCH, "dataset": DATASET, "seed": seed,
                "final_widths": {str(t): w for t, w in widths.items()}, "params": params,
                "widths_from": "make_misalloc_widths.py"}
        d = out_dir / "specs" / f"resnet18_{DATASET}"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"misalloc_static_s{seed}.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    print(f"stage-4 width {lo}: {params:,} params vs dense {dense:,} ({params / dense:.4%})")
    print({s: sorted({w for t, w in widths.items() if stage[t] == s}) for s in (1, 2, 3, 4)})


if __name__ == "__main__":
    main()
