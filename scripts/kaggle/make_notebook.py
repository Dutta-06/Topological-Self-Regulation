"""Write scripts/kaggle/ablations_t4x2.ipynb: one slice of the growth-ablation
campaign on a Kaggle T4x2 session.

Kaggle settings: Accelerator "GPU T4 x2", Internet ON. Run with "Save Version ->
Save & Run All (Commit)" so it runs headless and /kaggle/working is kept as the
version's output; download that output and copy ablations/<cell>/* into the
local results/ablations/<cell>/ (scripts/eval_ablations.py reads them as-is).

    python scripts/kaggle/make_notebook.py --cells vgg16_bn/cifar100 --seeds 44
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip().splitlines(keepends=True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=["vgg16_bn/cifar100"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[44])
    ap.add_argument("--arms", nargs="*", default=None, help="default: all arms")
    ap.add_argument("--branch", default="rebuttal-ablations")
    ap.add_argument("--repo", default="https://github.com/Dutta-06/Topological-Self-Regulation.git")
    ap.add_argument("--hours", type=float, default=8.5,
                    help="start no new job after this; leaves room for one ~3 h job under the 12 h cap")
    ap.add_argument("--runner", choices=["ablations", "sensitivity"], default="ablations")
    ap.add_argument("--only", nargs="*", default=None, help="sensitivity settings for this session")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        tag = "ablations" if args.runner == "ablations" else "sensitivity_" + "_".join(args.only or ["all"])
        args.out = str(HERE / f"{tag}_t4x2.ipynb")

    arms = f" --arms {' '.join(args.arms)}" if args.arms else ""
    cells = " ".join(args.cells)
    seeds = " ".join(map(str, args.seeds))
    if args.runner == "ablations":
        title = f"growth ablations: {cells}, seeds {seeds}"
        run_cmd = (f"python scripts/run_ablations.py --cells {cells} --seeds {seeds}{arms} \\\n"
                   f"    --gpus 0 1 --num-workers 2 --stop-starting-after {args.hours} \\\n"
                   f"    --data-root /kaggle/working/data --out-root /kaggle/working/ablations")
        eval_cmd = "python scripts/eval_ablations.py --root /kaggle/working/ablations --metric final"
        out_dir = "ablations"
    else:
        only = f" --only {' '.join(args.only)}" if args.only else ""
        title = f"sensitivity sweep: {args.cells[0]}, seed {args.seeds[0]}{only}"
        run_cmd = (f"python scripts/run_sensitivity.py --cell {args.cells[0]} --seed {args.seeds[0]}{only} \\\n"
                   f"    --gpus 0 1 --num-workers 2 --stop-starting-after {args.hours} \\\n"
                   f"    --data-root /kaggle/working/data --out-root /kaggle/working/sensitivity")
        eval_cmd = "ls -la /kaggle/working/sensitivity/*/"
        out_dir = "sensitivity"
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                     "language_info": {"name": "python"}},
        "cells": [
            md(f"""
# TSR-X {title}

Two jobs at a time, one per T4. Settings: **GPU T4 x2**, **Internet on**, run as a
committed version (Save Version -> Save & Run All). Outputs land in
`/kaggle/working/{out_dir}`; copy `{out_dir}/<cell>/*` into the local `results/{out_dir}/<cell>/`.
If the repository is private, add a Kaggle secret `GITHUB_TOKEN` (read access)."""),
            code(f"""
import os, subprocess
repo, branch = "{args.repo}", "{args.branch}"
try:
    from kaggle_secrets import UserSecretsClient
    tok = UserSecretsClient().get_secret("GITHUB_TOKEN")
    repo = repo.replace("https://", f"https://{{tok}}@")
except Exception:
    pass  # public repository, or no secret set
if not os.path.exists("/kaggle/working/tsr"):
    subprocess.run(["git", "clone", "--depth", "1", "--branch", branch, repo, "/kaggle/working/tsr"], check=True)
os.chdir("/kaggle/working/tsr")
print(subprocess.run(["git", "log", "-1", "--oneline"], capture_output=True, text=True).stdout)"""),
            code("""
import torch
print(torch.__version__, torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
assert torch.cuda.device_count() >= 2, "select the GPU T4 x2 accelerator"
"""),
            code(f"""
# Smoke test first: a 500-step discovery run must finish cleanly before the slice starts.
!cd /kaggle/working/tsr && python -m bench.train_tsrx --arch resnet18 --dataset cifar100 --smoke-test --epochs 1 \\
    --data-root /kaggle/working/data --num-workers 2 --out /kaggle/working/smoke/r18.pt | tail -6"""),
            code(f"""
!cd /kaggle/working/tsr && {run_cmd}"""),
            code(f"""
!cd /kaggle/working/tsr && {eval_cmd}
!rm -rf /kaggle/working/data /kaggle/working/smoke /kaggle/working/tsr   # keep the version output small
!du -sh /kaggle/working/{out_dir}"""),
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
