"""Stage 0 — Prediction 9.1: is a conventionally-trained reference
architecture equimarginal?

No training happens here. Load a trained checkpoint, attach candidates,
run backward passes over held-out data to read growth/removal densities,
and report the spread. Theorem 4.7 predicts max(gamma) > min(rho) by a
substantial margin; near-uniform densities falsify the enhancement thesis
(paper/tsr-framework.tex sec:thesis) and the program stops before any
further engineering.

Simplification for this cheap diagnostic (documented, not hidden):
gamma_ell = max_c ||u_c|| / kappa_ell and rho_ell = min_j |<u_j,v_j>| /
kappa_ell — first-order only, no h_c second-order correction (that needs
one extra forward pass PER CANDIDATE PER GROUP and is deferred to the
exchange operator, M6, where it prices an actual accept/reject decision).
Detecting gross equimarginal spread does not need that precision.
"""

import argparse
import json
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from tsrx.alloc.cost import kappa_params
from tsrx.graph.bundle import build_all_bundles
from tsrx.graph.groups import discover_groups
from tsrx.graph.trace import trace_model
from tsrx.sense.candidates import CandidateBank, UnsupportedGroupError
from tsrx.sense.saliency import first_order_saliency
from tsrx.sense.topo import WindowedSignal, compute_uc_norms


def run_gate(
    model: nn.Module,
    loader,
    loss_fn,
    k: int = 8,
    n_batches: int = 50,
    device: str = "cuda",
    eval_mode: bool = True,
) -> List[dict]:
    model = model.to(device)
    model.eval() if eval_mode else model.train()

    it = iter(loader)
    xb0, _ = next(it)
    traced = trace_model(model, (xb0[:2].to(device),))
    res = discover_groups(traced)
    bundles = build_all_bundles(res, model)

    skipped = []
    bank = CandidateBank.__new__(CandidateBank)
    nn.Module.__init__(bank)
    object.__setattr__(bank, "model", model)
    bank.k = k
    bank.handles = {}
    for tap, bd in bundles.items():
        if bd.size == 0 or not bd.producer_slots or not bd.consumer_slots:
            continue
        try:
            bank._attach(bd)
        except UnsupportedGroupError as e:
            skipped.append(str(e))
    if skipped:
        print(f"[gate] skipped {len(skipped)} group(s) (LN/GN, Tier 2 not yet supported):")
        for s in skipped:
            print(f"  - {s}")
    bank = bank.to(device)

    win = WindowedSignal(window=n_batches)
    saliency_sum: Dict[int, torch.Tensor] = {tap: None for tap in bank.handles}
    n_seen = 0

    batches = [(xb0, None)]  # placeholder, replaced below
    it = iter(loader)
    for step in range(n_batches):
        try:
            xb, yb = next(it)
        except StopIteration:
            break
        xb, yb = xb.to(device), yb.to(device)
        model.zero_grad(set_to_none=True)
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()

        for tap, h in bank.handles.items():
            u = compute_uc_norms(bank, tap)
            win.record(tap, u)
            sal = first_order_saliency(model, h.bundle, h.base_size)
            saliency_sum[tap] = sal if saliency_sum[tap] is None else saliency_sum[tap] + sal
        n_seen += 1

    report = []
    for tap, h in bank.handles.items():
        kp = kappa_params(h.bundle, model)
        if kp <= 0:
            continue
        gamma = win.best(tap)
        rho_vec = (saliency_sum[tap] / max(n_seen, 1)).cpu()
        rho = float(rho_vec.min().item())
        producer_names = sorted(set(s.module_name for s in h.bundle.producer_slots))
        report.append({
            "tap": tap,
            "producers": producer_names,
            "size": h.base_size,
            "kappa_params": kp,
            "gamma": gamma / kp,
            "rho": rho / kp,
        })

    report.sort(key=lambda r: -r["gamma"])
    return report


def summarize(report: List[dict]) -> dict:
    gammas = [r["gamma"] for r in report]
    rhos = [r["rho"] for r in report]
    max_gamma, min_rho = max(gammas), min(rhos)
    mean_g = sum(gammas) / len(gammas)
    cv = (sum((g - mean_g) ** 2 for g in gammas) / len(gammas)) ** 0.5 / max(mean_g, 1e-12)
    return {
        "n_groups": len(report),
        "max_gamma": max_gamma,
        "min_rho": min_rho,
        "spread_ratio": max_gamma / max(min_rho, 1e-12),
        "gamma_cv": cv,
        "equimarginal": max_gamma <= min_rho,
    }


def main():
    ap = argparse.ArgumentParser(description="Prediction 9.1 gate: equimarginal spread on a trained reference.")
    ap.add_argument("--arch", choices=["resnet18", "vgg16_bn"], required=True)
    ap.add_argument("--checkpoint", required=True, help="path to a trained state_dict (.pt)")
    ap.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-batches", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="optional path to dump the full JSON report")
    args = ap.parse_args()

    import torchvision
    from data.cifar import get_cifar10_loaders, get_cifar100_loaders

    num_classes = 10 if args.dataset == "cifar10" else 100
    model = {"resnet18": torchvision.models.resnet18,
              "vgg16_bn": torchvision.models.vgg16_bn}[args.arch](num_classes=num_classes)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)

    loader_fn = get_cifar10_loaders if args.dataset == "cifar10" else get_cifar100_loaders
    _, val_loader = loader_fn(root=args.data_root, batch_size=args.batch_size)

    report = run_gate(model, val_loader, torch.nn.functional.cross_entropy,
                       k=args.k, n_batches=args.n_batches, device=args.device)
    summary = summarize(report)

    print(f"\n{'tap':>4} {'size':>6} {'kappa_p':>10} {'gamma':>12} {'rho':>12}  producers")
    for r in report:
        print(f"{r['tap']:>4} {r['size']:>6} {r['kappa_params']:>10.0f} "
              f"{r['gamma']:>12.6e} {r['rho']:>12.6e}  {r['producers'][:2]}")

    print(f"\n=== Prediction 9.1 ===")
    print(f"n_groups     = {summary['n_groups']}")
    print(f"max(gamma)   = {summary['max_gamma']:.6e}")
    print(f"min(rho)     = {summary['min_rho']:.6e}")
    print(f"spread ratio = {summary['spread_ratio']:.2f}x")
    print(f"gamma CV     = {summary['gamma_cv']:.4f}")
    if summary["equimarginal"]:
        print("=> EQUIMARGINAL (max gamma <= min rho): thesis FALSIFIED for this architecture. STOP.")
    else:
        print("=> SPREAD confirmed: reference is not exchange-stable. Thesis supported, proceed.")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"report": report, "summary": summary}, f, indent=2)
        print(f"\nfull report written to {args.out}")


if __name__ == "__main__":
    main()
