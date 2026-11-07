"""OTOv2 (Chen et al., ICLR 2023) at TSR's matched parameter count.

OTOv2 trains a network once from scratch and drives whole channel groups
("zero-invariant groups", ZIGs) to exactly zero with its DHSPG optimizer; the
zeroed channels can then be removed without changing the function. It is the
closest published relative of removal-only TSR, so it is run here under the
same recipe as every other CIFAR arm (100 epochs, batch 128, lr 0.1 cosine,
momentum 0.9, weight decay 5e-4, AMP) on the same backbones.

Matching the count. DHSPG is configured by a target fraction of groups, not of
parameters, and its groups differ in size. At its partition step DHSPG ranks
all groups by its own redundancy score and penalizes the top fraction; here
that fraction is chosen at that moment, by binary search, as the smallest one
whose removal brings the exactly counted network (rebuilt by the coupling
engine, so consumer columns, normalization state and the cross terms between
adjacent groups are included; DHSPG's own group vectors omit the consumer
columns) to the target, 85% of dense as for TSR. Nothing else in DHSPG
changes. The deployed count is measured at the end by rebuilding the
network at the surviving widths, exactly as TSR's exported count is.

Uses the official library, only_train_once 2.0.16 (the last release that ships
DHSPG; 3.x replaced it with HESSO), with small shims for current Python and
PyTorch: `imp` and `distutils` were removed in Python 3.12, and the ONNX graph
optimizer it calls moved to torch.onnx.utils.

    python -m bench.train_oto --arch resnet18 --dataset cifar100 --seed 42 --out results/oto/resnet18_cifar100/oto_s42.pt
"""

import argparse
import json
import os
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def install_oto_shims(oto_root=None):
    """Make only_train_once 2.0.16 importable on current Python/PyTorch."""
    try:
        import imp  # noqa: F401  (removed in Python 3.12; OTO imports but never uses it)
    except ImportError:
        sys.modules["imp"] = types.ModuleType("imp")
    try:
        from distutils.version import LooseVersion  # noqa: F401  (removed in Python 3.12)
    except ImportError:
        from packaging.version import Version

        class LooseVersion:
            def __init__(self, v):
                self.v = Version(str(v).split("+")[0])

            def __ge__(self, o):
                return self.v >= o.v

            def __le__(self, o):
                return self.v <= o.v

            def __lt__(self, o):
                return self.v < o.v

            def __gt__(self, o):
                return self.v > o.v

        dist = types.ModuleType("distutils")
        ver = types.ModuleType("distutils.version")
        ver.LooseVersion = LooseVersion
        dist.version = ver
        sys.modules["distutils"] = dist
        sys.modules["distutils.version"] = ver
    import torch.onnx
    if not hasattr(torch.onnx, "_optimize_graph"):
        from torch.onnx import utils as _onnx_utils
        torch.onnx._optimize_graph = _onnx_utils._optimize_graph
    if oto_root is None:
        import importlib.util
        spec = importlib.util.find_spec("only_train_once")
        if spec is None or spec.submodule_search_locations is None:
            raise ImportError("only_train_once==2.0.16 is not installed")
        pkg = list(spec.submodule_search_locations)[0]
    else:
        pkg = os.path.join(oto_root, "only_train_once")
        sys.path.insert(0, oto_root)
    # the library imports its own subpackages (assets, operation, transform) as top-level modules
    sys.path.insert(0, pkg)


def coupling_map(model_builder, example_input):
    """The dense model's coupling groups (TSR's engine): {module name: tap}
    for every group producer, the dense width of each tap, the bundles, the
    dense count, and an exact counter `count(widths) -> deployed params` that
    rebuilds the network at those widths (so consumer columns, normalization
    state and the cross terms between adjacent edited groups are all exact)."""
    from bench.resize import resize_model_to_widths
    from tsrx.graph.bundle import build_all_bundles
    from tsrx.graph.groups import discover_groups
    from tsrx.graph.trace import trace_model

    m = model_builder()
    bundles = build_all_bundles(discover_groups(trace_model(m.eval(), (example_input,))), m)
    dense = sum(p.numel() for p in m.parameters())
    owner = {}
    for tap, bd in bundles.items():
        for slot in bd.producer_slots:
            owner[slot.module_name] = int(tap)
    bases = {int(t): int(bd.size) for t, bd in bundles.items()}

    def count(widths):
        return sum(p.numel() for p in resize_model_to_widths(model_builder(), widths, example_input).parameters())

    return owner, bases, bundles, dense, count


def budget_matched_partition(opt, group_tap, bases, count, target_params):
    """Replace DHSPG.partition_groups' fixed K with the smallest K whose
    removal, taking DHSPG's own top-ranked groups in its own score order,
    brings the exactly counted deployed network to `target_params` or below.
    `group_tap[cc_id]` is the coupling group an OTO group's channels belong
    to. Everything else is DHSPG's code path."""
    import only_train_once.optimizer.dhspg as dh
    original = dh.DHSPG.partition_groups

    def partition_groups(self):
        GROUP_TYPE = dh.GROUP_TYPE
        cos_all, norm_all, tap_all = [], [], []
        for group in self.param_groups:
            if group["group_type"] >= GROUP_TYPE["multi-head-linear"]:
                xs, grads = self.get_xs_grads(group)
                fx, fg = torch.cat(xs, dim=1), torch.cat(grads, dim=1)
                cos_all.append(F.cosine_similarity(-fx, -fg))
                size = fx.shape[1] if len(fx.shape) > 1 else 1
                norm_all.append(torch.norm(fx, dim=1) / size)
                tap_all.append(torch.full((group["num_groups"],), group_tap.get(group["cc_id"], -1),
                                          dtype=torch.long))
        cos_all, norm_all, tap_all = torch.cat(cos_all), torch.cat(norm_all), torch.cat(tap_all)
        # DHSPG's score, verbatim
        scores = 0.5 * F.normalize(cos_all, dim=0) + 0.5 * F.normalize(1.0 / (norm_all + 1e-6), dim=0)
        order = torch.argsort(scores, descending=True).cpu()
        ranked_taps = tap_all[order]
        total = scores.numel()

        def params_for(k):
            sel = ranked_taps[:k]
            sel = sel[sel >= 0]
            widths = dict(bases)
            if sel.numel():
                taps, counts = torch.unique(sel, return_counts=True)
                for t, c in zip(taps.tolist(), counts.tolist()):
                    widths[t] = max(1, widths[t] - c)
            return count(widths)

        lo, hi = 1, total - 1          # smallest k with params_for(k) <= target
        while lo < hi:
            mid = (lo + hi) // 2
            if params_for(mid) <= target_params:
                hi = mid
            else:
                lo = mid + 1
        k_star = lo
        planned = params_for(k_star)
        # DHSPG then penalizes int(total * (target + tolerance)) groups and stops
        # once target is reached; half a group of tolerance makes that int()
        # land on exactly k_star despite floating-point rounding.
        self.target_group_sparsity = k_star / total
        self.tolerance_group_sparsity = 0.5 / total
        self.budget_match = {"k_star": k_star, "total_groups": total,
                             "planned_params": int(planned), "target_params": int(target_params)}
        print(f"[oto] budget-matched partition: {k_star}/{total} groups "
              f"(group sparsity {self.target_group_sparsity:.4f}), planned deployed count "
              f"{planned:,} for target {target_params:,}", flush=True)
        return original(self)

    opt.partition_groups = types.MethodType(partition_groups, opt)


@torch.no_grad()
def surviving_widths(model, bundles):
    """Per coupling group, the channels not removed by OTO: a channel counts
    as removed only if every producer row and every affine entry of the group
    is exactly zero at that index (then its output is identically zero)."""
    mods = dict(model.named_modules())
    widths, removed = {}, {}
    for tap, bd in bundles.items():
        zero = None
        for slot in list(bd.producer_slots) + list(bd.affine_slots):
            mod = mods[slot.module_name]
            for pname in ("weight", "bias"):
                p = getattr(mod, pname, None)
                if p is None or p.dim() == 0:
                    continue
                z = (p.detach().reshape(p.shape[0], -1) == 0).all(dim=1)
                zero = z if zero is None else (zero & z)
        n = int(bd.size)
        k = int(zero.sum().item()) if zero is not None else 0
        widths[int(tap)] = n - k
        removed[int(tap)] = k
    return widths, removed


def evaluate(model, loader, device, use_amp):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.numel()
    return correct / total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "vgg16_bn"])
    ap.add_argument("--dataset", default="cifar100", choices=["cifar10", "cifar100"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--budget-ratio", type=float, default=0.85, help="deployed-count target as a fraction of dense")
    ap.add_argument("--start-pruning-epoch", type=float, default=20.0,
                    help="DHSPG partition and half-space projection start (OTOv2 CIFAR runs start at about 1/6 of training)")
    ap.add_argument("--epsilon", type=float, default=0.95, help="DHSPG half-space projection threshold")
    ap.add_argument("--lmbda", type=float, default=1e-2,
                    help="DHSPG group penalty; the official ResNet-18/CIFAR tutorial prescribes 1e-2 when group "
                         "sparsity is not produced at the library default 1e-3, which is the case at our budget")
    ap.add_argument("--max-steps", type=int, default=None, help="smoke tests only")
    ap.add_argument("--stop-after-epochs", type=int, default=None,
                    help="diagnostics only: keep the full-length schedule but stop after this many epochs")
    ap.add_argument("--amp", dest="amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--oto-root", default=None, help="dir containing only_train_once (default: installed package)")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    install_oto_shims(args.oto_root)
    from only_train_once import OTO

    from bench.models import build_model
    from data.cifar import get_cifar10_loaders, get_cifar100_loaders

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    num_classes = 10 if args.dataset == "cifar10" else 100
    loaders = get_cifar10_loaders if args.dataset == "cifar10" else get_cifar100_loaders
    train_loader, val_loader = loaders(root=args.data_root, batch_size=args.batch_size, num_workers=args.num_workers)
    steps_per_epoch = len(train_loader)
    total_steps = args.max_steps or args.epochs * steps_per_epoch

    def builder():
        return build_model(args.arch, num_classes, cifar_stem=True)

    ex = torch.randn(2, 3, 32, 32)
    module_tap, bases, bundles, dense, count = coupling_map(builder, ex)
    target = int(dense * args.budget_ratio)

    model = builder().to(args.device)
    oto = OTO(model=model, dummy_input=torch.zeros(1, 3, 32, 32, device=args.device))
    start = int(args.start_pruning_epoch * steps_per_epoch) if not args.max_steps else max(1, args.max_steps // 5)
    opt = oto.dhspg(variant="sgd", lr=args.lr, first_momentum=args.momentum, weight_decay=args.weight_decay,
                    target_group_sparsity=0.5, tolerance_group_sparsity=0.0, start_pruning_steps=start,
                    epsilon=args.epsilon, lmbda=args.lmbda)

    # map each OTO group to a coupling group through its parameter names
    group_tap, unmapped = {}, []
    for g in opt.param_groups:
        if "names" not in g:
            continue
        taps = {module_tap.get(n.rsplit(".", 1)[0]) for n in g["names"]} - {None}
        if len(taps) == 1:
            t = taps.pop()
            if g.get("num_groups") == bases[t]:
                group_tap[g["cc_id"]] = t
            else:
                unmapped.append((g["cc_id"], f"{g.get('num_groups')} groups vs width {bases[t]}"))
        elif g.get("num_groups", 0) > 1:
            unmapped.append((g["cc_id"], sorted(g["names"])[:3]))
    if unmapped:
        print(f"[oto] {len(unmapped)} OTO groups not mapped to one coupling group (left out of the count): "
              f"{unmapped[:4]}")
    budget_matched_partition(opt, group_tap, bases, count, target)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    use_amp = args.amp and args.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"[oto] {args.arch}/{args.dataset} seed {args.seed}: dense {dense:,}, target {target:,} "
          f"({args.budget_ratio:.0%}), partition at step {start}", flush=True)
    history, best, step = [], 0.0, 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        te = time.time()
        for x, y in train_loader:
            if step >= total_steps:
                break
            x, y = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = F.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
        acc = evaluate(model, val_loader, args.device, use_amp)
        best = max(best, acc)
        widths, removed = surviving_widths(model, bundles)
        from bench.resize import resize_model_to_widths
        params = sum(p.numel() for p in resize_model_to_widths(builder(), widths, ex).parameters())
        gs = float(getattr(opt, "curr_group_sparsity", 0.0))
        n_p, n_np, k_p, k_np = opt.compute_norm_group_partitions()
        mean_p = n_p / k_p if k_p else float("nan")
        mean_np = n_np / k_np if k_np else float("nan")
        history.append({"epoch": epoch, "val_acc": acc, "params": params, "group_sparsity": gs,
                        "mean_norm_penalized": mean_p, "mean_norm_kept": mean_np,
                        "seconds": time.time() - te})
        print(f"Epoch {epoch + 1:3d}/{args.epochs}  loss={loss.item():.4f}  val_acc={acc:.4f}  best={best:.4f}  "
              f"params={params:,}  group_sparsity={gs:.4f}  |x| penalized={mean_p:.3f} kept={mean_np:.3f}",
              flush=True)
        if step >= total_steps or (args.stop_after_epochs and epoch + 1 >= args.stop_after_epochs):
            break

    widths, removed = surviving_widths(model, bundles)
    from bench.resize import resize_model_to_widths
    deployed = resize_model_to_widths(builder(), widths, ex)
    params = sum(p.numel() for p in deployed.parameters())
    rec = {"arm": "oto", "arch": args.arch, "dataset": args.dataset, "seed": args.seed, "kind": "online",
           "best_val_acc": best, "final_val_acc": history[-1]["val_acc"] if history else None,
           "params": params, "baseline_params": dense, "target_params": target,
           "final_widths": widths, "removed_per_group": removed,
           "budget_match": getattr(opt, "budget_match", None),
           "train_seconds": time.time() - t0, "history": history,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "settings": {k: v for k, v in vars(args).items() if k not in ("out",)},
           "library": "only_train_once 2.0.16 (DHSPG)", "complete": step >= total_steps}
    torch.save({**rec, "model_state_dict": model.state_dict()}, out)
    out.with_suffix(".json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    print(f"[oto] done: last {rec['final_val_acc']:.4f}  best {best:.4f}  params {params:,} "
          f"(target {target:,}, {params / dense:.2%} of dense)", flush=True)


if __name__ == "__main__":
    main()
