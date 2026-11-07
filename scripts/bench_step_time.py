"""Wall-clock of every arm's training pipeline, from measured GPU step times.

Each arm's training step is timed on the same GPU with the same synthetic
CIFAR batch (128 x 3 x 32 x 32, AMP, SGD-Nesterov), so the
comparison isolates the model and the per-step work. Data loading (CPU
augmentation for CIFAR) and evaluation are identical across arms and excluded.

Arms and what a step does:
  ref       dense model
  online    model + dormant candidate bank, per-step sensing (u_c window,
            first-order saliency, E[a^2] hooks), port re-zeroing, and the
            structural update every K = 100 steps (real controller calls)
  online_legacy   the same, with the sensing path as released (module-tree
            walk per group per call, host copy per record)
  nobank    removal-only TSR with no candidate bank (--growth off --k 0):
            removal saliency, E[a^2] hooks and structural updates only
  oto_penalty  OTOv2's DHSPG step (official only_train_once 2.0.16) while it
            penalizes the selected groups toward zero (2-4 epochs in our runs)
  oto_plain    its step outside that phase (before the partition, and after
            the target group sparsity is reached), on the full-size network
            OTOv2 trains throughout (--with-oto; needs the library installed)
  c2        dense model at the widths TSR exported
  pruned    dense model at the widths l1 pruning exported at the same count

Pipeline cost (100 epochs per training run, 40 for fine-tuning):
  Ref = 100 ref; Online = 100 online; prune+FT = 100 ref + 40 pruned;
  prune+scratch = 100 ref + 100 pruned; C2 = 100 online + 100 c2;
  bank-free = 100 nobank (an upper bound: the real run trains a smaller
  network after the anneal); OTOv2 = 96 oto_plain + 4 oto_penalty.

Arms are timed in interleaved rounds and the median kept, to damp clock and
thermal drift.

    python scripts/bench_step_time.py --arch resnet18      # widths from results/bench/arm_widths.json
"""

import argparse
import collections
import json
import statistics as st
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bench.models import build_model  # noqa: E402
from bench.resize import resize_model_to_widths  # noqa: E402
from tsrx.alloc.exchange import apply_exchange, evaluate_structural_update  # noqa: E402
from tsrx.alloc.schedule import budget_at  # noqa: E402
from tsrx.graph.bundle import build_all_bundles  # noqa: E402
from tsrx.graph.groups import discover_groups  # noqa: E402
from tsrx.graph.trace import trace_model  # noqa: E402
from tsrx.sense.candidates import CandidateBank  # noqa: E402
from tsrx.sense.saliency import ActivationStats, first_order_saliency  # noqa: E402
from tsrx.sense.topo import WindowedSignal, compute_uc_norms  # noqa: E402

STEPS_PER_EPOCH = 391  # CIFAR train set / 128


class LegacyWindow(WindowedSignal):
    def record(self, tap, uc_norms):
        self._buf.setdefault(tap, collections.deque(maxlen=self.window)).append(uc_norms.detach().cpu())


def widths_from(src, arch=None, key=None):
    """Widths from a checkpoint (final_widths / discovered_widths), or from
    results/bench/arm_widths.json[arch][key] when `src` is a .json file."""
    if str(src).endswith(".json"):
        return {int(t): int(v) for t, v in json.loads(Path(src).read_text(encoding="utf-8"))[arch][key].items()}
    ck = torch.load(src, map_location="cpu", weights_only=False)
    return {int(t): int(v) for t, v in (ck.get("final_widths") or ck["discovered_widths"]).items()}


class Arm:
    def __init__(self, name, arch, dev, x, y, widths=None, online=False, legacy=False, K=100, k=4):
        self.name, self.dev, self.x, self.y, self.online, self.legacy, self.K = name, dev, x, y, online, legacy, K
        self.k = k
        self.growth = "grad" if k > 0 else "off"
        m = build_model(arch, 100, cifar_stem=True).to(dev)
        if widths:
            m = resize_model_to_widths(m, widths, x[:2]).to(dev)
        self.model, self.bank = m, None
        self.base = sum(p.numel() for p in m.parameters())
        if online:
            res = discover_groups(trace_model(m, (x[:2],)))
            self.bank = CandidateBank(m, build_all_bundles(res, m), k=k).to(dev)
            self.win = (LegacyWindow if legacy else WindowedSignal)(window=K)
            self.act = ActivationStats(m, self.bank)
            self.sal = {t: None for t in self.bank.handles}
            self.mods = dict(m.named_modules())
            self.n_seen = 0
        self.opt = torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4, nesterov=True)
        self.scaler = torch.amp.GradScaler("cuda")
        self.step_i = 0

    def step(self):
        m, opt = self.model, self.opt
        xb = self.x
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            loss = F.cross_entropy(m(xb), self.y)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(opt)
        if self.online:
            with torch.no_grad():
                for tap, h in self.bank.handles.items():
                    mods = None if self.legacy else self.mods
                    if self.k > 0:
                        self.win.record(tap, compute_uc_norms(self.bank, tap, modules=mods))
                    s = first_order_saliency(m, h.bundle, h.base_size, modules=mods)
                    self.sal[tap] = s if self.sal[tap] is None else self.sal[tap] + s
            self.n_seen += 1
        self.scaler.step(opt)
        self.scaler.update()
        if self.online and self.k > 0:
            self.bank.zero_ports()
        loss.item()
        self.step_i += 1
        if self.online and self.step_i % self.K == 0:
            self.structural_update()

    def structural_update(self):
        b = self.bank
        dep = b.deployed_params()
        # budget anneal as in training: 100 epochs, 85% target, ends at half
        B_t = budget_at(self.step_i, 100 * STEPS_PER_EPOCH, self.base, int(0.85 * self.base), end_frac=0.5)
        decs = evaluate_structural_update(bank=b, windowed_signal=self.win, saliency_sum=self.sal,
                                          n_seen=self.n_seen, deployed_params=dep, budget_at_t=B_t,
                                          act_stats=self.act, H_max=1.0, growth_mode=self.growth)
        applied = [d for d in decs if d.action != "none"]
        for d in applied:
            apply_exchange(d, b, optimizer=self.opt, eps=1e-3, grow_first=False)
        if applied:
            b.zero_ports()
            self.act.remove()
            self.act = ActivationStats(self.model, b)
            self.mods = dict(self.model.named_modules())
        self.sal = {t: None for t in b.handles}
        self.n_seen = 0
        self.act.reset()


class OtoArm:
    """OTOv2's training step: DHSPG on the full-size network, partitioned at
    step `partition` so the timed steps include its penalty and projection."""

    def __init__(self, name, arch, dev, x, y, partition=10):
        """partition=10 times the penalty phase; partition=None never
        partitions, which times DHSPG's plain update."""
        from bench.train_oto import install_oto_shims
        install_oto_shims()
        from only_train_once import OTO
        self.name, self.x, self.y = name, x, y
        self.model = build_model(arch, 100, cifar_stem=True).to(dev)
        oto = OTO(model=self.model, dummy_input=torch.zeros(1, 3, 32, 32, device=dev))
        self.opt = oto.dhspg(variant="sgd", lr=0.05, first_momentum=0.9, weight_decay=5e-4,
                             target_group_sparsity=0.08, tolerance_group_sparsity=0.0,
                             start_pruning_steps=10 ** 9 if partition is None else partition,
                             epsilon=0.95, lmbda=1e-2)
        self.scaler = torch.amp.GradScaler("cuda")

    def step(self):
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            loss = F.cross_entropy(self.model(self.x), self.y)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()
        loss.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--c2-widths", default=str(ROOT / "results" / "bench" / "arm_widths.json"),
                    help="TSR checkpoint, or the arm_widths.json table (default)")
    ap.add_argument("--pruned-widths", default=str(ROOT / "results" / "bench" / "arm_widths.json"),
                    help="l1-pruned checkpoint, or the arm_widths.json table (default)")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=200, help="timed steps per arm per round (2 structural updates)")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--with-oto", action="store_true", help="also time OTOv2's DHSPG step")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)
    x = torch.randn(128, 3, 32, 32, device=dev)
    y = torch.randint(0, 100, (128,), device=dev)
    arms = [Arm("ref", args.arch, dev, x, y),
            Arm("online", args.arch, dev, x, y, online=True),
            Arm("online_legacy", args.arch, dev, x, y, online=True, legacy=True),
            Arm("c2", args.arch, dev, x, y, widths=widths_from(args.c2_widths, args.arch, "c2")),
            Arm("pruned", args.arch, dev, x, y, widths=widths_from(args.pruned_widths, args.arch, "pruned")),
            Arm("nobank", args.arch, dev, x, y, online=True, k=0)]
    if args.with_oto:
        arms += [OtoArm("oto_penalty", args.arch, dev, x, y, partition=10),
                 OtoArm("oto_plain", args.arch, dev, x, y, partition=None)]
    for a in arms:
        for _ in range(args.warmup):
            a.step()
    times = collections.defaultdict(list)
    for r in range(args.rounds):
        for a in arms:
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(args.steps):
                a.step()
            torch.cuda.synchronize()
            times[a.name].append((time.perf_counter() - t) / args.steps * 1000)
    ms = {k: st.median(v) for k, v in times.items()}
    pipe = {
        "Ref": 100 * ms["ref"],
        "Online": 100 * ms["online"],
        "Online (released sensing)": 100 * ms["online_legacy"],
        "Prune + 40-epoch fine-tune": 100 * ms["ref"] + 40 * ms["pruned"],
        "Prune + retrain from scratch": 100 * ms["ref"] + 100 * ms["pruned"],
        "C2 (online + retrain)": 100 * ms["online"] + 100 * ms["c2"],
        "Bank-free removal-only TSR (upper bound)": 100 * ms["nobank"],
    }
    if "oto_plain" in ms:
        pipe["OTOv2 (96 plain + 4 penalty epochs)"] = 96 * ms["oto_plain"] + 4 * ms["oto_penalty"]
    gpu = torch.cuda.get_device_name(0)
    print(f"# {args.arch}, {gpu}: median ms/step over {args.rounds} rounds x {args.steps} steps")
    for k in [a for a in ("ref", "online", "online_legacy", "c2", "pruned", "nobank", "oto_penalty", "oto_plain")
              if a in ms]:
        print(f"  {k:<15} {ms[k]:7.2f} ms  ({ms[k] / ms['ref']:.2f}x ref)   rounds: "
              + " ".join(f"{v:.1f}" for v in times[k]))
    print("pipeline wall-clock (x Ref):")
    for k, v in pipe.items():
        print(f"  {k:<42} {v / pipe['Ref']:.2f}x")
    if args.out:
        Path(args.out).write_text(json.dumps({"arch": args.arch, "gpu": gpu, "ms_per_step": ms,
                                              "rounds": times, "pipeline_x_ref": {k: v / pipe["Ref"] for k, v in pipe.items()}},
                                             indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
