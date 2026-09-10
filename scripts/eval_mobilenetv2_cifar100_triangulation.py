"""Evaluation and Triangulation Summary for MobileNetV2 on CIFAR-100 Benchmark."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from data.cifar import get_cifar100_loaders
from bench.models import build_model
from bench.train_static_matched import resize_model_to_widths


def count_flops(model, input_size=(1, 3, 32, 32), device="cuda"):
    flops = 0
    hooks = []
    
    def conv_hook(self, input, output):
        nonlocal flops
        batch_size, in_c, in_h, in_w = input[0].shape
        out_c, out_h, out_w = output.shape[1:]
        k_h, k_w = self.kernel_size
        flops += 2 * (k_h * k_w * (in_c // self.groups)) * out_c * out_h * out_w

    def linear_hook(self, input, output):
        nonlocal flops
        in_features = self.in_features
        out_features = self.out_features
        flops += 2 * in_features * out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    x = torch.zeros(input_size, device=device)
    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()
        
    return flops


@torch.no_grad()
def eval_accuracies(model, loader, device="cuda"):
    model.eval()
    top1, top5, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.startswith("cuda")):
            out = model(x)
        _, pred = out.topk(min(5, out.size(1)), 1, True, True)
        pred = pred.t()
        correct = pred.eq(y.view(1, -1).expand_as(pred))
        top1 += correct[:1].reshape(-1).float().sum().item()
        top5 += correct[:min(5, out.size(1))].reshape(-1).float().sum().item()
        total += y.size(0)
    return (top1 / max(total, 1)) * 100.0, (top5 / max(total, 1)) * 100.0


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, val_loader = get_cifar100_loaders(batch_size=128)

    # 1. Arm 1: Standard Static Reference
    print("Evaluating Arm 1: Standard Static Reference Baseline (MobileNetV2 CIFAR-100)...")
    ck1 = torch.load("results/reference/mobilenetv2_cifar100.pt", map_location="cpu", weights_only=False)
    m1 = build_model("mobilenet_v2", 100, cifar_stem=True).to(device)
    m1.load_state_dict(ck1["model_state_dict"])
    p1 = sum(p.numel() for p in m1.parameters() if p.requires_grad)
    flops1 = count_flops(m1, device=device)
    top1_1, top5_1 = eval_accuracies(m1, val_loader, device=device)
    del m1

    # 2. Arm 3: C2 Static Matched Control
    print("Evaluating Arm 3: C2 Static Matched Control (MobileNetV2 CIFAR-100)...")
    ck3 = torch.load("results/static_matched/mobilenetv2_cifar100_two_regime.pt", map_location="cpu", weights_only=False)
    ck2_raw = torch.load("results/tsrx/mobilenetv2_cifar100_two_regime.pt", map_location="cpu", weights_only=False)
    widths = ck2_raw["discovered_widths"]
    m3 = build_model("mobilenet_v2", 100, cifar_stem=True)
    m3 = resize_model_to_widths(m3, widths, torch.zeros(2, 3, 32, 32)).to(device)
    m3.load_state_dict(ck3["model_state_dict"])
    p3 = sum(p.numel() for p in m3.parameters() if p.requires_grad)
    flops3 = count_flops(m3, device=device)
    top1_3, top5_3 = eval_accuracies(m3, val_loader, device=device)
    del m3

    # 3. Arm 2: TSR-X Dynamic Plasticity
    print("Evaluating Arm 2: TSR-X Dynamic Plasticity (MobileNetV2 CIFAR-100)...")
    ck2 = ck2_raw
    m2 = build_model("mobilenet_v2", 100, cifar_stem=True)
    m2 = resize_model_to_widths(m2, widths, torch.zeros(2, 3, 32, 32)).to(device)
    sd2 = {}
    for k, v in ck2["model_state_dict"].items():
        if ".cand_proj." in k or ".port" in k:
            continue
        if k in m2.state_dict():
            tgt_shape = m2.state_dict()[k].shape
            if v.shape == tgt_shape:
                sd2[k] = v
            else:
                slices = tuple(slice(0, s) for s in tgt_shape)
                sd2[k] = v[slices]
    m2.load_state_dict(sd2)
    p2 = sum(p.numel() for p in m2.parameters() if p.requires_grad)
    flops2 = count_flops(m2, device=device)
    top1_2, top5_2 = eval_accuracies(m2, val_loader, device=device)
    del m2

    param_reduction = (1.0 - (p2 / p1)) * 100.0
    flop_reduction = (1.0 - (flops2 / flops1)) * 100.0
    delta_acc = top1_2 - top1_1
    delta_plasticity = top1_2 - top1_3

    print("\n" + "="*85)
    print("      MOBILENETV2 CIFAR-100 3-WAY TRIANGULATION BENCHMARK SUMMARY")
    print("="*85)
    print(f"{'Metric':<35} | {'Arm 1 (Static Ref)':<18} | {'Arm 3 (Static Matched)':<22} | {'Arm 2 (TSR-X Plasticity)':<24}")
    print("-" * 110)
    print(f"{'Active Parameters':<35} | {p1:<18,d} | {p3:<22,d} | {p2:<24,d}")
    print(f"{'Param Reduction vs Baseline':<35} | {'0.0% (Baseline)':<18} | {f'-{param_reduction:.2f}%':<22} | {f'-{param_reduction:.2f}%':<24}")
    print(f"{'Forward GFLOPs':<35} | {flops1/1e9:<18.4f} | {flops3/1e9:<22.4f} | {flops2/1e9:<24.4f}")
    print(f"{'FLOP Reduction vs Baseline':<35} | {'0.0% (Baseline)':<18} | {f'-{flop_reduction:.2f}%':<22} | {f'-{flop_reduction:.2f}%':<24}")
    print(f"{'Top-1 Test Accuracy':<35} | {top1_1:<17.2f}% | {top1_3:<21.2f}% | {top1_2:<23.2f}%")
    print(f"{'Top-5 Test Accuracy':<35} | {top5_1:<17.2f}% | {top5_3:<21.2f}% | {top5_2:<23.2f}%")
    print("="*85)
    print(f"Dual Dominance Top-1 Gain:  {delta_acc:+.2f}%  ({top1_2:.2f}% vs {top1_1:.2f}%) with {param_reduction:.2f}% fewer params")
    print(f"Plasticity Advantage Delta: {delta_plasticity:+.2f}%  (Arm 2 vs Arm 3)")
    print("="*85 + "\n")

    summary_data = {
        "dataset": "cifar100",
        "arch": "mobilenet_v2",
        "arm1": {"params": p1, "flops": flops1, "top1": top1_1, "top5": top5_1},
        "arm2": {"params": p2, "flops": flops2, "top1": top1_2, "top5": top5_2, "discovered_widths": widths},
        "arm3": {"params": p3, "flops": flops3, "top1": top1_3, "top5": top5_3},
        "param_reduction_pct": param_reduction,
        "flop_reduction_pct": flop_reduction,
        "delta_acc": delta_acc,
        "delta_plasticity": delta_plasticity,
    }
    out_path = Path("results/mobilenetv2_cifar100_triangulation_summary.pt")
    torch.save(summary_data, out_path)
    print(f"Summary tensor successfully saved to: {out_path}")


if __name__ == "__main__":
    main()
