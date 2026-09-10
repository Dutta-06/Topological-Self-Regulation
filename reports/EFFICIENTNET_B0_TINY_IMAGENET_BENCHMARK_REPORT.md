# TSR-X Benchmark Report: EfficientNet-B0 on Tiny-ImageNet-200

## 3-Way Triangulation: Reference Baseline vs Dynamic Plasticity vs Matched Control

### 1. Executive Summary Table

| Metric | Arm 1 (Static Ref) | Arm 3 (C2 Matched Control) | Arm 2 (TSR-X Plasticity) |
| :--- | :---: | :---: | :---: |
| **Active Parameters** | 4,263,748 | 3,604,306 | 3,604,306 |
| **Parameter Reduction** | 0.0% (Baseline) | -15.47% | -15.47% |
| **Forward FLOPs** | 64.4 M | 66.7 M | 66.7 M |
| **FLOP Reduction** | 0.0% (Baseline) | --3.50% | --3.50% |
| **Top-1 Test Accuracy** | 12.03% | 14.78% | **14.55%** |
| **Top-5 Test Accuracy** | 31.54% | 36.15% | **36.12%** |
| **Dual Dominance Top-1 Δ**| — | — | **+2.52%** |
| **Plasticity Benefit Δ** | — | — | **-0.23%** |
