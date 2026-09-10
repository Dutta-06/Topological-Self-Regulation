# TSR-X Benchmark Report: EfficientNet-B0 on CIFAR-10

## 3-Way Triangulation: Reference Baseline vs Dynamic Plasticity vs Matched Control

### 1. Executive Summary Table

| Metric | Arm 1 (Static Ref) | Arm 3 (C2 Matched Control) | Arm 2 (TSR-X Plasticity) |
| :--- | :---: | :---: | :---: |
| **Active Parameters** | 4,020,358 | 3,417,087 | 3,417,087 |
| **Parameter Reduction** | 0.0% (Baseline) | -15.01% | -15.01% |
| **Forward FLOPs** | 64.0 M | 89.7 M | 89.7 M |
| **FLOP Reduction** | 0.0% (Baseline) | --40.17% | --40.17% |
| **Top-1 Test Accuracy** | 90.07% | 90.62% | **90.19%** |
| **Top-5 Test Accuracy** | 99.64% | 99.62% | **99.68%** |
| **Dual Dominance Top-1 Δ**| — | — | **+0.12%** |
| **Plasticity Benefit Δ** | — | — | **-0.43%** |
