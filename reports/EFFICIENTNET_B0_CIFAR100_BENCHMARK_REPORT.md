# TSR-X Benchmark Report: EfficientNet-B0 on CIFAR-100

## 3-Way Triangulation: Reference Baseline vs Dynamic Plasticity vs Matched Control

### 1. Executive Summary Table

| Metric | Arm 1 (Static Ref) | Arm 3 (C2 Matched Control) | Arm 2 (TSR-X Plasticity) |
| :--- | :---: | :---: | :---: |
| **Active Parameters** | 4,135,648 | 3,515,277 | 3,515,277 |
| **Parameter Reduction** | 0.0% (Baseline) | -15.00% | -15.00% |
| **Forward FLOPs** | 64.2 M | 61.9 M | 61.9 M |
| **FLOP Reduction** | 0.0% (Baseline) | -3.56% | -3.56% |
| **Top-1 Test Accuracy** | 64.62% | 67.99% | **67.62%** |
| **Top-5 Test Accuracy** | 88.57% | 90.84% | **90.61%** |
| **Dual Dominance Top-1 Δ**| — | — | **+3.00%** |
| **Plasticity Benefit Δ** | — | — | **-0.37%** |
