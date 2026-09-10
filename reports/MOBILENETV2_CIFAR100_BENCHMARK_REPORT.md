# TSR-X Benchmark Report: MobileNetV2 on CIFAR-100

## 3-Way Triangulation: Reference Baseline vs Dynamic Plasticity vs Matched Control

### 1. Executive Summary Table

| Metric | Arm 1 (Static Ref) | Arm 3 (C2 Matched Control) | Arm 2 (TSR-X Plasticity) |
| :--- | :---: | :---: | :---: |
| **Active Parameters** | 2,351,972 | 1,992,323 | 1,992,323 |
| **Parameter Reduction** | 0.0% (Baseline) | -15.29% | -15.29% |
| **Forward FLOPs** | 49.2 M | 43.6 M | 43.6 M |
| **FLOP Reduction** | 0.0% (Baseline) | -11.34% | -11.34% |
| **Top-1 Test Accuracy** | 63.27% | 66.83% | **68.23%** |
| **Top-5 Test Accuracy** | 88.25% | 90.74% | **91.29%** |
| **Dual Dominance Top-1 Δ**| — | — | **+4.96%** |
| **Plasticity Benefit Δ** | — | — | **+1.40%** |
