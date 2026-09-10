# TSR-X Benchmark Report: MobileNetV2 on Tiny-ImageNet-200

## 3-Way Triangulation: Reference Baseline vs Dynamic Plasticity vs Matched Control

### 1. Executive Summary Table

| Metric | Arm 1 (Static Ref) | Arm 3 (C2 Matched Control) | Arm 2 (TSR-X Plasticity) |
| :--- | :---: | :---: | :---: |
| **Active Parameters** | 2,480,072 | 2,101,085 | 2,101,085 |
| **Parameter Reduction** | 0.0% (Baseline) | -15.28% | -15.28% |
| **Forward FLOPs** | 49.4 M | 28.8 M | 28.8 M |
| **FLOP Reduction** | 0.0% (Baseline) | -41.70% | -41.70% |
| **Top-1 Test Accuracy** | 15.57% | 16.63% | **14.42%** |
| **Top-5 Test Accuracy** | 37.50% | 40.24% | **37.03%** |
| **Dual Dominance Top-1 Δ**| — | — | **-1.15%** |
| **Plasticity Benefit Δ** | — | — | **-2.21%** |
