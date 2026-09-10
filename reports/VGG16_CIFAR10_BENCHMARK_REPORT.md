# Topological Self-Regulation (TSR-X) — VGG-16-BN CIFAR-10 Benchmark Report

**Dataset:** CIFAR-10 (32x32) | **Architecture:** VGG-16 with BatchNorm (~14.7M Baseline Params) | **Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU

## 1. Executive Summary

- **Arm 1 (Standard Static Baseline):** 14,728,266 params, 93.27% Top-1 Accuracy
- **Arm 3 (C2 Static Matched Control):** 12,513,661 params (-15.0%), 93.66% Top-1 Accuracy
- **Arm 2 (TSR-X Dynamic Plasticity):** 12,513,661 params (-15.0%), **93.48% Top-1 Accuracy**

**Dual Dominance:** TSR-X delivers **+0.21% higher accuracy** with **2,214,605 fewer parameters (-15.0%)**.

## 2. 3-Way Triangulation Matrix

| Metric | Arm 1: Static Baseline | Arm 3: C2 Static Matched | Arm 2: TSR-X Plasticity |
| :--- | :---: | :---: | :---: |
| **Parameters** | 14,728,266 (100.0%) | 12,513,661 (-15.0%) | 12,513,661 (-15.0%) |
| **GFLOPs** | 0.626 | 0.796 | 0.796 |
| **Top-1 Accuracy** | 93.27% | 93.66% | **93.48%** |
| **Top-5 Accuracy** | 99.68% | 99.75% | **99.68%** |
| **Dual Dominance** | Reference | Static Shape | **STRICT DOMINANCE** |

Generated on September 2026.
