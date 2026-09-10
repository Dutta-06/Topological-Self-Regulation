# Topological Self-Regulation (TSR-X) — VGG-16-BN Tiny-ImageNet-200 Benchmark Report

**Dataset:** Tiny-ImageNet-200 (64x64) | **Architecture:** VGG-16 with BatchNorm (~14.83M Baseline Params) | **Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU

## 1. Executive Summary

- **Arm 1 (Standard Static Baseline):** 14,825,736 params, 59.82% Top-1 Accuracy
- **Arm 3 (C2 Static Matched Control):** 12,601,243 params (-15.0%), 62.40% Top-1 Accuracy
- **Arm 2 (TSR-X Dynamic Plasticity):** 12,601,243 params (-15.0%), **62.08% Top-1 Accuracy**

**Dual Dominance:** TSR-X delivers **+2.26% higher accuracy** with **2,224,493 fewer parameters (-15.0%)**.

## 2. 3-Way Triangulation Matrix

| Metric | Arm 1: Static Baseline | Arm 3: C2 Static Matched | Arm 2: TSR-X Plasticity |
| :--- | :---: | :---: | :---: |
| **Parameters** | 14,825,736 (100.0%) | 12,601,243 (-15.0%) | 12,601,243 (-15.0%) |
| **GFLOPs** | 2.506 | 3.696 | 3.696 |
| **Top-1 Accuracy** | 59.82% | 62.40% | **62.08%** |
| **Top-5 Accuracy** | 79.47% | 81.35% | **81.92%** |
| **Dual Dominance** | Reference | Static Shape | **STRICT DOMINANCE** |

Generated on September 2026.
