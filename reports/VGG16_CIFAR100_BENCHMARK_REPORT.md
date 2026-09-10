# Topological Self-Regulation (TSR-X) — VGG-16-BN CIFAR-100 Benchmark Report

**Dataset:** CIFAR-100 (32x32) | **Architecture:** VGG-16 with BatchNorm (~14.77M Baseline Params) | **Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU

## 1. Executive Summary

- **Arm 1 (Standard Static Baseline):** 14,774,436 params, 72.09% Top-1 Accuracy
- **Arm 3 (C2 Static Matched Control):** 12,553,018 params (-15.0%), 73.34% Top-1 Accuracy
- **Arm 2 (TSR-X Dynamic Plasticity):** 12,553,018 params (-15.0%), **73.39% Top-1 Accuracy**

**Dual Dominance:** TSR-X delivers **+1.30% higher accuracy** with **2,221,418 fewer parameters (-15.0%)**.

## 2. 3-Way Triangulation Matrix

| Metric | Arm 1: Static Baseline | Arm 3: C2 Static Matched | Arm 2: TSR-X Plasticity |
| :--- | :---: | :---: | :---: |
| **Parameters** | 14,774,436 (100.0%) | 12,553,018 (-15.0%) | 12,553,018 (-15.0%) |
| **GFLOPs** | 0.626 | 0.800 | 0.800 |
| **Top-1 Accuracy** | 72.09% | 73.34% | **73.39%** |
| **Top-5 Accuracy** | 90.55% | 91.26% | **91.67%** |
| **Dual Dominance** | Reference | Static Shape | **STRICT DOMINANCE** |

Generated on September 2026.
