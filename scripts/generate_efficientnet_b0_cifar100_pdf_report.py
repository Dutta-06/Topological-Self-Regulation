"""Generate High-Resolution Publication-Quality PDF and Markdown Reports for EfficientNet-B0 on CIFAR-100."""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#64748B"))
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, letter[1] - 40, letter[0] - 54, letter[1] - 40)
        self.drawString(54, letter[1] - 34, "Topological Self-Regulation (TSR-X) — Empirical Benchmark Report")
        self.line(54, 45, letter[0] - 54, 45)
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 32, page_str)
        self.drawString(54, 32, "Empirical Research • EfficientNet-B0 CIFAR-100 Benchmark")
        self.restoreState()


def generate_plots(output_dir: Path, data: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "efficientnet_b0_cifar100_pareto.png"

    arm1 = data["arm1"]
    arm2 = data["arm2"]
    arm3 = data["arm3"]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)

    ax.scatter([arm1["params"] / 1e6], [arm1["top1"]], color="#64748B", s=140, zorder=5, label="Arm 1: Static Baseline (100%)", edgecolors="black", linewidth=1.2)
    ax.scatter([arm3["params"] / 1e6], [arm3["top1"]], color="#F59E0B", s=140, marker="s", zorder=5, label="Arm 3: C2 Static Matched", edgecolors="black", linewidth=1.2)
    ax.scatter([arm2["params"] / 1e6], [arm2["top1"]], color="#10B981", s=180, marker="*", zorder=6, label="Arm 2: TSR-X Plasticity (-15%)", edgecolors="black", linewidth=1.2)

    ax.annotate(f"Arm 1 (Ref)\n{arm1['top1']:.2f}%\n{arm1['params']/1e6:.2f}M", 
                xy=(arm1["params"]/1e6, arm1["top1"]), xytext=(10, -15), textcoords="offset points",
                fontsize=8, fontweight="bold", color="#334155")
    ax.annotate(f"Arm 3 (C2 Control)\n{arm3['top1']:.2f}%\n{arm3['params']/1e6:.2f}M", 
                xy=(arm3["params"]/1e6, arm3["top1"]), xytext=(-100, -25), textcoords="offset points",
                fontsize=8, fontweight="bold", color="#B45309")
    ax.annotate(f"Arm 2 (TSR-X Dual Dominance)\n{arm2['top1']:.2f}%\n{arm2['params']/1e6:.2f}M (-15.0%)", 
                xy=(arm2["params"]/1e6, arm2["top1"]), xytext=(-70, 15), textcoords="offset points",
                fontsize=9, fontweight="bold", color="#047857",
                arrowprops=dict(arrowstyle="->", color="#10B981", lw=1.5))

    ax.set_title("EfficientNet-B0 CIFAR-100: Accuracy vs Active Parameters", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Active Parameters (Millions)", fontsize=10, labelpad=8)
    ax.set_ylabel("Top-1 Test Accuracy (%)", fontsize=10, labelpad=8)
    ax.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", loc="lower right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    return plot_path


def build_pdf(pdf_path: Path, data: dict, plot_path: Path):
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle", parent=styles["Heading1"],
        fontName="Helvetica-Bold", fontSize=20, leading=24,
        textColor=colors.HexColor("#0F172A"), spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "DocSub", parent=styles["Normal"],
        fontName="Helvetica", fontSize=10, leading=14,
        textColor=colors.HexColor("#475569"), spaceAfter=14,
    )
    h2_style = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        fontName="Helvetica-Bold", fontSize=13, leading=17,
        textColor=colors.HexColor("#1E293B"), spaceBefore=12, spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontName="Helvetica", fontSize=9, leading=13,
        textColor=colors.HexColor("#334155"), spaceAfter=6,
    )

    story = []
    story.append(Paragraph("TSR-X Benchmark: EfficientNet-B0 on CIFAR-100", title_style))
    story.append(Paragraph("3-Way Triangulation: Static Reference Baseline vs. Online Dynamic Plasticity vs. Discovered Control", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E2E8F0"), spaceAfter=12))

    arm1, arm2, arm3 = data["arm1"], data["arm2"], data["arm3"]
    p_red = data["param_reduction_pct"]
    f_red = data["flop_reduction_pct"]
    d_acc = data["delta_acc"]
    d_plas = data["delta_plasticity"]

    story.append(Paragraph("1. Triangulation Summary", h2_style))
    table_data = [
        ["Metric", "Arm 1 (Static Ref)", "Arm 3 (C2 Matched)", "Arm 2 (TSR-X Plasticity)"],
        ["Parameters", f"{arm1['params']:,}", f"{arm3['params']:,}", f"{arm2['params']:,}"],
        ["Param Reduction", "0.0% (Baseline)", f"-{p_red:.2f}%", f"-{p_red:.2f}%"],
        ["FLOPs (MFLOPs)", f"{arm1['flops']/1e6:.1f}", f"{arm3['flops']/1e6:.1f}", f"{arm2['flops']/1e6:.1f}"],
        ["FLOP Reduction", "0.0% (Baseline)", f"-{f_red:.2f}%", f"-{f_red:.2f}%"],
        ["Top-1 Accuracy", f"{arm1['top1']:.2f}%", f"{arm3['top1']:.2f}%", f"{arm2['top1']:.2f}%"],
        ["Top-5 Accuracy", f"{arm1['top5']:.2f}%", f"{arm3['top5']:.2f}%", f"{arm2['top5']:.2f}%"],
        ["Dual Dominance Gain", "-", "-", f"<b>{d_acc:+.2f}%</b>"],
        ["Plasticity Delta", "-", "-", f"<b>{d_plas:+.2f}%</b>"],
    ]
    t = Table(table_data, colWidths=[1.8*inch, 1.6*inch, 1.6*inch, 1.8*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F8FAFC")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('BACKGROUND', (3, 1), (3, -1), colors.HexColor("#F0FDF4")),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    story.append(Paragraph("2. Pareto Frontier & Efficiency Analysis", h2_style))
    if plot_path.exists():
        story.append(Image(str(plot_path), width=6*inch, height=3.6*inch))
    story.append(Spacer(1, 10))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF report compiled to: {pdf_path}")


def build_markdown(md_path: Path, data: dict):
    arm1, arm2, arm3 = data["arm1"], data["arm2"], data["arm3"]
    p_red = data["param_reduction_pct"]
    f_red = data["flop_reduction_pct"]
    d_acc = data["delta_acc"]
    d_plas = data["delta_plasticity"]

    content = f"""# TSR-X Benchmark Report: EfficientNet-B0 on CIFAR-100

## 3-Way Triangulation: Reference Baseline vs Dynamic Plasticity vs Matched Control

### 1. Executive Summary Table

| Metric | Arm 1 (Static Ref) | Arm 3 (C2 Matched Control) | Arm 2 (TSR-X Plasticity) |
| :--- | :---: | :---: | :---: |
| **Active Parameters** | {arm1['params']:,} | {arm3['params']:,} | {arm2['params']:,} |
| **Parameter Reduction** | 0.0% (Baseline) | -{p_red:.2f}% | -{p_red:.2f}% |
| **Forward FLOPs** | {arm1['flops']/1e6:.1f} M | {arm3['flops']/1e6:.1f} M | {arm2['flops']/1e6:.1f} M |
| **FLOP Reduction** | 0.0% (Baseline) | -{f_red:.2f}% | -{f_red:.2f}% |
| **Top-1 Test Accuracy** | {arm1['top1']:.2f}% | {arm3['top1']:.2f}% | **{arm2['top1']:.2f}%** |
| **Top-5 Test Accuracy** | {arm1['top5']:.2f}% | {arm3['top5']:.2f}% | **{arm2['top5']:.2f}%** |
| **Dual Dominance Top-1 Δ**| — | — | **{d_acc:+.2f}%** |
| **Plasticity Benefit Δ** | — | — | **{d_plas:+.2f}%** |
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Markdown report saved to: {md_path}")


def main():
    summary_path = Path("results/efficientnet_b0_cifar100_triangulation_summary.pt")
    if not summary_path.exists():
        print(f"Error: {summary_path} not found.")
        sys.exit(1)

    data = torch.load(summary_path, map_location="cpu", weights_only=False)
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_path = generate_plots(out_dir, data)
    pdf_path = out_dir / "EFFICIENTNET_B0_CIFAR100_BENCHMARK_REPORT.pdf"
    md_path = out_dir / "EFFICIENTNET_B0_CIFAR100_BENCHMARK_REPORT.md"

    build_pdf(pdf_path, data, plot_path)
    build_markdown(md_path, data)


if __name__ == "__main__":
    main()
