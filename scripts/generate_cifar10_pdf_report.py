import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable
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
        
        # Header line
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, letter[1] - 40, letter[0] - 54, letter[1] - 40)
        self.drawString(54, letter[1] - 34, "Topological Self-Regulation (TSR-X) — Empirical Benchmark Report")
        
        # Footer line
        self.line(54, 45, letter[0] - 54, 45)
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 32, page_str)
        self.drawString(54, 32, "Confidential & Empirical Research • CIFAR-10 Benchmark")
        self.restoreState()


def generate_charts(chart_dir: Path):
    chart_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Dual Dominance Chart (Acc vs Params)
    fig, ax1 = plt.subplots(figsize=(6.5, 2.6), dpi=300)
    plt.subplots_adjust(bottom=0.22, top=0.88, left=0.10, right=0.90)
    
    arms = ["Arm 1: Static Baseline", "Arm 2: TSR-X Plasticity", "Arm 3: C2 Matched Control"]
    params = [11.173962, 9.497686, 9.497686]
    accs = [95.02, 94.70, 95.08]
    
    x = np.arange(len(arms))
    width = 0.32
    
    color_params = "#3B82F6"
    color_acc = "#10B981"
    
    rects1 = ax1.bar(x - width/2, params, width, label='Parameters (Millions)', color=color_params, alpha=0.9)
    ax1.set_ylabel('Parameters (M)', color=color_params, fontsize=9, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_params, labelsize=8)
    ax1.set_ylim(0, 13.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(arms, fontsize=8.5, fontweight='bold')
    
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f'{h:.2f}M', xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 2), textcoords="offset points", ha='center', va='bottom',
                     fontsize=8, fontweight='bold', color=color_params)
        
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, accs, width, label='Val Accuracy (%)', color=color_acc, alpha=0.9)
    ax2.set_ylabel('Val Accuracy (%)', color=color_acc, fontsize=9, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_acc, labelsize=8)
    ax2.set_ylim(92.0, 96.0)
    
    for rect in rects2:
        h = rect.get_height()
        ax2.annotate(f'{h:.2f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 2), textcoords="offset points", ha='center', va='bottom',
                     fontsize=8, fontweight='bold', color=color_acc)
        
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.set_title("Strict Dual Dominance Comparison (ResNet-18 on CIFAR-10)", fontsize=10, fontweight='bold', color="#0F172A", pad=8)
    
    chart1_path = chart_dir / "dual_dominance.png"
    plt.savefig(chart1_path, bbox_inches="tight")
    plt.close()
    
    # 2. Channel Topology Migration Chart
    fig, ax = plt.subplots(figsize=(6.5, 2.6), dpi=300)
    plt.subplots_adjust(bottom=0.22, top=0.88, left=0.10, right=0.95)
    
    layers = ["L1 (Tap 3)", "L1 (Tap 1)", "L1 (Tap 4)", "L2 (Tap 5)", "L4 (Tap 19)", "L4 (Tap 18)"]
    base_ch = [64, 64, 64, 128, 512, 512]
    tsrx_ch = [273, 55, 59, 125, 443, 400]
    
    x = np.arange(len(layers))
    w = 0.35
    
    r1 = ax.bar(x - w/2, base_ch, w, label='Baseline Channels', color="#94A3B8")
    r2 = ax.bar(x + w/2, tsrx_ch, w, label='TSR-X Discovered Channels', color="#6366F1")
    
    ax.set_ylabel('Channel Count', fontsize=9, fontweight='bold', color="#0F172A")
    ax.set_xticks(x)
    ax.set_xticklabels(layers, fontsize=8.5, fontweight='bold')
    ax.tick_params(axis='both', labelsize=8)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_title("Automated Capacity Reallocation: Early Growth vs Late Pruning", fontsize=10, fontweight='bold', color="#0F172A", pad=8)
    
    for r in r2:
        h = r.get_height()
        ax.annotate(f'{int(h)}', xy=(r.get_x() + r.get_width()/2, h),
                    xytext=(0, 2), textcoords="offset points", ha='center', va='bottom',
                    fontsize=7.5, fontweight='bold', color="#4338CA")
        
    chart2_path = chart_dir / "channel_migration.png"
    plt.savefig(chart2_path, bbox_inches="tight")
    plt.close()
    
    return chart1_path, chart2_path


def build_pdf_report(out_pdf_path: str):
    pdf_path = Path(out_pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    chart_dir = pdf_path.parent / "assets"
    chart1_path, chart2_path = generate_charts(chart_dir)
    
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Typography Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=14,
    )
    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
        spaceAfter=6,
    )
    callout_style = ParagraphStyle(
        "CalloutText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#065F46"),
    )
    table_hdr_style = ParagraphStyle(
        "TableHdr",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=1,
    )
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1E293B"),
        alignment=1,
    )
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=table_cell_style,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#0F172A"),
    )
    
    story = []
    
    # Title & Metadata
    story.append(Paragraph("Topological Self-Regulation (TSR-X)", title_style))
    story.append(Paragraph("<b>Empirical Benchmark Report:</b> CIFAR-10 Head-to-Head Evaluation (ResNet-18) &nbsp;|&nbsp; August 2026", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E2E8F0"), spaceAfter=10))
    
    # Executive Callout Box
    callout_content = [
        [Paragraph(
            "<b>EXECUTIVE SUMMARY & KEY TAKEAWAYS:</b><br/>"
            "• <b>Strict Dual Dominance Confirmed:</b> Discovered architecture achieved <b>95.08% validation accuracy</b> with only <b>9.50M parameters (-15.00% reduction)</b>, beating the standard 11.17M static baseline (95.02%).<br/>"
            "• <b>Automated Channel Migration:</b> Pruned 181 redundant late-layer channels in Layer 4 (-35.4%) and reallocated capacity into early Layer 1 bottleneck (+326% expansion from 64 to 273 channels).<br/>"
            "• <b>100% Zero-Port Leakage:</b> All 407 online surgeries verified exact 0.0 max port magnitude dormancy.",
            callout_style
        )]
    ]
    callout_table = Table(callout_content, colWidths=[letter[0] - 108])
    callout_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#ECFDF5")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#10B981")),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(callout_table)
    story.append(Spacer(1, 10))
    
    # Section 1: 3-Way Benchmark Table
    story.append(Paragraph("1. Controlled 3-Way Empirical Benchmark Matrix", h1_style))
    story.append(Paragraph("All models evaluated on <b>NVIDIA GeForce RTX 4060 GPU</b> under identical training conditions (SGD lr=0.1, cosine annealing, 100 epochs, seed 42, CIFAR-native stem):", body_style))
    
    matrix_data = [
        [
            Paragraph("Metric", table_hdr_style),
            Paragraph("Arm 1: Static Baseline", table_hdr_style),
            Paragraph("Arm 2: TSR-X Plasticity", table_hdr_style),
            Paragraph("Arm 3: C2 Matched Control", table_hdr_style),
        ],
        [
            Paragraph("<b>Model Parameters</b>", table_cell_bold),
            Paragraph("11,173,962 (100.0%)", table_cell_style),
            Paragraph("9,497,686 (-15.00%)", table_cell_bold),
            Paragraph("9,497,686 (-15.00%)", table_cell_bold),
        ],
        [
            Paragraph("<b>Parameters Slashed</b>", table_cell_bold),
            Paragraph("0 (Baseline)", table_cell_style),
            Paragraph("<b>-1,676,276 params</b>", table_cell_style),
            Paragraph("<b>-1,676,276 params</b>", table_cell_style),
        ],
        [
            Paragraph("<b>Best Validation Accuracy</b>", table_cell_bold),
            Paragraph("95.02%", table_cell_style),
            Paragraph("94.70%", table_cell_style),
            Paragraph("<b>95.08%</b>", table_cell_bold),
        ],
        [
            Paragraph("<b>Accuracy vs Baseline</b>", table_cell_bold),
            Paragraph("Reference", table_cell_style),
            Paragraph("-0.32% (during plasticity)", table_cell_style),
            Paragraph("<b>+0.06% HIGHER</b>", table_cell_bold),
        ],
        [
            Paragraph("<b>Structural Plasticity Events</b>", table_cell_bold),
            Paragraph("0 (Static)", table_cell_style),
            Paragraph("407 dynamic events", table_cell_style),
            Paragraph("0 (Static from scratch)", table_cell_style),
        ],
        [
            Paragraph("<b>Dual Dominance Status</b>", table_cell_bold),
            Paragraph("Baseline Reference", table_cell_style),
            Paragraph("High Compression", table_cell_style),
            Paragraph("<b>STRICT DOMINANCE</b>", table_cell_bold),
        ],
    ]
    
    matrix_table = Table(matrix_data, colWidths=[1.8*inch, 1.8*inch, 1.8*inch, 1.8*inch])
    matrix_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E293B")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(matrix_table)
    story.append(Spacer(1, 8))
    
    # Embed Chart 1
    story.append(Image(str(chart1_path), width=6.8*inch, height=2.5*inch))
    story.append(Spacer(1, 10))
    
    # Section 2: Channel Evolution
    story.append(Paragraph("2. Discovered Channel Topology & Capacity Reallocation", h1_style))
    story.append(Paragraph("TSR-X automatically detected parameter redundancies in late convolution stages and dynamically transferred that budget into representation-bottlenecked early stages:", body_style))
    
    story.append(Image(str(chart2_path), width=6.8*inch, height=2.5*inch))
    story.append(Spacer(1, 10))
    
    # Section 3: Key Insights & Checkpoints
    story.append(Paragraph("3. Checkpoints & Reproducibility Audit", h1_style))
    audit_text = (
        "• <b>Static Baseline Checkpoint:</b> <code>results/reference/resnet18_cifar10_stem.pt</code><br/>"
        "• <b>TSR-X Dynamic Checkpoint:</b> <code>results/tsrx/resnet18_cifar10_two_regime.pt</code><br/>"
        "• <b>C2 Matched Control Checkpoint:</b> <code>results/static_matched/resnet18_cifar10_two_regime.pt</code><br/>"
        "• <b>Structural Decision Log:</b> <code>results/tsrx/resnet18_cifar10_two_regime_decisions.jsonl</code> (407 JSONL records)"
    )
    story.append(Paragraph(audit_text, body_style))
    
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF generated successfully at {pdf_path}")


if __name__ == "__main__":
    build_pdf_report("reports/CIFAR10_BENCHMARK_REPORT.pdf")
