"""
Generate the thesis methodology pipeline diagram.
Outputs: results/figures/methodology_pipeline.jpeg (150 dpi)
         results/figures/methodology_pipeline.pdf
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 10, 16
COL = 0.5          # horizontal centre of main column (figure fraction)
BOX_W = 0.62       # width of main single boxes
BOX_H = 0.052      # height of each single box
ARROW_DX = 0       # horizontal shift for arrows
GAP = 0.018        # gap between boxes

COLORS = {
    "data":        "#D6EAF8",   # light blue  — data stages
    "preprocess":  "#D5F5E3",   # light green — processing
    "train":       "#FEF9E7",   # light yellow — training
    "eval":        "#FDEDEC",   # light red/pink — evaluation
    "output":      "#E8DAEF",   # light purple — output / decision support
    "border":      "#2C3E50",   # dark navy border
    "arrow":       "#2C3E50",
    "panel_bg":    "#FFFDE7",   # training panel background
    "panel_border":"#F39C12",   # training panel border
    "text_dark":   "#1A1A2E",
    "subbox_ml":   "#D6EAF8",
    "subbox_bilstm":"#D5F5E3",
    "subbox_xlmr": "#FCF3CF",
}


def _box(ax, cx, cy, width, height, label, color, fontsize=10.5,
         bold=False, text_color="#1A1A2E", radius=0.012):
    """Draw a rounded rectangle centred at (cx, cy) with label."""
    left = cx - width / 2
    bottom = cy - height / 2
    box = FancyBboxPatch(
        (left, bottom), width, height,
        boxstyle=f"round,pad=0.01,rounding_size={radius}",
        linewidth=1.4, edgecolor=COLORS["border"], facecolor=color,
        transform=ax.transAxes, zorder=3,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(cx, cy, label, transform=ax.transAxes,
            ha="center", va="center", fontsize=fontsize,
            fontweight=weight, color=text_color, zorder=4,
            wrap=True)


def _arrow(ax, x, y_top, y_bottom, label=""):
    """Draw a downward arrow from y_top to y_bottom at horizontal x."""
    ax.annotate(
        "", xy=(x, y_bottom), xytext=(x, y_top),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="->,head_width=0.22,head_length=0.012",
            color=COLORS["arrow"], lw=1.6,
        ),
        zorder=5,
    )
    if label:
        ax.text(x + 0.015, (y_top + y_bottom) / 2, label,
                transform=ax.transAxes, fontsize=8, color="#555555",
                va="center", zorder=6)


def main():
    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.patch.set_facecolor("white")

    # Title
    ax.text(0.5, 0.975,
            "Methodology Pipeline: Amharic Social Media Sentiment Analysis",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=13, fontweight="bold", color=COLORS["text_dark"])

    # -----------------------------------------------------------------------
    # Define boxes top → bottom (centre y positions)
    # -----------------------------------------------------------------------
    # We'll space 9 main items + 1 compound panel evenly.
    # Manual y positions (in axes fraction, counting from top):
    y = [
        0.920,   # 0 Raw Amharic Social Media Data
        0.855,   # 1 Data Collection
        0.790,   # 2 Preprocessing
        0.725,   # 3 Annotated Dataset
        0.660,   # 4 Train/Val/Test Split
        0.560,   # 5 Model Training & Comparison PANEL (taller)
        0.455,   # 6 Evaluation
        0.390,   # 7 Best Model Selection
        0.325,   # 8 Sentiment Trend Analysis
        0.260,   # 9 Policy Decision Support Dashboard
    ]

    # Row 0 — Raw Data
    _box(ax, COL, y[0], BOX_W, BOX_H + 0.005,
         "Raw Amharic Social Media Data\n(Twitter/X · Facebook · Telegram)",
         COLORS["data"], fontsize=10.5, bold=True)

    # Row 1 — Data Collection
    _box(ax, COL, y[1], BOX_W, BOX_H,
         "Data Collection\n(AfriSenti ~8,950 tweets + 2,530 policy-domain tweets ≈ 11,480 total)",
         COLORS["data"], fontsize=9.5)

    # Row 2 — Preprocessing
    _box(ax, COL, y[2], BOX_W, BOX_H + 0.005,
         "Preprocessing\n(Unicode NFC norm. · Ethiopic char. normalisation · noise removal\n"
         "whitespace tokenisation · Amharic stopword filtering)",
         COLORS["preprocess"], fontsize=9.0)

    # Row 3 — Annotated Dataset
    _box(ax, COL, y[3], BOX_W, BOX_H,
         "Annotated Dataset\n(Positive 26.9 % · Negative 34.9 % · Neutral 38.2 %)",
         COLORS["data"], fontsize=9.5)

    # Row 4 — Split
    _box(ax, COL, y[4], BOX_W, BOX_H,
         "Train / Validation / Test Split\n(70 % / 15 % / 15 %, stratified — 7,680 / 1,646 / 1,646)",
         COLORS["preprocess"], fontsize=9.5)

    # -----------------------------------------------------------------------
    # Row 5 — Training panel (compound box with three sub-boxes)
    # -----------------------------------------------------------------------
    panel_top    = 0.610
    panel_bottom = 0.510
    panel_h      = panel_top - panel_bottom
    panel_left   = COL - BOX_W / 2
    panel_right  = COL + BOX_W / 2

    panel = FancyBboxPatch(
        (panel_left, panel_bottom), BOX_W, panel_h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.8, edgecolor=COLORS["panel_border"],
        facecolor=COLORS["panel_bg"],
        transform=ax.transAxes, zorder=2,
    )
    ax.add_patch(panel)
    ax.text(COL, panel_top - 0.012,
            "Model Training & Comparison",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=10.5, fontweight="bold", color=COLORS["text_dark"], zorder=4)

    # Three sub-boxes inside the panel
    sub_y   = panel_bottom + 0.026
    sub_h   = 0.050
    sub_w   = 0.174
    sub_gap = 0.012
    sub_centres = [
        panel_left + sub_w / 2 + sub_gap,
        COL,
        panel_right - sub_w / 2 - sub_gap,
    ]
    sub_labels = [
        "ML Baselines\n(Naïve Bayes · LR\nSVM · KNN)",
        "Bi-LSTM\n(char-level,\n2-layer, att.)",
        "XLM-RoBERTa\n(multilingual\ntransformer)",
    ]
    sub_colors = [COLORS["subbox_ml"], COLORS["subbox_bilstm"], COLORS["subbox_xlmr"]]
    for cx, lbl, col in zip(sub_centres, sub_labels, sub_colors):
        sub = FancyBboxPatch(
            (cx - sub_w / 2, sub_y), sub_w, sub_h,
            boxstyle="round,pad=0.008,rounding_size=0.010",
            linewidth=1.2, edgecolor=COLORS["border"], facecolor=col,
            transform=ax.transAxes, zorder=4,
        )
        ax.add_patch(sub)
        ax.text(cx, sub_y + sub_h / 2, lbl,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8.5, color=COLORS["text_dark"], zorder=5)

    # Row 6 — Evaluation
    _box(ax, COL, y[6], BOX_W, BOX_H,
         "Evaluation\n(Accuracy · Macro-Precision · Macro-Recall · Macro-F1)",
         COLORS["eval"], fontsize=9.5)

    # Row 7 — Best Model Selection
    _box(ax, COL, y[7], BOX_W, BOX_H,
         "Best Model Selection\n(held-out test set, macro-F1 criterion)",
         COLORS["eval"], fontsize=9.5)

    # Row 8 — Sentiment Trend Analysis
    _box(ax, COL, y[8], BOX_W, BOX_H,
         "Sentiment Trend Analysis & Insights\n(per-class F1 · error analysis · cross-model comparison)",
         COLORS["output"], fontsize=9.5)

    # Row 9 — Policy Dashboard
    _box(ax, COL, y[9], BOX_W, BOX_H + 0.005,
         "Policy Decision Support Dashboard\n"
         "(real-time public-sentiment monitoring for Ethiopian government stakeholders)",
         COLORS["output"], fontsize=9.5, bold=True)

    # -----------------------------------------------------------------------
    # Arrows between boxes
    # -----------------------------------------------------------------------
    arrow_specs = [
        (y[0] - BOX_H / 2 - 0.004,   y[1] + (BOX_H + 0.005) / 2 + 0.003),
        (y[1] - BOX_H / 2 - 0.004,   y[2] + (BOX_H + 0.005) / 2 + 0.003),
        (y[2] - (BOX_H + 0.005) / 2 - 0.004, y[3] + BOX_H / 2 + 0.003),
        (y[3] - BOX_H / 2 - 0.004,   y[4] + BOX_H / 2 + 0.003),
        (y[4] - BOX_H / 2 - 0.004,   panel_top + 0.003),
        (panel_bottom - 0.004,        y[6] + BOX_H / 2 + 0.003),
        (y[6] - BOX_H / 2 - 0.004,   y[7] + BOX_H / 2 + 0.003),
        (y[7] - BOX_H / 2 - 0.004,   y[8] + BOX_H / 2 + 0.003),
        (y[8] - BOX_H / 2 - 0.004,   y[9] + (BOX_H + 0.005) / 2 + 0.003),
    ]
    for y_top, y_bot in arrow_specs:
        _arrow(ax, COL, y_top, y_bot)

    # -----------------------------------------------------------------------
    # Footer note
    # -----------------------------------------------------------------------
    ax.text(0.5, 0.012,
            "Figure: Thesis methodology pipeline — each stage is a prerequisite for the next.",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8, color="#777777", style="italic")

    plt.tight_layout(pad=0.3)

    jpeg_path = out_dir / "methodology_pipeline.jpeg"
    pdf_path  = out_dir / "methodology_pipeline.pdf"
    fig.savefig(jpeg_path, dpi=150, bbox_inches="tight", format="jpeg")
    fig.savefig(pdf_path,  dpi=150, bbox_inches="tight", format="pdf")
    plt.close(fig)

    print(f"Saved: {jpeg_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
