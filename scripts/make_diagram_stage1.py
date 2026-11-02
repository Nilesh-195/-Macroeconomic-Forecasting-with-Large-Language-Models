"""
Generate high-resolution block diagram for Original Research Method (Stage 1 / Paper).
"""
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_stage1"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

ROOT = Path("/Users/nilesh/LLM")
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(exist_ok=True)
IMG_PATH = OUT_DIR / "block_diagram_stage1_original.png"

fig = plt.figure(figsize=(16, 8), facecolor="#f8fafc")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 8)
ax.axis("off")

# Title Banner
ax.text(8.0, 7.4, "Stage 1: Original Research Method Architecture (Carriero et al., 2026)",
        ha="center", va="center", fontsize=18, fontweight="bold", color="#0f172a")
ax.text(8.0, 7.0, "Zero-Shot Time Series Foundation Models vs. Classical Econometric Baselines",
        ha="center", va="center", fontsize=11, color="#64748b")

def draw_card(x, y, w, h, title, subtitle, items, bg_color, border_color, title_color="#0f172a"):
    # Card shadow
    shadow = FancyBboxPatch((x + 0.05, y - 0.05), w, h, boxstyle="round,pad=0.1,rounding_size=0.15",
                            facecolor="#cbd5e1", edgecolor="none", alpha=0.5, zorder=1)
    ax.add_patch(shadow)
    # Card body
    card = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.15",
                          facecolor=bg_color, edgecolor=border_color, linewidth=2, zorder=2)
    ax.add_patch(card)
    
    # Header
    ax.text(x + w/2, y + h - 0.35, title, ha="center", va="center", fontsize=12, fontweight="bold", color=title_color, zorder=3)
    if subtitle:
        ax.text(x + w/2, y + h - 0.65, subtitle, ha="center", va="center", fontsize=9, fontstyle="italic", color="#475569", zorder=3)
    
    # Divider line
    ax.plot([x + 0.2, x + w - 0.2], [y + h - 0.85, y + h - 0.85], color=border_color, linewidth=1, zorder=3)
    
    # Bullet items
    curr_y = y + h - 1.15
    for item in items:
        ax.text(x + 0.3, curr_y, f"• {item}", ha="left", va="center", fontsize=9, color="#1e293b", zorder=3)
        curr_y -= 0.35

def draw_arrow(x1, y1, x2, y2, text=""):
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18,
                            linewidth=2.5, color="#2563eb", zorder=4)
    ax.add_patch(arrow)
    if text:
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2 + 0.2
        ax.text(mid_x, mid_y, text, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#1d4ed8",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#eff6ff", edgecolor="#bfdbfe", alpha=0.9), zorder=5)

# 1. Dataset Box
draw_card(0.8, 2.0, 3.2, 4.2,
          "1. FRED-MD Dataset", "Federal Reserve St. Louis",
          ["60-120 Macro Series", "Jan 1959 – Dec 2022", "Monthly Frequency", "Output, Labor, Prices,", "Rates, Money, Housing", "McCracken-Ng Codes", "Stationary Transform"],
          "#eff6ff", "#3b82f6", "#1d4ed8")

# Arrow 1 -> 2
draw_arrow(4.1, 4.1, 5.0, 4.1, "Stationary Data")

# 2A. Econometric Baselines (Top)
draw_card(5.1, 4.3, 4.4, 2.0,
          "2A. Econometric Models", "Statistical Benchmarks",
          ["AR(1) (Mandatory Base)", "BVAR (Minnesota Prior)", "Factor-AR (PCA Factors)"],
          "#fef2f2", "#ef4444", "#b91c1c")

# 2B. Zero-Shot Foundation Models (Bottom)
draw_card(5.1, 1.6, 4.4, 2.4,
          "2B. Zero-Shot TSFMs", "Time Series Foundation Models",
          ["Google TimesFM 2.5", "Amazon Chronos-Bolt", "Salesforce Moirai", "Nixtla TimeGPT", "Lag-Llama & IBM TTM"],
          "#f0fdf4", "#22c55e", "#15803d")

# Arrows 2 -> 3
draw_arrow(9.6, 5.3, 10.6, 4.5, "Parametric")
draw_arrow(9.6, 2.8, 10.6, 3.7, "Zero-Shot")

# 3. Forecast Evaluation Box
draw_card(10.7, 1.8, 4.5, 4.4,
          "3. Out-of-Sample Evaluation", "Recursive Rolling Window",
          ["Context: 36 Months", "Horizons: h = 1, 3, 6, 12", "Origins: 1985-2019 (Paper)", "Origins: 2010-2022 (Repro)", "Metric: Relative RMSFE", "Relative to AR(1) baseline", "Value < 1.0 beats AR(1)"],
          "#faf5ff", "#a855f7", "#7e22ce")

plt.savefig(IMG_PATH, dpi=300)
print(f"Original Method Diagram saved successfully to: {IMG_PATH}")

