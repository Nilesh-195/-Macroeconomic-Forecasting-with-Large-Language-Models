"""
Generate high-resolution block diagram for Modified Research Method (Stage 2: FinBERT Multimodal Engine).
"""
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_stage2"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

ROOT = Path("/Users/nilesh/LLM")
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(exist_ok=True)
IMG_PATH = OUT_DIR / "block_diagram_stage2_modified.png"

fig = plt.figure(figsize=(18, 9), facecolor="#f8fafc")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 18)
ax.set_ylim(0, 9)
ax.axis("off")

# Title Banner
ax.text(9.0, 8.4, "Stage 2: Modified Multimodal Architecture (FinBERT + TimesFM 2.5)",
        ha="center", va="center", fontsize=18, fontweight="bold", color="#0f172a")
ax.text(9.0, 8.0, "Augmenting Foundation Time-Series Models with 64 Years of Semantic Financial News",
        ha="center", va="center", fontsize=11, color="#64748b")

def draw_card(x, y, w, h, title, subtitle, items, bg_color, border_color, title_color="#0f172a"):
    shadow = FancyBboxPatch((x + 0.05, y - 0.05), w, h, boxstyle="round,pad=0.1,rounding_size=0.15",
                            facecolor="#cbd5e1", edgecolor="none", alpha=0.5, zorder=1)
    ax.add_patch(shadow)
    card = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.15",
                          facecolor=bg_color, edgecolor=border_color, linewidth=2, zorder=2)
    ax.add_patch(card)
    
    ax.text(x + w/2, y + h - 0.35, title, ha="center", va="center", fontsize=11.5, fontweight="bold", color=title_color, zorder=3)
    if subtitle:
        ax.text(x + w/2, y + h - 0.65, subtitle, ha="center", va="center", fontsize=9, fontstyle="italic", color="#475569", zorder=3)
    
    ax.plot([x + 0.2, x + w - 0.2], [y + h - 0.85, y + h - 0.85], color=border_color, linewidth=1, zorder=3)
    
    curr_y = y + h - 1.15
    for item in items:
        ax.text(x + 0.3, curr_y, f"• {item}", ha="left", va="center", fontsize=8.8, color="#1e293b", zorder=3)
        curr_y -= 0.33

def draw_arrow(x1, y1, x2, y2, text=""):
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                            linewidth=2.2, color="#2563eb", zorder=4)
    ax.add_patch(arrow)
    if text:
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2 + 0.18
        ax.text(mid_x, mid_y, text, ha="center", va="center", fontsize=8, fontweight="bold", color="#1d4ed8",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#eff6ff", edgecolor="#bfdbfe", alpha=0.9), zorder=5)

# Top Stream: News Text Processing
draw_card(0.6, 4.7, 3.8, 2.8,
          "1A. Unstructured News", "NYT Historical Archives",
          ["1,575,882 Macro Articles", "1959–2023 (747 Months)", "Inflation, Fed, Trade,", "Banking, Labor, Energy"],
          "#eff6ff", "#3b82f6", "#1d4ed8")

draw_card(4.9, 4.7, 3.8, 2.8,
          "2A. Local FinBERT GPU", "Apple M4 Silicon ('mps')",
          ["ProsusAI/finbert (110M)", "Full Forward Pass @ 520 art/s", "768-dim [CLS] Embeddings", "3-Class Sentiment Logits"],
          "#fdf4ff", "#d946ef", "#a21caf")

draw_arrow(4.5, 6.1, 4.8, 6.1, "1.57M Text")

draw_card(9.2, 4.7, 3.8, 2.8,
          "3A. PCA Factor Reduction", "Dimensionality Compression",
          ["768 dims → 8 PCA Factors", "Retains 92.1% Total Variance", "+3 Sentiment (Pos, Neg, Neu)", "Total 11 News Covariates"],
          "#fefce8", "#eab308", "#a16207")

draw_arrow(8.8, 6.1, 9.1, 6.1, "768-dim")

# Bottom Stream: Macro Numerical Data
draw_card(0.6, 1.2, 3.8, 2.8,
          "1B. Numerical Series", "FRED-MD Database",
          ["60 Core Macro Indicators", "McCracken-Ng Transforms", "Stationary Target Series", "Jan 1959 – Dec 2022"],
          "#f8fafc", "#64748b", "#334155")

draw_card(4.9, 1.2, 3.8, 2.8,
          "2B. Context Expansion", "Business Cycle Memory",
          ["Context: 120 Months (10 Yrs)", "Captures Full Boom & Bust", "Preserves Seasonal Patterns", "Replaces 36-Month Window"],
          "#ecfdf5", "#10b981", "#047857")

draw_arrow(4.5, 2.6, 4.8, 2.6, "Stationary")

draw_card(9.2, 1.2, 3.8, 2.8,
          "3B. Google TimesFM 2.5", "Foundation Time-Series Model",
          ["200M Parameter Transformer", "Zero-Shot Numerical Prior", "Generates Base Trajectory", "Horizons h = 1 to 12"],
          "#f0fdf4", "#22c55e", "#15803d")

draw_arrow(8.8, 2.6, 9.1, 2.6, "120-mo Ctx")

# Merging Step 4: Semantic Meta-Learner
draw_card(13.5, 2.5, 3.9, 3.8,
          "4. Rolling Meta-Learner", "Ridge Semantic Fusion",
          ["Inputs: TimesFM Forecast", "+ 8 PCA News Factors", "+ 3 FinBERT Sentiment Scores", "Rolling 144 Monthly Origins", "Ridge Regularized (alpha=1.0)", "Adjusts for Narrative Shifts", "Outputs Final h=1..12 Forecast"],
          "#fff1f2", "#f43f5e", "#be123c")

draw_arrow(13.1, 6.1, 13.4, 4.8, "11 News Signals")
draw_arrow(13.1, 2.6, 13.4, 3.8, "Base Forecast")

plt.savefig(IMG_PATH, dpi=300)
print(f"Modified Method Diagram saved successfully to: {IMG_PATH}")

