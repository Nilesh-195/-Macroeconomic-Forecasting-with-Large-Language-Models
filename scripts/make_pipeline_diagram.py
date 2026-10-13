"""
Professional pipeline diagram for the Stage 1 project.
Run from /Users/nilesh/LLM:
    python scripts/make_pipeline_diagram.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

# ── Colour palette ────────────────────────────────────────────────────────────
BG          = "#0D1117"
CARD_DARK   = "#161B22"
CARD_MID    = "#21262D"
BORDER      = "#30363D"

C_DATA      = "#1F6FEB"   # blue  — data
C_ECON      = "#388BFD"   # lighter blue — econometric
C_TSFM      = "#3FB950"   # green — TSFMs
C_HYBRID    = "#D29922"   # yellow — hybrid
C_EVAL      = "#BC8CFF"   # purple — evaluation
C_OUT       = "#F78166"   # orange/red — output
C_ARROW     = "#8B949E"
WHITE       = "#E6EDF3"
GREY        = "#8B949E"
ACCENT      = "#58A6FF"

fig = plt.figure(figsize=(20, 13), facecolor=BG)
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 20)
ax.set_ylim(0, 13)
ax.set_facecolor(BG)
ax.axis("off")

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def card(x, y, w, h, color, alpha=0.18, radius=0.25):
    """Draw a rounded-rect card with coloured glow border."""
    # shadow / glow
    glow = FancyBboxPatch((x-0.04, y-0.04), w+0.08, h+0.08,
                           boxstyle=f"round,pad=0,rounding_size={radius+0.05}",
                           linewidth=0, facecolor=color, alpha=0.12, zorder=1)
    ax.add_patch(glow)
    # fill
    fill = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad=0,rounding_size={radius}",
                           linewidth=1.4, edgecolor=color,
                           facecolor=CARD_DARK, alpha=1, zorder=2)
    ax.add_patch(fill)
    # top accent bar
    bar = FancyBboxPatch((x+0.05, y+h-0.12), w-0.1, 0.1,
                          boxstyle="round,pad=0,rounding_size=0.05",
                          linewidth=0, facecolor=color, alpha=0.85, zorder=3)
    ax.add_patch(bar)


def label(x, y, text, size=8.5, color=WHITE, weight="normal", ha="center", va="center", zorder=5):
    ax.text(x, y, text, color=color, fontsize=size, fontweight=weight,
            ha=ha, va=va, zorder=zorder,
            fontfamily="monospace" if "`" in text else "DejaVu Sans")


def sub(x, y, text, size=7, color=GREY):
    ax.text(x, y, text, color=color, fontsize=size, ha="center", va="center", zorder=5,
            style="italic")


def arrow(x1, y1, x2, y2, color=C_ARROW, lw=1.5, style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                 arrowprops=dict(arrowstyle=style, color=color,
                                 lw=lw, connectionstyle="arc3,rad=0.0"),
                 zorder=4)


def curved_arrow(x1, y1, x2, y2, color=C_ARROW, lw=1.5, rad=0.25):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                 arrowprops=dict(arrowstyle="->", color=color,
                                 lw=lw, connectionstyle=f"arc3,rad={rad}"),
                 zorder=4)


def section_title(x, y, text, color):
    ax.text(x, y, text, color=color, fontsize=8, fontweight="bold",
            ha="center", va="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.18,
                      edgecolor=color, linewidth=0.8))


# ─────────────────────────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────────────────────────
ax.text(10, 12.5, "Macroeconomic Forecasting with Large Language Models",
        color=WHITE, fontsize=17, fontweight="bold", ha="center", va="center", zorder=6)
ax.text(10, 12.1, "End-to-End Forecasting Pipeline  ·  Stage 1 Reproduction  ·  Carriero, Pettenuzzo & Shekhar (2026)",
        color=GREY, fontsize=9, ha="center", va="center", zorder=6)

# divider
ax.plot([0.5, 19.5], [11.78, 11.78], color=BORDER, lw=0.8, zorder=3)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 1 — DATA SOURCE
# ─────────────────────────────────────────────────────────────────────────────
section_title(10, 11.45, "① DATA", C_DATA)

# FRED-MD card
card(3.5, 10.2, 5.5, 1.0, C_DATA)
label(6.25, 10.9,  "FRED-MD Dataset",        size=10, weight="bold", color=C_DATA)
label(6.25, 10.58, "120 Monthly US Macro Series  ·  Jan 1959 – Dec 2022", size=8)
label(6.25, 10.3,  "Output · Labor · Housing · Money · Rates · Prices · Stocks", size=7, color=GREY)

# Subset card
card(11.0, 10.2, 5.5, 1.0, C_DATA)
label(13.75, 10.9,  "60-Series Subset  (prep_data.py)",  size=10, weight="bold", color=C_DATA)
label(13.75, 10.58, "Category-proportional  ·  Same 8 macro groups", size=8)
label(13.75, 10.3,  "McCracken-Ng Transform Codes 1–7 applied per series", size=7, color=GREY)

arrow(9.0, 10.7, 11.0, 10.7, color=C_DATA, lw=2)
ax.text(10.0, 10.82, "select", color=C_DATA, fontsize=7, ha="center", va="bottom")

# ─────────────────────────────────────────────────────────────────────────────
# ROW 2 — EXPANDING WINDOW ENGINE
# ─────────────────────────────────────────────────────────────────────────────
arrow(13.75, 10.2, 13.75, 9.55, color=C_ARROW, lw=1.8)
arrow(6.25, 10.2, 6.25, 8.35, color=C_ARROW, lw=1.8)   # to BVAR later

section_title(10, 9.45, "② EXPANDING WINDOW FORECAST LOOP  (run_forecast_loop.py)", C_ECON)

card(6.5, 8.45, 7.0, 0.85, C_ECON, alpha=0.12)
label(10.0, 8.94, "48 Quarterly Origins  ·  2010 Q1 → 2022 Q4", size=9.5, weight="bold", color=C_ECON)
label(10.0, 8.62, "Each origin: train on all data up to that date  →  forecast h = 1, 3, 6, 12 months ahead", size=7.8)

# show expanding arrow
ax.annotate("", xy=(17.5, 8.87), xytext=(6.5, 8.87),
             arrowprops=dict(arrowstyle="-|>", color=C_ECON, lw=1.2,
                             mutation_scale=10), zorder=5)
ax.text(17.65, 8.87, "+1Q", color=C_ECON, fontsize=7, va="center")

# ─────────────────────────────────────────────────────────────────────────────
# ROW 3 — MODELS (split into two tracks)
# ─────────────────────────────────────────────────────────────────────────────
section_title(3.3,  8.0, "③ ECONOMETRIC MODELS", C_ECON)
section_title(13.2, 8.0, "③ ZERO-SHOT TSFM MODELS", C_TSFM)

# ── ECONOMETRIC ──────────────────────────────────────────────────────────────
arrow(10.0, 8.45, 3.3, 7.85, color=C_ECON, lw=1.5)
arrow(10.0, 8.45, 13.2, 7.85, color=C_TSFM, lw=1.5)

# AR(1)
card(0.3, 6.8, 2.8, 0.85, C_ECON)
label(1.7, 7.36, "AR(1)",          size=9.5, weight="bold", color=C_ECON)
label(1.7, 7.05, "OLS benchmark",  size=7.2, color=GREY)

# BVAR
card(0.3, 5.7, 2.8, 0.95, C_ECON)
label(1.7, 6.35, "BVAR",            size=9.5, weight="bold", color=C_ECON)
label(1.7, 6.08, "Minnesota prior", size=7.2)
label(1.7, 5.82, "p=6 lags · λ=0.2", size=7, color=GREY)

# Factor-AR
card(0.3, 4.55, 2.8, 0.95, C_ECON)
label(1.7, 5.18, "Factor-AR",        size=9.5, weight="bold", color=C_ECON)
label(1.7, 4.91, "PCA + direct-h",   size=7.2)
label(1.7, 4.65, "regression (Eq 6–7)", size=7, color=GREY)

# vertical connector for econ models
ax.plot([1.7, 1.7], [7.65, 4.55], color=C_ECON, lw=1.0, ls="--", alpha=0.4, zorder=2)
ax.plot([1.7, 1.7], [7.65, 7.65], color=C_ECON, lw=1.0, ls="--", alpha=0.4, zorder=2)

# ── TSFMs ─────────────────────────────────────────────────────────────────────
tsfm_models = [
    ("Chronos",   "Amazon · ~200M",   "Bolt-small · quantization", C_TSFM),
    ("TimesFM",   "Google · 200M",    "v2.5 · patch-based",        C_TSFM),
    ("Moirai",    "Salesforce · 91M", "Base · multi-resolution",   C_TSFM),
    ("LagLlama",  "Meta · 25M",       "Llama-based · lag feats",   C_TSFM),
    ("TTM",       "IBM · 5M",         "Tiny Time Mixer",           C_TSFM),
    ("Time-GPT",  "Nixtla · ~311M",   "Hosted API · batch",        C_TSFM),
]

xs = [9.8, 11.6, 13.4, 15.2, 17.0, 18.8]
ys_top = 7.65

for i, (name, provider, detail, col) in enumerate(tsfm_models):
    x0 = xs[i] - 0.82
    card(x0, 4.55, 1.62, 3.0, col, alpha=0.12, radius=0.2)
    label(xs[i], 7.32, name,     size=8.5, weight="bold", color=col)
    label(xs[i], 7.03, provider, size=6.8, color=GREY)
    # divider
    ax.plot([x0+0.1, x0+1.52], [6.82, 6.82], color=BORDER, lw=0.6, zorder=4)
    label(xs[i], 6.6,  "Zero-Shot", size=7.2, color=ACCENT)
    label(xs[i], 6.33, detail,      size=6.5, color=GREY)
    # small icon dots
    for j in range(3):
        ax.plot(xs[i]-0.2+j*0.2, 5.9, "o", ms=3, color=col, alpha=0.5, zorder=5)
    label(xs[i], 5.55, "No fine-tuning", size=6.5, color=GREY)
    label(xs[i], 5.25, "Full history", size=6.5, color=GREY)
    label(xs[i], 4.9,  "context", size=6.5, color=GREY)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 4 — HYBRID MODEL
# ─────────────────────────────────────────────────────────────────────────────
section_title(6.6, 4.25, "④ HYBRID TSFM-BVAR MODEL  (run_hybrid.py)", C_HYBRID)

# arrows from BVAR and TimesFM down to hybrid
arrow(1.7, 5.7, 4.6, 3.85, color=C_HYBRID, lw=1.5)
arrow(11.6, 4.55, 7.6, 3.85, color=C_HYBRID, lw=1.5)

card(3.8, 2.95, 5.6, 0.82, C_HYBRID, alpha=0.15)
label(6.6, 3.53, "BVAR Forecast  +  TimesFM( BVAR Residuals )", size=9.5, weight="bold", color=C_HYBRID)
label(6.6, 3.19, "ŷ_hybrid  =  ŷ_BVAR  +  TimesFM̂(y − ŷ_BVAR)   ·   Corrects systematic BVAR bias", size=7.8)

# step labels
for i, (sx, sy, txt) in enumerate([
    (3.3, 3.67, "① BVAR\nforecast"),
    (5.3, 3.67, "② Extract\nresiduals"),
    (7.2, 3.67, "③ TSFM\npredicts residual"),
    (9.5, 3.67, "④ Add\ncorrection"),
]):
    pass  # simplified — label inline above

# ─────────────────────────────────────────────────────────────────────────────
# ROW 5 — EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
section_title(13.5, 4.25, "⑤ EVALUATION METRIC", C_EVAL)

card(10.2, 2.95, 8.8, 0.82, C_EVAL, alpha=0.12)
label(14.6, 3.53, "RMSFE  =  √( Σ e²_model  /  Σ e²_AR1 )     per series  ×  per horizon", size=9.5, weight="bold", color=C_EVAL)
label(14.6, 3.19, "< 1.0  →  beats AR(1) benchmark   ·   median across 60 series reported   ·   Paper's Equation 11", size=7.8)

# arrows into eval
arrow(1.7,  4.55, 10.5, 3.45, color=C_EVAL, lw=1.2)   # AR1
arrow(9.4,  2.95, 10.5, 3.45, color=C_EVAL, lw=1.2)   # hybrid
arrow(18.8, 4.55, 19.0, 3.45, color=C_EVAL, lw=1.2)   # TSFMs

# ─────────────────────────────────────────────────────────────────────────────
# ROW 6 — RESULTS
# ─────────────────────────────────────────────────────────────────────────────
arrow(10.0, 2.95, 10.0, 2.38, color=C_OUT, lw=2.0)
arrow(14.6, 2.95, 14.6, 2.38, color=C_OUT, lw=2.0)

section_title(10.0, 2.22, "⑥ RESULTS & OUTPUTS", C_OUT)

# three output cards
for i, (ox, ow, title, body, col) in enumerate([
    (0.3,  6.0,  "rmsfe_summary.csv",  "Median · Std · Min · Max per model × horizon", C_OUT),
    (6.5,  6.5,  "rmsfe_boxplots.png", "Distribution across all 60 series (mirrors paper Fig. 2/3)", C_OUT),
    (13.2, 6.5,  "STAGE1_REPORT_FULL.pdf + STAGE1_COMPARISON.pdf", "Full report + paper vs. ours side-by-side", C_OUT),
]):
    card(ox, 0.45, ow, 1.4, col, alpha=0.12)
    label(ox+ow/2, 1.5,  title, size=8.5, weight="bold", color=col)
    label(ox+ow/2, 1.18, body,  size=7.2)
    arrow(ox+ow/2, 2.2, ox+ow/2, 1.85, color=col, lw=1.5)

# ─────────────────────────────────────────────────────────────────────────────
# LEGEND
# ─────────────────────────────────────────────────────────────────────────────
legend_items = [
    (C_DATA,   "Data / Preparation"),
    (C_ECON,   "Econometric Models"),
    (C_TSFM,   "Zero-Shot TSFMs"),
    (C_HYBRID, "Hybrid Model"),
    (C_EVAL,   "Evaluation"),
    (C_OUT,    "Outputs"),
]
for i, (col, txt) in enumerate(legend_items):
    lx = 0.45 + i * 3.25
    ax.plot(lx, 0.22, "s", ms=8, color=col, zorder=6)
    ax.text(lx+0.15, 0.22, txt, color=WHITE, fontsize=7, va="center", zorder=6)

ax.plot([0.3, 19.7], [0.38, 0.38], color=BORDER, lw=0.6, zorder=3)

plt.savefig("output/pipeline.png", dpi=160, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
print("✅  Saved: output/pipeline.png")
