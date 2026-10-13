"""Pipeline workflow diagram for the Stage 1 report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def draw_box(ax, cx, cy, w, h, text, color="#dbe9f6", fontsize=8.5):
    """Box centered at (cx, cy). Returns dict of anchor points: N,S,E,W."""
    x, y = cx - w / 2, cy - h / 2
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                        linewidth=1.1, edgecolor="#333333", facecolor=color)
    ax.add_patch(b)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize)
    return {"N": (cx, y + h), "S": (cx, y), "E": (x + w, cy), "W": (x, cy)}


def arrow(ax, p1, p2, color="#333333"):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12,
                         linewidth=1.1, color=color, connectionstyle="arc3,rad=0.0")
    ax.add_patch(a)


def make(out_path="output/workflow_diagram.png"):
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 9.2)
    ax.axis("off")

    b1 = draw_box(ax, 1.7, 8.5, 3.0, 0.7, "FRED-MD raw CSV\n(128 series, 1959-2026)")
    b2 = draw_box(ax, 5.5, 8.5, 3.0, 0.7, "prep_data.py\nselect 60 series + transforms")
    b3 = draw_box(ax, 9.2, 8.5, 2.8, 0.7, "fredmd_clean.parquet\n1959-03 to 2022-12")
    arrow(ax, b1["E"], b2["W"])
    arrow(ax, b2["E"], b3["W"])

    bE = draw_box(ax, 2.6, 7.0, 4.4, 0.75, "Econometric: AR(1) | BVAR (Minnesota) | Factor-AR", "#f6e6d0")
    bT = draw_box(ax, 8.2, 7.0, 4.6, 0.75, "6 TSFMs, zero-shot: Chronos-Bolt | TimesFM | Moirai | TTM | LagLlama | Time-GPT", "#dcecdc", fontsize=7.8)
    arrow(ax, (9.2, 8.5 - 0.35), bE["N"], color="#666666")
    arrow(ax, b3["S"], bT["N"])

    bL = draw_box(ax, 5.5, 5.5, 8.6, 0.75,
                  "run_forecast_loop.py — expanding-window recursion\n48 quarterly origins (2010-2021) x horizons {1,3,6,12}", "#eef0f2")
    arrow(ax, bE["S"], (bL["W"][0] + 1.0, bL["N"][1]))
    arrow(ax, bT["S"], (bL["E"][0] - 1.0, bL["N"][1]))

    bR = draw_box(ax, 2.2, 4.0, 3.6, 0.75, "BVAR in-sample residuals\n(bvar_residuals)", "#f6e6d0")
    bH = draw_box(ax, 7.2, 4.0, 4.4, 0.75, "run_hybrid.py: BVAR forecast +\nTimesFM-modeled residual forecast", "#e8dcf2")
    arrow(ax, (3.0, bL["S"][1]), bR["N"])
    arrow(ax, bR["E"], bH["W"])
    arrow(ax, (8.0, bL["S"][1]), bH["N"])  # TSFM branch also feeds the hybrid's residual model

    bV = draw_box(ax, 5.5, 2.5, 8.6, 0.75,
                  "rmsfe_table(): RMSFE = sqrt(Sum e_model^2 / Sum e_AR1^2)\nmerge_tsfm_errors.py folds every model into one error table", "#eef0f2")
    arrow(ax, (4.0, bL["S"][1] - 1.2), bV["N"], color="#999999")  # loop's own AR1/BVAR/Factor errors
    arrow(ax, bH["S"], (7.5, bV["N"][1]))

    bO1 = draw_box(ax, 3.0, 0.9, 4.4, 0.7, "rmsfe_summary.csv\nmedian/std/min/max by horizon", "#f0f0d8")
    bO2 = draw_box(ax, 8.0, 0.9, 4.4, 0.7, "rmsfe_boxplots.png\n(Figure 2/3 style)", "#f0f0d8")
    arrow(ax, (4.0, bV["S"][1]), bO1["N"])
    arrow(ax, (7.0, bV["S"][1]), bO2["N"])

    fig.suptitle("Figure 1 — End-to-end pipeline", fontsize=13, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    make()
