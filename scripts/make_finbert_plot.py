"""
Generate High-Resolution Benchmark Plot: FinBERT Hybrid vs Google TimesFM 2.5
across all 12 forecast horizons (h=1..12).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load forecasts
df = pd.read_parquet("output/forecasts_finbert_hybrid.parquet")

horizons = list(range(1, 13))
rmse_tsfm = []
rmse_fb = []
gains = []

for h in horizons:
    sub = df[df["horizon"] == h]
    tf = np.sqrt((sub["err_tsfm"]**2).mean())
    fb = np.sqrt((sub["err_finbert"]**2).mean())
    rmse_tsfm.append(tf)
    rmse_fb.append(fb)
    gains.append(((tf - fb) / tf) * 100)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1.2]})

# Top plot: Raw RMSFE
ax1.plot(horizons, rmse_tsfm, marker="o", color="#d95f02", linewidth=2.5, label="Google TimesFM 2.5 (10-Yr Context)")
ax1.plot(horizons, rmse_fb, marker="s", color="#1b9e77", linewidth=2.5, label="FinBERT-Augmented Hybrid")
ax1.fill_between(horizons, rmse_tsfm, rmse_fb, where=[fb < tf for fb, tf in zip(rmse_fb, rmse_tsfm)],
                 color="#1b9e77", alpha=0.18, label="FinBERT Semantic Gain")
ax1.set_ylabel("RMSFE (Lower is Better)", fontsize=12, fontweight="bold")
ax1.set_title("Macroeconomic Forecast Error Across Horizons h = 1 to 12 Months\n(Evaluated on 60 FRED-MD Series, 144 Rolling Origins, 1.57M Embedded Articles)",
              fontsize=13, fontweight="bold", pad=12)
ax1.legend(loc="upper left", frameon=True, fontsize=10)
ax1.grid(True, linestyle="--", alpha=0.6)

# Bottom plot: FinBERT Gain %
bars = ax2.bar(horizons, gains, color=["#e41a1c" if g < 0 else "#2ca02c" for g in gains], width=0.55, edgecolor="black", alpha=0.85)
ax2.axhline(0, color="black", linewidth=1.0)
ax2.set_xlabel("Forecast Horizon (Months Ahead)", fontsize=12, fontweight="bold")
ax2.set_ylabel("FinBERT Gain (%)", fontsize=11, fontweight="bold")
ax2.set_xticks(horizons)
ax2.set_xticklabels([f"h={h}" for h in horizons], fontweight="bold")
ax2.grid(True, linestyle="--", alpha=0.6)

# Annotate bars
for bar in bars:
    yval = bar.get_height()
    va = "bottom" if yval >= 0 else "top"
    ax2.annotate(f"{yval:+.1f}%",
                 xy=(bar.get_x() + bar.get_width() / 2, yval),
                 xytext=(0, 3 if yval >= 0 else -10),
                 textcoords="offset points",
                 ha="center", va=va, fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("output/finbert_vs_timesfm_horizons.png", dpi=300)
print("Plot successfully saved to output/finbert_vs_timesfm_horizons.png")

