"""Stage 1, Section 9 deliverable: boxplots of RMSFE-vs-AR(1) by model and
horizon, mirroring the paper's Figure 2/3. Pre-COVID window (origins before
2019-06), matching the paper's own choice to exclude the pandemic outlier
period from its main comparison figure.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import sys

sys.path.insert(0, "scripts")
from run_forecast_loop import rmsfe_table

MODELS = ["BVAR", "FactorAR", "Chronos", "TimesFM", "Moirai", "TTM", "LagLlama", "TimeGPT", "Hybrid_timesfm"]
LABELS = ["BVAR", "Factor", "Chronos", "TimesFM", "Moirai", "TTM", "LagLlama", "Time-GPT", "Hybrid"]
HORIZONS = [1, 3, 6, 12]


def make(errors_path="output/forecast_errors.parquet", out_path="output/rmsfe_boxplots.png"):
    errs = pd.read_parquet(errors_path)
    pre_covid = errs[errs.origin < "2019-06-01"]
    tbl = rmsfe_table(pre_covid)
    present = [m for m in MODELS if m in tbl.columns]
    labels = [LABELS[MODELS.index(m)] for m in present]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    for ax, h in zip(axes.flat, HORIZONS):
        sub = tbl[tbl.index.get_level_values("horizon") == h]
        data = [sub[m].dropna().values for m in present]
        ax.boxplot(data, tick_labels=labels, showfliers=False)
        ax.axhline(1.0, color="red", linestyle="--", linewidth=1)
        ax.set_title(f"Horizon = {h}")
        ax.set_ylabel("RMSFE ratio vs AR(1)")
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("RMSFE relative to AR(1) by model and horizon (pre-COVID origins, 2010-2019)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def demo():
    """ponytail: check the plot pipeline runs on synthetic data, not the real numbers."""
    import numpy as np
    rng = np.random.default_rng(0)
    rows = []
    for h in HORIZONS:
        for m in ["BVAR", "Chronos"]:
            for s in range(5):
                rows.append(dict(origin=pd.Timestamp("2015-01-01"), series=f"s{s}", horizon=h,
                                  model=m, sq_err=rng.random()))
                rows.append(dict(origin=pd.Timestamp("2015-01-01"), series=f"s{s}", horizon=h,
                                  model="AR1", sq_err=rng.random()))
    errs = pd.DataFrame(rows)
    errs.to_parquet("/tmp/_plot_demo_errs.parquet")
    make(errors_path="/tmp/_plot_demo_errs.parquet", out_path="/tmp/_plot_demo.png")
    import os
    assert os.path.exists("/tmp/_plot_demo.png")
    print("demo ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        demo()
    else:
        make()
