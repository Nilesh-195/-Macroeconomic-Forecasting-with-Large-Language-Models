"""
Step 3 of 3 — FinBERT-Augmented Hybrid Forecasting Engine
===========================================================
- Full 12-month horizon (h = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
- Macroeconomic context length: CONTEXT_LEN = 120 months (10 full business-cycle years)
- Input 1: Google TimesFM 2.5 (200M) zero-shot foundation model
- Input 2: Full FinBERT news factor representation (8 PCA factors + 3 sentiment channels)
- Architecture: Two-pass rolling origin learning via Ridge regularized meta-learner

Output:
  output/forecasts_finbert_hybrid.parquet
  output/rmsfe_finbert_summary.csv
"""

import warnings, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import timesfm

# ── Paths ─────────────────────────────────────────────────────────────────
FRED_FILE    = Path("data/fredmd_clean_60.parquet")
NEWS_FILE    = Path("data/news_factors.parquet")
OUT_FCST     = Path("output/forecasts_finbert_hybrid.parquet")
OUT_SUMMARY  = Path("output/rmsfe_finbert_summary.csv")
Path("output").mkdir(exist_ok=True)

# ── User-Specified Configuration ──────────────────────────────────────────
HORIZONS    = list(range(1, 13))  # 1 to 12 months ahead
CONTEXT_LEN = 120                 # 10 full years of historical macroeconomic context
MIN_TRAIN   = 48                  # rolling in-sample burn-in
N_ORIGINS   = 144                 # 144 monthly origins (2010 to 2021)

# ── Load FRED-MD ──────────────────────────────────────────────────────────
print("Loading 60 FRED-MD Macroeconomic Series...")
fred = pd.read_parquet(FRED_FILE)
if "date" not in fred.columns:
    fred = fred.reset_index().rename(columns={"sasdate": "date"})
fred["date"] = pd.to_datetime(fred["date"])
fred = fred.sort_values("date").reset_index(drop=True)
series_cols = [c for c in fred.columns if c != "date"]
print(f"  {len(series_cols)} series, {len(fred)} monthly observations.")

# ── Load Full FinBERT News Factors ────────────────────────────────────────
print("Loading Full FinBERT News Factors...")
news = pd.read_parquet(NEWS_FILE)
news["date"] = pd.to_datetime(news["date"])
news = news.sort_values("date").reset_index(drop=True)
news_feat_cols = [c for c in news.columns if c != "date"]
print(f"  {len(news_feat_cols)} FinBERT features across {len(news)} historical months.")

# ── Align Macroeconomic & News Data ───────────────────────────────────────
fred["_period"] = fred["date"].dt.to_period("M")
news["_period"] = news["date"].dt.to_period("M")
merged = fred.merge(news.drop(columns="date"), on="_period", how="left").drop(columns="_period")
merged[news_feat_cols] = merged[news_feat_cols].fillna(0.0)
print(f"  Merged dataset shape: {merged.shape}")

# ── Initialize TimesFM 2.5 ────────────────────────────────────────────────
print("\nInitializing Google TimesFM 2.5 (200M PyTorch)...")
tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
tfm.compile(timesfm.ForecastConfig(
    max_context=CONTEXT_LEN,
    max_horizon=max(HORIZONS),
    per_core_batch_size=32,
))
print(f"TimesFM compiled: max_context={CONTEXT_LEN}, max_horizon={max(HORIZONS)}.\n")

# ── AR(1) Benchmark Function ──────────────────────────────────────────────
def ar1_forecast(series: np.ndarray, h: int) -> float:
    if len(series) < 5:
        return float(series[-1])
    y = series[1:]
    x = add_constant(series[:-1])
    try:
        res = OLS(y, x).fit()
        c, phi = res.params[0], res.params[1]
        last = series[-1]
        # Iterate AR(1) forward h steps
        val = last
        for _ in range(h):
            val = c + phi * val
        return float(val)
    except Exception:
        return float(series[-1])

# ── Rolling Origins ───────────────────────────────────────────────────────
total_len     = len(merged)
total_origins = min(N_ORIGINS, total_len - CONTEXT_LEN - max(HORIZONS))
origin_indices = list(range(total_len - total_origins - max(HORIZONS),
                             total_len - max(HORIZONS)))

print(f"Rolling forecast evaluation: {len(origin_indices)} monthly origins.")
print(f"Forecast horizons: {HORIZONS} (1 through 12 months ahead)")
print(f"Total batches to forecast: {len(origin_indices)}\n")

# ══════════════════════════════════════════════════════════════════════════
# PASS 1: TimesFM Multi-Series Batch Inference
# ══════════════════════════════════════════════════════════════════════════
print("── PASS 1: Generating TimesFM 2.5 Zero-Shot Forecasts ──")
tsfm_store = {}  # (origin_pos, col) -> array of 12 horizon predictions

for oi, origin_pos in enumerate(origin_indices):
    contexts, valid_cols = [], []
    for col in series_cols:
        y = merged[col].values.astype(np.float32)
        ctx_start = max(0, origin_pos - CONTEXT_LEN + 1)
        ctx = np.nan_to_num(y[ctx_start: origin_pos + 1], nan=0.0)
        if len(ctx) >= 10:
            contexts.append(ctx)
            valid_cols.append(col)

    if not contexts:
        continue

    try:
        point_forecasts, _ = tfm.forecast(horizon=max(HORIZONS), inputs=contexts)
        for col, preds in zip(valid_cols, point_forecasts):
            tsfm_store[(origin_pos, col)] = np.array(preds[:max(HORIZONS)], dtype=np.float32)
    except Exception as e:
        print(f"  Error at origin {origin_pos}: {e}")

    if (oi + 1) % 20 == 0 or oi == 0 or oi == len(origin_indices) - 1:
        print(f"  [{oi+1:>3}/{len(origin_indices)}] Origin {merged['date'].iloc[origin_pos].strftime('%Y-%m')}")

print(f"\nPass 1 complete: {len(tsfm_store):,} (origin, series) forecasts generated.\n")

# ══════════════════════════════════════════════════════════════════════════
# PASS 2: Rolling Meta-Learner (TimesFM + FinBERT News Signal)
# ══════════════════════════════════════════════════════════════════════════
print("── PASS 2: Training FinBERT + TimesFM Meta-Learner across Horizons 1–12 ──")
all_rows = []
news_arr = merged[news_feat_cols].values.astype(np.float32)

for s_idx, col in enumerate(series_cols):
    y_full = merged[col].values.astype(np.float32)

    for h in HORIZONS:
        X_all, y_all, idx_all = [], [], []

        for origin_pos in origin_indices:
            if origin_pos + h >= len(y_full):
                continue
            key = (origin_pos, col)
            if key not in tsfm_store:
                continue
            yhat_tsfm = tsfm_store[key][h - 1]
            y_true    = y_full[origin_pos + h]
            if np.isnan(y_true) or np.isnan(yhat_tsfm):
                continue

            # Feature vector: [TimesFM baseline forecast, 11 FinBERT news features]
            feat = np.concatenate([[yhat_tsfm], news_arr[origin_pos]])
            X_all.append(feat)
            y_all.append(y_true)
            idx_all.append(origin_pos)

        if len(X_all) < MIN_TRAIN + 1:
            for i, origin_pos in enumerate(idx_all):
                key = (origin_pos, col)
                yhat_tsfm = float(tsfm_store[key][h - 1])
                y_true    = float(y_all[i])
                ar_ctx    = np.nan_to_num(y_full[max(0, origin_pos-CONTEXT_LEN): origin_pos+1], nan=0.0)
                yhat_ar1  = ar1_forecast(ar_ctx, h)
                all_rows.append({
                    "series": col, "origin": origin_pos, "horizon": h,
                    "y_true": y_true, "yhat_ar1": yhat_ar1,
                    "yhat_tsfm": yhat_tsfm, "yhat_finbert": yhat_tsfm,
                    "err_ar1": abs(y_true - yhat_ar1),
                    "err_tsfm": abs(y_true - yhat_tsfm),
                    "err_finbert": abs(y_true - yhat_tsfm),
                })
            continue

        X_all = np.array(X_all, dtype=np.float32)
        y_all = np.array(y_all, dtype=np.float32)

        # Rolling out-of-sample prediction
        for test_i in range(MIN_TRAIN, len(X_all)):
            origin_pos = idx_all[test_i]
            X_tr = X_all[:test_i]
            y_tr = y_all[:test_i]
            X_te = X_all[test_i:test_i+1]

            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)

            ridge = Ridge(alpha=10.0)
            ridge.fit(X_tr_s, y_tr)
            yhat_finbert = float(ridge.predict(X_te_s)[0])

            yhat_tsfm = float(X_all[test_i][0])
            y_true    = float(y_all[test_i])
            ar_ctx    = np.nan_to_num(y_full[max(0, origin_pos-CONTEXT_LEN): origin_pos+1], nan=0.0)
            yhat_ar1  = ar1_forecast(ar_ctx, h)

            all_rows.append({
                "series":       col,
                "origin":       origin_pos,
                "horizon":      h,
                "y_true":       y_true,
                "yhat_ar1":     yhat_ar1,
                "yhat_tsfm":    yhat_tsfm,
                "yhat_finbert": yhat_finbert,
                "err_ar1":      abs(y_true - yhat_ar1),
                "err_tsfm":     abs(y_true - yhat_tsfm),
                "err_finbert":  abs(y_true - yhat_finbert),
            })

    if (s_idx + 1) % 10 == 0 or s_idx == len(series_cols) - 1:
        print(f"  Series {s_idx+1:>2}/{len(series_cols)} processed ({len(all_rows):,} forecast evaluations)")

print(f"\nPass 2 complete. Total out-of-sample forecasts: {len(all_rows):,}")

# ── Save Forecasts ────────────────────────────────────────────────────────
df_out = pd.DataFrame(all_rows)
df_out.to_parquet(OUT_FCST, index=False)
print(f"Forecasts saved to: {OUT_FCST}")

# ── Compute Horizon-by-Horizon Relative RMSFE ──────────────────────────────
print("\n" + "="*70)
print("      FINAL BENCHMARK RESULTS: RELATIVE RMSFE (vs AR(1) Baseline)")
print("="*70)
print(f"{'Horizon':<10}{'AR(1)':<10}{'TimesFM':<14}{'FinBERT-Hybrid':<18}{'Advantage':<12}")
print("-"*70)

rows = []
for h in HORIZONS:
    sub = df_out[df_out["horizon"] == h]
    def rmsfe(col): return float(np.sqrt((sub[col] ** 2).mean()))

    rmsfe_ar1     = rmsfe("err_ar1")
    rmsfe_tsfm    = rmsfe("err_tsfm")
    rmsfe_finbert = rmsfe("err_finbert")
    rel_tsfm      = rmsfe_tsfm    / rmsfe_ar1
    rel_finbert   = rmsfe_finbert / rmsfe_ar1

    adv = "FinBERT ✅" if rel_finbert < rel_tsfm else "TimesFM"
    diff_pct = ((rel_tsfm - rel_finbert) / rel_tsfm) * 100

    print(f"h = {h:<7}1.0000    {rel_tsfm:<14.4f}{rel_finbert:<18.4f}{adv} ({diff_pct:+.1f}%)")

    rows.append({
        "horizon":            h,
        "RMSFE_AR1":          round(rmsfe_ar1, 4),
        "RMSFE_TimesFM":      round(rmsfe_tsfm, 4),
        "RMSFE_FinBERT":      round(rmsfe_finbert, 4),
        "Rel_TimesFM_vs_AR1": round(rel_tsfm, 4),
        "Rel_FinBERT_vs_AR1": round(rel_finbert, 4),
        "FinBERT_Gain_Pct":   round(diff_pct, 2),
    })

df_summary = pd.DataFrame(rows)
df_summary.to_csv(OUT_SUMMARY, index=False)
print("-"*70)
print(f"Summary saved to: {OUT_SUMMARY}\n")
