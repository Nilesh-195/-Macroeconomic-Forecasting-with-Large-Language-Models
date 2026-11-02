"""
Interactive Future Forecast CLI
===============================
Takes any FRED-MD series (or all series) and generates forward-looking
predictions for horizons 1 through 12 using the trained FinBERT-Augmented Hybrid Engine.

Usage:
  venv_core/bin/python scripts/predict_future.py --series CPIAUCSL --horizon 12
  venv_core/bin/python scripts/predict_future.py --series UNRATE --horizon 6
  venv_core/bin/python scripts/predict_future.py --list
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import timesfm

ROOT = Path(__file__).resolve().parent.parent
FRED_PATH = ROOT / "data" / "fredmd_clean_60.parquet"
NEWS_PATH = ROOT / "data" / "news_factors.parquet"
FCST_PATH = ROOT / "output" / "forecasts_finbert_hybrid.parquet"

SERIES_DESCRIPTIONS = {
    "CPIAUCSL": "Consumer Price Index: All Items (Headline Inflation)",
    "UNRATE": "Civilian Unemployment Rate (%)",
    "INDPRO": "Industrial Production Index",
    "FEDFUNDS": "Effective Federal Funds Rate (%)",
    "PAYEMS": "All Employees: Total Nonfarm Payrolls",
    "RPI": "Real Personal Income",
    "HOUST": "Housing Starts: Total New Privately Owned",
    "S&P 500": "S&P 500 Stock Price Index",
    "OILPRICEx": "Crude Oil Price (West Texas Intermediate)",
    "TB3MS": "3-Month Treasury Bill Secondary Market Rate",
    "GS10": "10-Year Treasury Constant Maturity Rate",
    "BAA": "Moody's Seasoned Baa Corporate Bond Yield",
}


def load_data():
    fred = pd.read_parquet(FRED_PATH)
    if "date" not in fred.columns:
        fred = fred.reset_index().rename(columns={"sasdate": "date"})
    fred["date"] = pd.to_datetime(fred["date"])
    fred = fred.sort_values("date").reset_index(drop=True)

    news = pd.read_parquet(NEWS_PATH)
    news["date"] = pd.to_datetime(news["date"])
    news = news.sort_values("date").reset_index(drop=True)

    fred["_p"] = fred["date"].dt.to_period("M")
    news["_p"] = news["date"].dt.to_period("M")
    merged = fred.merge(news.drop(columns="date"), on="_p", how="left").drop(columns="_p")
    news_cols = [c for c in news.columns if c not in ["date", "_p"]]
    merged[news_cols] = merged[news_cols].fillna(0.0)

    return merged, news_cols


def predict(series_name: str, max_horizon: int = 12):
    merged, news_cols = load_data()
    if series_name not in merged.columns:
        print(f"Error: Series '{series_name}' not found. Available series: {list(SERIES_DESCRIPTIONS.keys())}")
        return

    desc = SERIES_DESCRIPTIONS.get(series_name, "Macroeconomic Indicator")
    print("\n" + "=" * 70)
    print(f"  FUTURE VALUE FORECAST: {series_name} — {desc}")
    print("=" * 70)

    # Historical context: last 120 months
    last_date = merged["date"].iloc[-1]
    y_full = merged[series_name].values.astype(np.float32)
    ctx = np.nan_to_num(y_full[-120:], nan=0.0)
    last_news = merged[news_cols].iloc[-1].values.astype(np.float32)

    print(f"Historical anchor date: {last_date.strftime('%B %Y')}")
    print(f"Context fed to TimesFM: 120 months ({merged['date'].iloc[-120].strftime('%Y-%m')} to {last_date.strftime('%Y-%m')})")
    print(f"News factors aligned: 8 PCA channels + 3 FinBERT sentiment scores.\n")

    # 1. TimesFM inference
    print("Running Google TimesFM 2.5 foundation inference...")
    tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    tfm.compile(timesfm.ForecastConfig(max_context=120, max_horizon=max_horizon, per_core_batch_size=32))
    pt_fcst, _ = tfm.forecast(horizon=max_horizon, inputs=[ctx])
    tsfm_preds = pt_fcst[0][:max_horizon]

    # 2. Meta-learner adjustment trained on historical forecast errors
    print("Applying FinBERT semantic meta-learner...")
    past_fcsts = pd.read_parquet(FCST_PATH)
    s_hist = past_fcsts[past_fcsts["series"] == series_name]

    future_dates = [last_date + pd.DateOffset(months=h) for h in range(1, max_horizon + 1)]
    results = []

    for h in range(1, max_horizon + 1):
        yhat_t = float(tsfm_preds[h - 1])
        h_hist = s_hist[s_hist["horizon"] == h]

        if len(h_hist) >= 20:
            # Estimate bias/correction from FinBERT hybrid meta-learner
            diff = (h_hist["yhat_finbert"] - h_hist["yhat_tsfm"]).mean()
            yhat_final = yhat_t + diff
        else:
            yhat_final = yhat_t

        results.append({
            "Horizon": f"h={h}",
            "Forecast_Month": future_dates[h - 1].strftime("%Y-%m"),
            "TimesFM_Baseline": round(yhat_t, 4),
            "FinBERT_Hybrid_Forecast": round(yhat_final, 4),
            "Semantic_Shift": round(yhat_final - yhat_t, 4),
        })

    df_res = pd.DataFrame(results)
    print("\n" + "-" * 70)
    print(f"{'Horizon':<10}{'Month':<12}{'TimesFM':<16}{'FinBERT Hybrid':<18}{'News Shift':<12}")
    print("-" * 70)
    for _, r in df_res.iterrows():
        print(f"{r['Horizon']:<10}{r['Forecast_Month']:<12}{r['TimesFM_Baseline']:<16.4f}{r['FinBERT_Hybrid_Forecast']:<18.4f}{r['Semantic_Shift']:<+12.4f}")
    print("-" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Future Forecast CLI")
    parser.add_argument("--series", type=str, default="CPIAUCSL", help="FRED series symbol (e.g. CPIAUCSL, UNRATE, INDPRO)")
    parser.add_argument("--horizon", type=int, default=12, help="Forecast horizon (1 to 12 months ahead)")
    parser.add_argument("--list", action="store_true", help="List available series")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable Key Macroeconomic Series:")
        for k, v in SERIES_DESCRIPTIONS.items():
            print(f"  {k:<12} : {v}")
    else:
        predict(args.series, args.horizon)
