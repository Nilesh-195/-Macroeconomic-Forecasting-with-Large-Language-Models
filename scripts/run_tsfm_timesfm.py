"""
Phase 2: Google TimesFM 2.5 (200M), zero-shot univariate forecasting
with Monthly rolling forecast origins (freq="MS") across 120 series.
Run inside venv_core (PyTorch checkpoint).
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import timesfm

ROOT = Path(__file__).resolve().parent.parent
HORIZONS = [1, 3, 6, 12]
DEFAULT_START = "2015-01-01"
DEFAULT_END = "2021-12-01"


def load_model():
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(timesfm.ForecastConfig(
        max_context=1024, max_horizon=max(HORIZONS), per_core_batch_size=32,
    ))
    return model


def run(src="data/fredmd_clean_120.parquet",
        dst="output/forecasts_timesfm.parquet",
        origin_start=DEFAULT_START,
        origin_end=DEFAULT_END,
        freq="MS"):
    src_path = ROOT / src
    dst_path = ROOT / dst
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(src_path)
    origins = pd.date_range(origin_start, origin_end, freq=freq)
    print(f"Running TimesFM on {df.shape[1]} series across {len(origins)} monthly origins ({origins[0].date()} to {origins[-1].date()})...")

    model = load_model()

    rows = []
    for oi, t in enumerate(origins):
        hist = df.loc[:t]
        contexts, valid_cols = [], []
        for c in df.columns:
            ctx = hist[c].dropna().values
            if len(ctx) >= 10:
                contexts.append(ctx.astype(np.float32))
                valid_cols.append(c)
        if not contexts:
            continue
        point_forecast, _ = model.forecast(horizon=max(HORIZONS), inputs=contexts)  # (n_series, H)
        for col, preds in zip(valid_cols, point_forecast):
            for h in HORIZONS:
                rows.append(dict(origin=t, series=col, horizon=h, model="TimesFM", value=preds[h - 1]))
        if (oi + 1) % 6 == 0 or oi == len(origins) - 1:
            print(f"[{oi + 1}/{len(origins)}] TimesFM origin {t.date()} done")

    out = pd.DataFrame(rows)
    out.to_parquet(dst_path)
    print(f"Saved {len(out)} TimesFM forecast rows -> {dst_path}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Run TimesFM monthly forecast loop")
    parser.add_argument("--series", type=int, choices=[60, 120], default=120)
    parser.add_argument("--start", type=str, default=DEFAULT_START)
    parser.add_argument("--end", type=str, default=DEFAULT_END)
    parser.add_argument("--freq", type=str, default="MS")
    args = parser.parse_args()

    src_file = f"data/fredmd_clean_{args.series}.parquet"
    run(src=src_file, origin_start=args.start, origin_end=args.end, freq=args.freq)


if __name__ == "__main__":
    main()
