"""Stage 1, Section 6: Time-GPT (Nixtla), zero-shot, via API.
Run inside venv_core. Requires NIXTLA_API_KEY env var (free tier at
dashboard.nixtla.io) -- the guide's only model needing network access
per forecast rather than local compute; watch your quota.
"""
import os
import pandas as pd
from nixtla import NixtlaClient

HORIZONS = [1, 3, 6, 12]
ORIGIN_START = "2010-01-01"
ORIGIN_END = "2021-12-01"


def load_client():
    key = os.environ.get("NIXTLA_API_KEY")
    if not key:
        raise SystemExit("Set NIXTLA_API_KEY first: export NIXTLA_API_KEY=your_key_here")
    return NixtlaClient(api_key=key)


def run(src="data/fredmd_clean.parquet", dst="output/forecasts_timegpt.parquet"):
    df = pd.read_parquet(src)
    origins = pd.date_range(ORIGIN_START, ORIGIN_END, freq="3MS")
    client = load_client()

    rows = []
    for oi, t in enumerate(origins):
        hist = df.loc[:t]
        # one API call per origin covering all series at once (long-format,
        # ragged per-series history is fine) instead of one call per series --
        # cuts 2880 calls down to 48, comfortably under the 150/min rate limit
        frames = []
        for c in df.columns:
            ctx = hist[c].dropna()
            if len(ctx) >= 10:
                frames.append(pd.DataFrame({"unique_id": c, "ds": ctx.index, "y": ctx.values}))
        if not frames:
            continue
        long_df = pd.concat(frames, ignore_index=True)
        fc = client.forecast(df=long_df, h=max(HORIZONS), freq="MS")
        for c, group in fc.groupby("unique_id"):
            preds = group["TimeGPT"].values
            for h in HORIZONS:
                rows.append(dict(origin=t, series=c, horizon=h, model="TimeGPT", value=preds[h - 1]))
        print(f"[{oi + 1}/{len(origins)}] origin {t.date()} done")

    out = pd.DataFrame(rows)
    out.to_parquet(dst)
    print(f"saved {len(out)} forecast rows -> {dst}")
    return out


def demo():
    """ponytail: one series, one call, checks the API key/quota work at all."""
    idx = pd.date_range("2000-01-01", periods=60, freq="MS")
    import numpy as np
    long_df = pd.DataFrame({"unique_id": "a", "ds": idx, "y": np.sin(np.arange(60) / 3.0)})
    client = load_client()
    fc = client.forecast(df=long_df, h=12, freq="MS")
    assert len(fc) == 12
    print("demo ok")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        demo()
    else:
        run()
