"""Stage 1, Section 6: TTM (IBM Tiny Time Mixer), zero-shot, univariate.
Run inside venv_ttm (granite-tsfm from GitHub, Python 3.11 -- the PyPI
"granite-tsfm" stub package is empty; the real toolkit needs <3.14).
"""
import numpy as np
import pandas as pd
import torch
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

HORIZONS = [1, 3, 6, 12]
ORIGIN_START = "2010-01-01"
ORIGIN_END = "2021-12-01"
MODEL_NAME = "ibm-granite/granite-timeseries-ttm-r2"  # ~5M params
CONTEXT_LEN = 512  # fixed context/prediction combo this checkpoint was trained on


def load_model():
    model = TinyTimeMixerForPrediction.from_pretrained(MODEL_NAME, num_input_channels=1)
    model.eval()
    return model


def forecast_one(model, series: np.ndarray):
    ctx = series[-CONTEXT_LEN:]
    if len(ctx) < CONTEXT_LEN:
        ctx = np.pad(ctx, (CONTEXT_LEN - len(ctx), 0), mode="edge")
    mean, std = ctx.mean(), ctx.std() or 1.0
    x = torch.tensor(((ctx - mean) / std).reshape(1, CONTEXT_LEN, 1), dtype=torch.float32)
    with torch.no_grad():
        out = model(past_values=x).prediction_outputs  # (1, pred_len, 1)
    return out[0, :, 0].numpy() * std + mean


def run(src="data/fredmd_clean.parquet", dst="output/forecasts_ttm.parquet"):
    df = pd.read_parquet(src)
    origins = pd.date_range(ORIGIN_START, ORIGIN_END, freq="3MS")
    model = load_model()

    rows = []
    for oi, t in enumerate(origins):
        hist = df.loc[:t]
        for c in df.columns:
            ctx = hist[c].dropna().values
            if len(ctx) < 10:
                continue
            preds = forecast_one(model, ctx)
            for h in HORIZONS:
                if h <= len(preds):
                    rows.append(dict(origin=t, series=c, horizon=h, model="TTM", value=preds[h - 1]))
        print(f"[{oi + 1}/{len(origins)}] origin {t.date()} done")

    out = pd.DataFrame(rows)
    out.to_parquet(dst)
    print(f"saved {len(out)} forecast rows -> {dst}")
    return out


def demo():
    model = load_model()
    ctx = np.sin(np.arange(600) / 3.0).astype(np.float32)
    preds = forecast_one(model, ctx)
    assert len(preds) >= max(HORIZONS)
    print("demo ok")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        demo()
    else:
        run()
