"""Stage 1, Section 6: Chronos (Amazon), zero-shot, univariate.
Run inside venv_core (has torch + chronos-forecasting installed separately
from the econometric env's plain pip packages, per the guide's per-model
env isolation -- avoids dependency clashes with the other TSFMs later).

Uses Chronos-Bolt (direct multi-quantile regression), not the original
T5 autoregressive-sampling Chronos: chronos-t5-base took ~13 min/origin
on CPU (10+ hours for the full loop) since it samples token-by-token;
Bolt gives quantiles in a single forward pass and is Amazon's own
benchmarked-faster successor at the same size class, not a scope cut.
"""
import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline

HORIZONS = [1, 3, 6, 12]
ORIGIN_START = "2010-01-01"
ORIGIN_END = "2021-12-01"
MODEL_NAME = "amazon/chronos-bolt-small"  # ~48M params


def load_pipe():
    pipe = BaseChronosPipeline.from_pretrained(MODEL_NAME, device_map="cpu", torch_dtype=torch.float32)
    return pipe, pipe.quantiles.index(0.5)


def run(src="data/fredmd_clean.parquet", dst="output/forecasts_chronos.parquet"):
    df = pd.read_parquet(src)
    origins = pd.date_range(ORIGIN_START, ORIGIN_END, freq="3MS")
    pipe, median_idx = load_pipe()

    rows = []
    for oi, t in enumerate(origins):
        hist = df.loc[:t]
        contexts = [torch.tensor(hist[c].dropna().values, dtype=torch.float32) for c in df.columns]
        valid_cols = [c for c, ctx in zip(df.columns, contexts) if len(ctx) >= 10]
        contexts = [ctx for ctx in contexts if len(ctx) >= 10]
        if not contexts:
            continue
        forecast = pipe.predict(contexts, prediction_length=max(HORIZONS))  # (n_series, n_quantiles, H)
        median = forecast[:, median_idx, :].numpy()  # (n_series, H)
        for col, preds in zip(valid_cols, median):
            for h in HORIZONS:
                rows.append(dict(origin=t, series=col, horizon=h, model="Chronos", value=preds[h - 1]))
        print(f"[{oi + 1}/{len(origins)}] origin {t.date()} done")

    out = pd.DataFrame(rows)
    out.to_parquet(dst)
    print(f"saved {len(out)} forecast rows -> {dst}")
    return out


def demo():
    """ponytail: one series, one origin, sanity check only -- not the full loop."""
    idx = pd.date_range("2000-01-01", periods=60, freq="MS")
    df = pd.DataFrame({"a": np.sin(np.arange(60) / 3.0)}, index=idx)
    pipe, median_idx = load_pipe()
    ctx = torch.tensor(df["a"].values, dtype=torch.float32)
    fc = pipe.predict([ctx], prediction_length=12)
    assert fc.shape[0] == 1 and fc.shape[-1] == 12
    print("demo ok")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        demo()
    else:
        run()
