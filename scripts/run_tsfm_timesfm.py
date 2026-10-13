"""Stage 1, Section 6: TimesFM (Google), zero-shot, univariate.
Run inside venv_core (PyTorch checkpoint, per guide -- avoids jax-metal).
"""
import numpy as np
import pandas as pd
import timesfm

HORIZONS = [1, 3, 6, 12]
ORIGIN_START = "2010-01-01"
ORIGIN_END = "2021-12-01"


def load_model():
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(timesfm.ForecastConfig(
        max_context=1024, max_horizon=max(HORIZONS), per_core_batch_size=32,
    ))
    return model


def run(src="data/fredmd_clean.parquet", dst="output/forecasts_timesfm.parquet"):
    df = pd.read_parquet(src)
    origins = pd.date_range(ORIGIN_START, ORIGIN_END, freq="3MS")
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
        print(f"[{oi + 1}/{len(origins)}] origin {t.date()} done")

    out = pd.DataFrame(rows)
    out.to_parquet(dst)
    print(f"saved {len(out)} forecast rows -> {dst}")
    return out


def demo():
    idx = pd.date_range("2000-01-01", periods=60, freq="MS")
    ctx = np.sin(np.arange(60) / 3.0).astype(np.float32)
    model = load_model()
    point_forecast, _ = model.forecast(horizon=max(HORIZONS), inputs=[ctx])
    assert point_forecast.shape == (1, max(HORIZONS))
    print("demo ok")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        demo()
    else:
        run()
